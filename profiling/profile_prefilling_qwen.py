import functools
import os
import argparse
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
print(transformers.__version__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=1)
    p.add_argument("--attn_implementation", type=str, default="eager")
    p.add_argument("--use_generate", action="store_true")
    p.add_argument("--trace_dir", default=os.path.join(SCRIPT_DIR, "traces/llm"))
    p.add_argument("--sweep", action="store_true",
                    help="Run the prefill seq_len sweep instead of a single-shot profile.")
    p.add_argument("--seq_lens", type=str, default="16,64,256,1024,4096",
                    help="Comma-separated exact token counts to sweep, for --sweep.")
    return p.parse_args()


def wrap_forward_with_record_function(module, name):
    """Wrap module.forward so every call is nested under a named
    record_function region in the profiler trace. This is what makes
    "attention CUDA time" vs "MLP CUDA time" attributable at all: both
    submodules dispatch to the same aten::linear/addmm ops and the same
    cutlass GEMM kernels, so op/kernel names alone can't tell them apart.
    Structural scoping (this) is the only reliable way to split them.
    """
    orig_forward = module.forward

    @functools.wraps(orig_forward)
    def wrapped(*call_args, **call_kwargs):
        with torch.profiler.record_function(name):
            return orig_forward(*call_args, **call_kwargs)

    module.forward = wrapped


def instrument_attn_and_mlp(model):
    for layer in model.model.layers:
        wrap_forward_with_record_function(layer.self_attn, "attn_block")
        wrap_forward_with_record_function(layer.mlp, "mlp_block")


def make_synthetic_inputs(vocab_size, seq_len, device):
    # Exact token-count control, bypassing word/phrase-count ambiguity entirely
    # (tokenizer merging at repeated-phrase boundaries means "N words" is not a
    # precise proxy for seq_len). Same "only the shape matters for compute
    # cost" philosophy as one.py/two.py using torch.randn for synthetic
    # tensors - real text content doesn't affect GEMM/attention FLOPs, only
    # the shape does.
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def measure_wall_time_ms(step_fn, iters=10):
    # Deliberately un-profiled: the profiler itself has real overhead (CUPTI
    # buffer requests, event recording), so wall-clock/tokens-per-sec numbers
    # come from a plain timed loop, not from the profiled pass.
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step_fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def profile_breakdown(step_fn, active=3):
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=active, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(2 + active):
            step_fn()
            prof.step()
    torch.cuda.synchronize()

    events = prof.key_averages()
    # record_function scopes (ProfilerStep*, attn_block, mlp_block, ...) are
    # marker/container events, not real work - their self time is a profiler
    # bookkeeping artifact, not compute. Excluding is_user_annotation events
    # is what the printed "Self CPU/CUDA time total" table lines do
    # internally; summing self_device_time_total blindly over ALL events
    # (including markers) massively over-counts.
    self_cpu_us = sum(e.self_cpu_time_total for e in events if not e.is_user_annotation)
    self_cuda_us = sum(e.self_device_time_total for e in events if not e.is_user_annotation)

    # Each record_function scope shows up TWICE in key_averages(): once as a
    # DeviceType.CPU entry (device_time_total = real correlated child-kernel
    # execution time, what we want) and once as a DeviceType.CUDA "device
    # timeline span" entry (spans from first-kernel-start to last-kernel-end,
    # including any GPU idle gaps in between - not actual compute time, and
    # much larger). Must filter to the CPU-side entry only, or this double/
    # over-counts exactly like the ProfilerStep* case above.
    def scope_device_us(name):
        return sum(
            e.device_time_total for e in events
            if e.key == name and str(e.device_type) == "DeviceType.CPU"
        )

    attn_cuda_us = scope_device_us("attn_block")
    mlp_cuda_us = scope_device_us("mlp_block")

    return {
        "self_cpu_ms": self_cpu_us / active / 1000,
        "self_cuda_ms": self_cuda_us / active / 1000,
        "attn_cuda_ms": attn_cuda_us / active / 1000,
        "mlp_cuda_ms": mlp_cuda_us / active / 1000,
    }


def run_sweep(model, tokenizer, seq_lens, warmup_iters=3, timed_iters=10):
    vocab_size = model.config.vocab_size
    rows = []
    for seq_len in seq_lens:
        try:
            model_inputs = make_synthetic_inputs(vocab_size, seq_len, model.device)

            def step():
                with torch.no_grad():
                    return model(**model_inputs)

            for _ in range(warmup_iters):
                step()
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            wall_ms = measure_wall_time_ms(step, iters=timed_iters)
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            breakdown = profile_breakdown(step)

            tokens_per_sec = seq_len / (wall_ms / 1000)
            row = {
                "seq_len": seq_len,
                "wall_ms": wall_ms,
                "tokens_per_sec": tokens_per_sec,
                "peak_mem_mb": peak_mem_mb,
                **breakdown,
            }
            rows.append(row)
            print(f"seq_len={seq_len:6d}  wall_ms={wall_ms:9.3f}  tok/s={tokens_per_sec:10.1f}  "
                  f"self_cpu_ms={breakdown['self_cpu_ms']:8.3f}  self_cuda_ms={breakdown['self_cuda_ms']:8.3f}  "
                  f"attn_cuda_ms={breakdown['attn_cuda_ms']:8.3f}  mlp_cuda_ms={breakdown['mlp_cuda_ms']:8.3f}  "
                  f"peak_mem_mb={peak_mem_mb:9.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"seq_len={seq_len}: OOM, stopping sweep here.")
            torch.cuda.empty_cache()
            break
    return rows


def plot_sweep(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    sns.lineplot(data=df, x="seq_len", y="tokens_per_sec", marker="o", ax=ax)
    ax.set_xscale("log", base=2)
    ax.set_title("Prefill throughput vs seq_len")
    ax.set_ylabel("tokens/sec")

    ax = axes[0, 1]
    cpu_cuda = df.melt(id_vars="seq_len", value_vars=["self_cpu_ms", "self_cuda_ms"],
                        var_name="component", value_name="ms")
    sns.lineplot(data=cpu_cuda, x="seq_len", y="ms", hue="component", marker="o", ax=ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Self CPU vs Self CUDA time (overhead-bound -> GPU-bound)")
    ax.set_ylabel("ms")

    ax = axes[1, 0]
    attn_mlp = df.melt(id_vars="seq_len", value_vars=["attn_cuda_ms", "mlp_cuda_ms"],
                        var_name="component", value_name="ms")
    sns.lineplot(data=attn_mlp, x="seq_len", y="ms", hue="component", marker="o", ax=ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_title("Attention vs MLP CUDA time (log-log: watch the slopes)")
    ax.set_ylabel("ms")

    ax = axes[1, 1]
    sns.lineplot(data=df, x="seq_len", y="peak_mem_mb", marker="o", ax=ax)
    ax.set_xscale("log", base=2)
    ax.set_title("Peak GPU memory vs seq_len")
    ax.set_ylabel("MB")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved plot: {out_path}")


def main():
    args = parse_args()
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype="auto",
        attn_implementation=args.attn_implementation,
    ).to(device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    os.makedirs(args.trace_dir, exist_ok=True)

    if args.sweep:
        instrument_attn_and_mlp(model)
        seq_lens = [int(s) for s in args.seq_lens.split(",")]
        rows = run_sweep(model, tokenizer, seq_lens)

        import csv
        csv_path = os.path.join(args.trace_dir, "prefill_sweep.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved sweep results: {csv_path}")

        plot_sweep(rows, os.path.join(args.trace_dir, "prefill_sweep.png"))
        return

    prompt = "what is your name? " * args.N
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f"model inputs: {model_inputs}")

    def step():
        # --use_generate reproduces the WRONG way to profile prefill, kept only for
        # an A/B comparison against the direct-call path: model.generate(), even
        # with max_new_tokens=1, wraps the forward pass in heavy Python-side
        # orchestration (generation config validation, KV cache object init,
        # logits-processor pipeline, stopping-criteria checks, sampling/argmax) and
        # all of that gets captured inside this record_function scope too,
        # contaminating "prefill" with overhead that has nothing to do with the
        # model's actual op dispatch chain. The default (False) path is a direct
        # call: exactly one forward pass over the full prompt, the real prefill,
        # nothing else.
        with torch.profiler.record_function("prefill_llm"), torch.no_grad():
            if args.use_generate:
                return model.generate(**model_inputs, max_new_tokens=1)
            return model(**model_inputs)

    run_tag = "generate_wrong" if args.use_generate else "direct"
    table_path = os.path.join(args.trace_dir, f"llm_{run_tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"llm_{run_tag}.json")

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            step()
            prof.step()
    torch.cuda.synchronize()
    print(f"saving traces: {trace_path}")
    prof.export_chrome_trace(trace_path)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    print("table: ", table)
    with open(table_path, "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()
