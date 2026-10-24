import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from benchmark_kv_cache import GPT_CONFIG, tokens_per_sec
from gpt import GPTModel

# --- design tokens (light mode) -------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CACHE_BLUE = "#2a78d6"                             # the series the story is about
NO_CACHE_GRAY = MUTED                              # de-emphasised baseline
BATCH_RAMP = {1: "#86b6ef", 8: "#2a78d6", 32: "#104281"}   # ordinal: light -> dark = small -> large batch

L, D, V = GPT_CONFIG["n_layers"], GPT_CONFIG["emb_dim"], GPT_CONFIG["vocab_size"]
CTX = GPT_CONFIG["context_length"]
BYTES = 4                                          # fp32


# --- cost model -----------------------------------------------------------------
# Matmul FLOPs (2 per multiply-add), batch 1. Per token: qkv 6d^2 + out 2d^2 + ffn 16d^2 per layer, plus the lm head.
# Attention: QK^T and AV cost 4*d per (query, key) pair per layer. Checked against torch FlopCounterMode.
PER_TOKEN = L * 24 * D * D + 2 * D * V

def step_flops(T, q):
    """FLOPs for one forward pass with q query tokens attending over T keys."""
    return q * PER_TOKEN + L * 4 * D * q * T

def generation_flops(prompt_len, num_new):
    """Per-step FLOPs (arrays of length num_new) for generating with / without the KV cache."""
    T = prompt_len + np.arange(num_new)            # sequence length at each step
    no_cache = step_flops(T, T)                    # whole sequence re-run every step
    cache = step_flops(T, np.ones_like(T))         # one new token per step...
    cache[0] = step_flops(T[0], T[0])              # ...after the prompt is prefilled
    return T, no_cache, cache

KV_BYTES_PER_TOKEN = 2 * L * D * BYTES             # K and V, every layer, one sequence


# --- chart helpers --------------------------------------------------------------
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=9, length=0, pad=6)

def titles(ax, headline, subtitle):
    ax.set_title(headline, loc="left", fontsize=12, fontweight="semibold", color=INK, pad=28)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")

def end_dot(ax, x, y, color):
    ax.plot([x], [y], "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=5, clip_on=False)

def plain_number(v, _):
    return f"{v:g}"

def column(ax, x, height, width, color, radius_px=4):
    """Column with a rounded data-end and a square baseline (the rounded foot is clipped off by the axes)."""
    bbox = ax.get_window_extent()
    x_per_px = np.diff(ax.get_xlim())[0] / bbox.width
    y_per_px = np.diff(ax.get_ylim())[0] / bbox.height
    foot = 3 * radius_px * y_per_px
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, -foot), width, height + foot,
        boxstyle=f"round,pad=0,rounding_size={radius_px * x_per_px}",
        mutation_aspect=y_per_px / x_per_px, fc=color, ec="none", zorder=3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_cache_tradeoffs.png"))
    parser.add_argument("--bench-tokens", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    args = parser.parse_args()

    # measured: tokens/sec on this machine
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG).eval()
    idx = torch.tensor(tiktoken.get_encoding("gpt2").encode("welcome to my world")).unsqueeze(0)
    P = idx.shape[1]
    print("measuring tokens/sec ...")
    tps = {use_cache: tokens_per_sec(model, idx, args.bench_tokens, use_cache) for use_cache in (False, True)}
    weights_gib = sum(p.numel() for p in model.parameters()) * BYTES / 2**30

    # computed: FLOPs and cache size
    N = args.max_new_tokens
    T, step_no, step_yes = generation_flops(P, N)
    new_tokens = np.arange(1, N + 1)
    cum_no, cum_yes = np.cumsum(step_no), np.cumsum(step_yes)
    flop_cut_at_bench = cum_no[args.bench_tokens - 1] / cum_yes[args.bench_tokens - 1]
    speedup = tps[True] / tps[False]
    gib_per_token = KV_BYTES_PER_TOKEN / 2**30
    b32_gib = gib_per_token * CTX * 32

    # --- table view (every plotted value is also printed) ---
    print(f"\n{'new tokens':>10} {'no cache TFLOPs':>16} {'cache TFLOPs':>13} {'saved':>7}")
    for n in (10, 100, 500, N):
        print(f"{n:>10} {cum_no[n-1]/1e12:>16.3f} {cum_yes[n-1]/1e12:>13.3f} {cum_no[n-1]/cum_yes[n-1]:>6.0f}x")
    print(f"\nKV cache: {KV_BYTES_PER_TOKEN/1024:.0f} KiB/token/sequence; weights {weights_gib*1024:.0f} MiB")
    for b in BATCH_RAMP:
        print(f"  batch {b:>2} @ {CTX} tokens: {gib_per_token*CTX*b*1024:,.0f} MiB")
    print(f"\ntokens/sec ({args.bench_tokens} new tokens): no cache {tps[False]:.1f}, cache {tps[True]:.1f} -> {speedup:.1f}x")

    # --- figure ---
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), facecolor=SURFACE, layout="constrained")
    ax_cum, ax_step, ax_mem, ax_speed = axes.flat
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))   # leave a strip at the bottom for the footer
    for ax in axes.flat:
        style_axes(ax)

    # 1. cumulative compute
    titles(ax_cum, f"Compute: {cum_no[-1]/cum_yes[-1]:.0f}× fewer FLOPs over {N:,} tokens",
           f"Cumulative matmul TFLOPs to generate N new tokens ({P}-token prompt, batch 1, log scale)")
    ax_cum.plot(new_tokens, cum_no / 1e12, color=NO_CACHE_GRAY, lw=2, solid_capstyle="round", label="No cache")
    ax_cum.plot(new_tokens, cum_yes / 1e12, color=CACHE_BLUE, lw=2, solid_capstyle="round", label="KV cache")
    end_dot(ax_cum, N, cum_no[-1] / 1e12, NO_CACHE_GRAY)
    end_dot(ax_cum, N, cum_yes[-1] / 1e12, CACHE_BLUE)
    ax_cum.set_yscale("log")
    ax_cum.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax_cum.set_xlim(0, N * 1.28)
    ax_cum.set_xticks([0, 250, 500, 750, 1000])
    ax_cum.set_xlabel("new tokens generated", color=INK2, fontsize=9)
    ax_cum.annotate(f"{cum_no[-1]/1e12:.1f} TFLOPs", (N, cum_no[-1] / 1e12), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax_cum.annotate(f"{cum_yes[-1]/1e12:.2f} TFLOPs", (N, cum_yes[-1] / 1e12), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax_cum.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)

    # 2. per-step compute
    ratio_step = step_no[-1] / step_yes[-1]
    titles(ax_step, "Per step: the cache keeps cost flat as the sequence grows",
           f"GFLOPs for one decode step at sequence length T (log scale); at T={T[-1]:,} that is {ratio_step:,.0f}× apart")
    ax_step.plot(T[1:], step_no[1:] / 1e9, color=NO_CACHE_GRAY, lw=2, solid_capstyle="round", label="No cache")
    ax_step.plot(T[1:], step_yes[1:] / 1e9, color=CACHE_BLUE, lw=2, solid_capstyle="round", label="KV cache")
    end_dot(ax_step, T[-1], step_no[-1] / 1e9, NO_CACHE_GRAY)
    end_dot(ax_step, T[-1], step_yes[-1] / 1e9, CACHE_BLUE)
    ax_step.set_yscale("log")
    ax_step.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax_step.set_xlim(0, T[-1] * 1.28)
    ax_step.set_xticks([0, 250, 500, 750, 1000])
    ax_step.set_xlabel("sequence length T (tokens)", color=INK2, fontsize=9)
    ax_step.annotate(f"{step_no[-1]/1e9:.0f} GFLOPs", (T[-1], step_no[-1] / 1e9), xytext=(10, 0),
                     textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax_step.annotate(f"{step_yes[-1]/1e9:.2f} GFLOPs", (T[-1], step_yes[-1] / 1e9), xytext=(10, 0),
                     textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax_step.legend(loc="center", bbox_to_anchor=(0.4, 0.3), frameon=False, fontsize=9, labelcolor=INK2)   # gap between the two lines

    # 3. cache memory
    titles(ax_mem, f"Memory: {b32_gib:.2f} GiB of cache at batch 32, {b32_gib/weights_gib:.1f}× the weights",
           f"KV cache size vs. sequence length ({KV_BYTES_PER_TOKEN/1024:.0f} KiB per token per sequence, fp32)")
    seq = np.arange(0, CTX + 1)
    for b, color in BATCH_RAMP.items():
        gib = gib_per_token * seq * b
        ax_mem.plot(seq, gib, color=color, lw=2, solid_capstyle="round", label=f"batch {b}")
        end_dot(ax_mem, CTX, gib[-1], color)
        ax_mem.annotate(f"{gib[-1]:.2f} GiB" if gib[-1] >= 1 else f"{gib[-1]*1024:.0f} MiB", (CTX, gib[-1]),
                        xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax_mem.axhline(weights_gib, color=AXIS, lw=1, zorder=1)
    ax_mem.text(CTX, weights_gib + 0.04, f"model weights {weights_gib*1024:.0f} MiB", fontsize=9, color=INK2,
                ha="right", va="bottom")
    ax_mem.set_xlim(0, CTX * 1.28)
    ax_mem.set_ylim(0, b32_gib * 1.08)
    ax_mem.set_xticks([0, 256, 512, 768, 1024])
    ax_mem.set_xlabel("sequence length (tokens)", color=INK2, fontsize=9)
    ax_mem.set_ylabel("GiB", color=INK2, fontsize=9)
    ax_mem.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)

    # 4. measured speed
    titles(ax_speed, f"Speed: {speedup:.1f}× faster, from a {flop_cut_at_bench:.0f}× cut in FLOPs",
           f"Measured tokens/sec on this machine ({args.bench_tokens} new tokens, CPU, after warm-up)")
    ax_speed.set_xlim(-0.6, 1.6)
    ax_speed.set_ylim(0, max(tps.values()) * 1.18)
    ax_speed.set_xticks([0, 1], ["No cache", "KV cache"])
    ax_speed.tick_params(axis="x", labelsize=10, colors=INK2)
    ax_speed.set_ylabel("tokens / sec", color=INK2, fontsize=9)
    fig.canvas.draw()                              # fix layout so pixel-based corner radii are right
    for x, use_cache, color in ((0, False, NO_CACHE_GRAY), (1, True, CACHE_BLUE)):
        column(ax_speed, x, tps[use_cache], 0.2, color)
        ax_speed.text(x, tps[use_cache] + max(tps.values()) * 0.02, f"{tps[use_cache]:.1f}",
                      ha="center", va="bottom", fontsize=10, color=INK)

    fig.suptitle("KV caching is not a free lunch", x=0.012, ha="left", fontsize=17, fontweight="semibold", color=INK)
    fig.text(0.012, 0.005,
             f"GPT-style model: {L} layers, d={D}, vocab {V:,}, fp32 weights {weights_gib*1024:.0f} MiB. "
             "FLOPs and cache size are computed (FLOPs checked against torch FlopCounterMode); speed is measured.",
             fontsize=8.5, color=MUTED, ha="left", va="bottom")
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
