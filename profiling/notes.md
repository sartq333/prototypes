# Profiling notes

## Overhead-bound matmul (64x64, bf16, eager, cold)

From `traces/matrix_multiplication/64_bf16_cold_eager.txt`:

- Self CPU time total: 325.940us
- Self CUDA time total: 6.976us

The GPU work (`cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...>` for the matmul,
4.480us self CUDA; `vectorized_elementwise_kernel` for the add, 2.496us self CUDA)
adds up to just ~2.1% of total self time (6.976us out of ~332.9us combined). The rest
is CPU: the `matmul_add` op itself, `aten::matmul`/`aten::mm` dispatch, and kernel
launch overhead (`cudaLaunchKernel`, `cuLaunchKernel`, `cudaDeviceSynchronize`).

The GPU stays idle most of the time, which is an immediate red flag. The reason this
happens is that the GPU can compute a small matmul (64x64) very quickly, so almost all
the wall time goes into the CPU preparing kernels, launching them, and synchronizing —
not into the GPU actually multiplying. This is an overhead-bound algorithm: at this
size, dispatch/launch overhead dominates, not compute.

Takeaway: to see genuine compute-bound behavior (and to actually judge kernel
performance), increase `--size` so the matmul itself takes long enough to amortize the
launch overhead.

## Compute-bound matmul (4096x4096, bf16, eager, cold)

From `traces/matrix_multiplication/4096_bf16_cold_eager.txt`:

- Self CPU time total: 11.228ms
- Self CUDA time total: 9.075ms

Breaking the CPU side down: actual CPU dispatch work (`aten::matmul` 10.790us,
`aten::mm` 129.160us, `aten::add` 24.300us, `cuLaunchKernel` 25.420us,
`cudaLaunchKernel` 14.400us) adds up to only ~200us. The rest of "Self CPU time" —
10.897ms, 97.05% of it — is a single `cudaDeviceSynchronize` call: the CPU finishes
launching every kernel almost immediately, then blocks waiting for the GPU to
actually finish the work.

GPU side: the matmul kernel (`cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...>`,
note this is a *different* cutlass kernel than the wmma_tensorop one picked for the
64x64 case) takes 8.277ms self CUDA across 3 calls (2.759ms avg, 91.21% of CUDA
time). The add kernel (`vectorized_elementwise_kernel`) adds another 798.097us
(8.79%).

This flips the picture from the 64x64 run: dispatch/launch overhead (~200us) is now
negligible next to actual GPU compute (~9ms). The CPU isn't the bottleneck anymore —
it does its work fast and then just waits. This is the compute-bound regime: at
4096x4096 the matmul is large enough that GPU execution time dominates wall time,
so kernel/algorithm performance (not launch overhead) is what actually matters here.

Side note: PyTorch picked a different cutlass kernel variant for 4096 than for 64
(`cutlass_80_tensorop_bf16` vs `cutlass_80_wmma_tensorop_bf16`) — worth digging into
later why the kernel selection changes with size.

## Sanity checks: is the GPU actually doing the work, or is a bug hiding the truth?

A profiler trace only tells you *that* some kernel ran and *how long* it took — it
doesn't tell you the kernel produced the right answer, or that the "work" wasn't
skipped/cached/faked somewhere upstream. Checks worth running before trusting a
trace:

1. **Correctness, not just timing.** Compute the same op on CPU/fp32 as a reference
   and compare with `torch.testing.assert_close(gpu_out.cpu().float(), ref_out,
   atol=..., rtol=...)`. A kernel that's fast because it's wrong (or a no-op) will
   still show up in the trace looking legitimate.
2. **Vary the input, check the output changes.** Run with two different random
   seeds; if the output is identical, something is being cached/memoized
   (e.g. a stale `torch.compile` graph reused across shapes, or a buffer being
   reused without being overwritten).
3. **Check `# of Calls` against what you scheduled.** `schedule(wait=1, warmup=1,
   active=3, repeat=1)` + `step()` called 5x should show exactly 3 active calls to
   `matmul_add`. If the count is off, the scheduling/profiling wiring is wrong and
   you may be measuring the wrong phase (e.g. warmup instead of steady state).
4. **Confirm tensors are actually on GPU.** `assert x.device.type == "cuda"` right
   after creation — a silent fallback to CPU (e.g. `device` string typo, CUDA not
   actually available) would still "run" but the CUDA activity in the trace would
   be empty or trivial, not the multi-ms kernel you expect.
5. **Cross-check with an independent timer.** Wrap the op in
   `torch.cuda.Event(enable_timing=True)` start/end events (with a
   `torch.cuda.synchronize()`) and compare against the profiler's reported time.
   Two independent measurements agreeing is much stronger evidence than one.
6. **Sanity-check against the roofline.** Compute expected FLOPs
   (`2 * size**3` for the matmul) and divide by the measured kernel time to get
   achieved TFLOPS. Compare against the GPU's published peak (e.g. bf16 tensor-core
   peak). If achieved throughput is *above* hardware peak, the kernel almost
   certainly didn't do the real work (constant-folded, dead-code-eliminated, or
   reading a cached result).
7. **Watch live GPU utilization.** Run `nvidia-smi dmon` (or `nvidia-smi -l 1`)
   alongside the script. Expect utilization to spike near 100% during the
   compute-bound (4096) run and stay near idle during the overhead-bound (64) run —
   matches what the trace claims. If utilization doesn't move at all, the "GPU
   time" in the trace is suspect.
8. **Temporarily turn on `record_shapes=True`.** It's off by default here for CPU
   overhead reasons, but flipping it on for a one-off debug run confirms the shapes
   actually flowing through the op match `--size`, catching bugs like accidental
   broadcasting or a stale cached tensor of the wrong shape.
9. **Check memory allocated, not just kernel names.** `torch.cuda.memory_allocated()`
   before/after tensor creation should roughly match `3 * size**2 * dtype.itemsize`
   (x, w, b). If it doesn't scale with `--size`, tensors aren't being created the way
   you think.

## Understanding `schedule(wait, warmup, active, repeat)`

Doubts while reading a blog post on the profiler schedule (`wait=1, warmup=1,
active=3, repeat=1`), and the resolution:

**Q1: what does `repeat` do? (blog didn't explain it)**

`wait + warmup + active` together form one *cycle*. `repeat` is how many times
that whole cycle runs before the profiler goes idle. `repeat=1` → the cycle runs
once (1 wait + 1 warmup + 3 active = 5 steps total), and any `prof.step()` calls
after that record nothing. `repeat=2` would run the cycle twice back-to-back (10
steps total), giving two independent active windows in the same trace — useful in
a real training loop to sample multiple points instead of trusting one window.
`repeat=0` means repeat indefinitely for the life of the profiler context — the
usual choice when profiling an actual long training run rather than a short
benchmark script like this one.

**Q2: our loop calls `step()` 5 times (`for _ in range(5)`) — does that mean the
function actually runs 5 x 3 = 15 times because of `active=3`?**

No — it's 1:1, not multiplicative. Each loop iteration advances the schedule by
exactly one step, and `prof.step()` is what moves the phase counter forward:

- iteration 1 → `wait` (function runs, profiler not attached, nothing recorded)
- iteration 2 → `warmup` (function runs, profiler attached and collecting, but the
  result is thrown away — this exists to let profiler-side bookkeeping, e.g. CUPTI
  buffers, warm up so the *first* attached call doesn't pollute the trace)
- iterations 3, 4, 5 → `active` (function runs, this time recorded into the trace)

Total function calls = 5 (matches `range(5)`), total *recorded* calls = 3. This
matches what we already saw in our own trace: `matmul_add` had `# of Calls = 3`,
i.e. exactly `active`.

The misconception to drop: `wait`/`warmup` don't mean "skip running the code" —
the real computation happens on every single call regardless of phase. What
differs per phase is only whether the profiler is attached and whether it keeps
what it captured. `wait` exists to let cold-start noise (first CUDA context use,
cudnn/cutlass algorithm selection, page faults) settle before profiling even
attaches; `warmup` exists to let the profiler's own instrumentation overhead
settle before its data is trusted.

## Not every trace gap is your code's fault: CUPTI "Activity Buffer Request"

From the [HF PyTorch profiler blog post](https://huggingface.co/blog/torch-profiler),
Figures 8-11: the CPU and GPU lanes in a trace can show an offset (their example:
~2.5ms) between "CPU submitted the kernel" and "GPU shows it executing." The
instinct is to blame launch overhead or a stalled GPU — but that's not what this is.

**What a "buffer request" is:** GPU-side activity recording (kernel start/end
timestamps, etc.) isn't free — PyTorch's profiler relies on CUPTI, which needs a
chunk of GPU VRAM ("buffer") to write activity records into as they happen. An
"Activity Buffer Request" is CUPTI asking to allocate one. It's profiler
bookkeeping, not part of the actual matmul/add compute.

**Why the initial offset happens:** before any GPU activity can be recorded, the
first buffer has to be allocated, and that allocation isn't instant. It happens
right as the first kernels are submitted, so the recording lags the real
submission — even though the GPU may have already started. Counterintuitively,
`wait`/`warmup` don't prevent this: they warm up *your* code path, not the
profiler's own first-buffer-allocation cost, since buffers aren't pre-allocated
during those phases either.

**Why a gap can appear mid-trace (e.g. between the matmul and add kernel in one
step):** each buffer has finite capacity. If it fills up mid-step, CUPTI has to
pause, request a new one, and resume — that pause shows up as a gap between two
kernels that should otherwise run back-to-back.

**How to tell this apart from a real bug:** run many more active iterations (the
post used `active=20`) and see if the gap recurs on a predictable, periodic
pattern (→ likely a real per-step cost in your code) or shows up once/rarely (→
almost certainly a one-off profiler buffer reallocation, not something wrong with
your algorithm). In the post's case the gap appeared exactly once across 20
iterations, confirming it was profiler-internal.

Ties back to the sanity-check list above: a trace can lie to you not just via bugs
in your code, but via the profiler's own instrumentation overhead. Always ask
*why* a gap exists — profiler housekeeping and real algorithmic stalls can look
identical at a glance.

## `aten::mm` has two CUDA calls, `aten::add` has one — but the exact API names can differ

The blog post notes that `aten::mm` triggers two CUDA Runtime calls
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor` + `cudaLaunchKernel`), while
`aten::add` only triggers one (`cudaLaunchKernel`). Checked our own
`64_bf16_cold_eager.json` and found the same *shape* of asymmetry, but different
specific calls:

```
aten::mm  (dur=89.92us)
├── cudaDeviceGetAttribute  (cuda_runtime)
└── cuLaunchKernel          (cuda_driver)

aten::add (dur=18.28us)
└── cudaLaunchKernel        (cuda_runtime)
```

Two calls for `mm`, one for `add` — matches the post. What differs is which API
fills each slot:

- **Query call** (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` vs
  `cudaDeviceGetAttribute`): both are the matmul kernel-selection logic
  (cutlass/cuBLAS) asking the device something before picking a kernel
  variant/launch config. Which exact call gets used depends on the
  cutlass/cuBLAS version compiled into that PyTorch build. `add` never needs this
  step since there's no kernel variant to choose between.
- **Launch call** (`cudaLaunchKernel` vs `cuLaunchKernel`): same job (launch the
  kernel), different API layer — Runtime API vs the lower-level Driver API. Some
  kernel-dispatch paths (e.g. cutlass kernels loaded via a driver-level launcher)
  go through the driver API directly instead of the runtime wrapper.

Takeaway: the *specific* CUDA API calls you see in a trace can vary with PyTorch
version, CUDA/cutlass version, and GPU architecture — don't expect to see the
exact same function names as someone else's trace (including a blog post's). What
should transfer is the underlying intuition: a kernel that needs to choose among
implementations/launch configs will show extra CPU-side "query" overhead before
launch; a simple, fixed-dispatch op like elementwise add won't.

## Does `--compile` actually fuse matmul + add into one GPU kernel?

This happens when `--compile` is used (i.e. `torch.compile(matrix_multiplication)`
is applied — see `args.compile` in `one.py`), which routes execution through
Inductor instead of eager mode. The question worth asking of any "fusion" claim:
did it actually produce a fused CUDA kernel, or just fuse at a higher level?

Inductor takes `torch.add(torch.matmul(x, w), b)` and rewrites it into a single
`aten::addmm(b, x, w)` call — one dispatcher-level op instead of two
(`aten::matmul` + `aten::add`). But the actual GPU work underneath is still the
*same* cuBLAS/cutlass GEMM kernel eager mode already used (e.g.
`ampere_bf16_s16816gemm_bf16_128x256_...` or the `cutlass::Kernel2<...>` kernels
we've already seen) — no new fused CUDA kernel was generated. `addmm` already
existed as a single fused-bias-matmul operator in eager mode too; Inductor is just
choosing to dispatch to it instead of the separate `matmul` + `add` calls.

Takeaway: "fusion" is not one thing. Here it's fusion *at the dispatcher/graph
level* (fewer op-dispatch calls, less Python/aten overhead) — not fusion *at the
kernel level* (a new single kernel doing both matmul and add in one GPU launch,
which is what horizontal/vertical kernel fusion via Inductor's Triton codegen
would look like for elementwise-heavy graphs). Don't assume "fused" in a trace
means "one new kernel" — check the actual kernel name in the CUDA lane to be
sure, same as the sanity-check habit above of not trusting a summary claim without
looking at the raw trace.
