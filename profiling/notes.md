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

## `torch.compile(mode="reduce-overhead")`: which mode cuts CPU overhead, and why it's a trade, not a free lunch

The blog leaves "which `torch.compile` mode would cut CPU overhead" as an
assignment. Checked `torch._inductor.list_mode_options()` — of the modes, only
`"reduce-overhead"` sets `triton.cudagraphs: True` without also paying for
`max_autotune`'s expensive kernel search. Reasoning before looking: CUDA graphs
capture a whole sequence of kernel launches once, then replay them via a single
`cudaGraphLaunch`, skipping the Python interpreter and the ATen dispatcher
per-op on every subsequent call. This doesn't change *compute* (same kernels,
same FLOPs) — it only removes *launch* overhead. So it should help most exactly
where launch overhead dominates (our overhead-bound 64x64 case) and do nothing
where it's already negligible (our compute-bound 4096x4096 case).

Ran `--compile` with `mode="reduce-overhead"` hardcoded in `one.py`, warm, at
both sizes, to check the prediction against real traces:

**Size 64 (warm): confirmed, ~17% faster.**

| | eager | compile (`reduce-overhead`) |
|---|---|---|
| Per-step wall time (`ProfilerStep` CPU total avg) | 133.804us | 111.520us |
| Self CUDA total | 6.911us | 9.537us |

Eager's CPU chain (`aten::matmul` → `aten::mm` → `cuLaunchKernel`, then
`aten::add` → `cudaLaunchKernel` — two dispatched ops, two launches) collapses
into `TorchDynamo Cache Lookup` (guard check) → `AOTDispatcher` prologue → one
`cudaGraphLaunch` replaying everything.

Bonus finding, worth being skeptical of "fusion" claims about (see the addmm
entry above): checked the *untruncated* GPU kernel names via the raw trace
JSON, and the matmul kernel is byte-identical between eager and compiled
(`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_nn_align8`), same
duration (4.609us vs 4.447us) — but the separate `vectorized_elementwise_kernel`
that did the `add` in eager is **completely absent** from the compiled trace.
No kernel anywhere does a standalone add. Best explanation: cutlass GEMM
templates commonly support an optional bias-add *epilogue* as a runtime
parameter (not a separate template instantiation), so the same kernel symbol
can run GEMM-only (eager, `aten::mm`) or GEMM+bias-add (compiled) depending on
the `Params` passed at launch — genuine kernel-level fusion this time, unlike
the dispatcher-only `addmm` fusion from the blog's example. Not 100% certain
without deeper tooling (e.g. Nsight Compute) — flagged as the likely
explanation, not a confirmed one.

**Size 4096 (warm): not just "no gain" — actively slower.**

| | eager | compile (`reduce-overhead`) |
|---|---|---|
| Self CUDA total | 9.057ms | 13.047ms |
| Self CPU total | 11.025ms | 13.258ms |

Two concrete, distinct causes found in the trace, not just generic "compile
setup cost":

1. **`# of Calls` for the GEMM kernel: 3 (eager) vs 4 (compile).** Only 3
   `step()` calls happen during `active`, so where's the 4th real GEMM
   execution coming from? A CUDA graph *replay* still runs the real kernels on
   the GPU — it only skips CPU-side dispatch/launch, not compute. Best
   explanation: graph *capture* itself requires actually executing the kernels
   once for real, in "stream capture mode." The standalone 3-iteration warmup
   (run before the profiler even attaches) likely wasn't enough to finish
   Dynamo tracing + AOTDispatcher + capture, so the capture-execution leaked
   into the first profiled `active` step: 1 real capture-run + 3 real replays
   = 4, and the extra ~2.7ms of Self CUDA time (10.940ms − 8.253ms ≈ 2.69ms)
   is almost exactly one GEMM call's cost. Testable: increase the standalone
   warmup count and check whether `# of Calls` drops back to 3.
2. **A new kernel, `memcpy128` (644.042us total), doesn't exist in eager at
   all.** This is `aten::_foreach_copy_` → `multi_tensor_apply_kernel`, a
   **device-to-device** (GPU VRAM → GPU VRAM, *not* CPU→GPU — the tensors are
   already created with `device="cuda"`, never touch host memory) copy of the
   live input tensors into the CUDA graph's static memory buffer. Graph replay
   needs its inputs at fixed addresses, and Inductor can't assume the caller
   passes the same tensor objects every time (even though our script happens
   to), so it conservatively re-copies inputs into the static pool on *every*
   call — `# of Calls` for this copy is 3, matching every active step, not a
   one-time setup cost. At size 64 the analogous kernel (`memcpy32_post`) cost
   1.6us total, invisible. At 4096 (≈33MB per bf16 tensor) it costs 644us —
   this copy scales with tensor size the same way the actual matmul does,
   while the launch overhead it's meant to amortize away is roughly constant
   regardless of size. So the bigger the tensors, the worse this specific
   trade gets, on top of the launch-overhead savings having nothing to work
   with in the first place at this size.

Takeaway: `reduce-overhead`/CUDA graphs isn't free — it swaps *launch*
overhead (roughly constant per op, independent of tensor size) for a
*copy-in* overhead (scales with tensor size) plus a one-time capture cost that
can leak into measurements if warmup is insufficient. It's a genuine win only
when launch overhead was the actual bottleneck (small, overhead-bound
workloads) — exactly the same overhead-bound-vs-compute-bound framing from the
64 vs 4096 matmul comparison earlier in these notes, just now costing you
extra when applied to the wrong regime instead of merely not helping.

# Part 2: nn.Linear / MLP (`two.py`)

## What is a GEMM "epilogue"?

A cutlass GEMM kernel has two conceptually separate internal phases:

1. **Mainloop** — the actual `A @ B` accumulation: tiles of A and B loaded,
   multiplied via tensor-core MMA instructions, accumulated in
   registers/shared memory across the K dimension. The "real" matmul compute.
2. **Epilogue** — what happens to that accumulated result *before it's
   written out to HBM*: identity (just store it), bias-add (`addmm`), scale
   (the classic GEMM `alpha`/`beta` convention: `alpha*(A@B) + beta*C`), an
   activation function (ReLU, GELU), a dtype cast (accumulate fp32, write
   bf16), etc.

cutlass templates these as two independently swappable pieces, which is why
kernel names encode epilogue capability separately from mainloop tile/stage
sizing (`gemm_relu_bf16_64x64_32x6` — `relu` is the epilogue slot,
`64x64_32x6` is the mainloop's tile/pipeline-stage config).

**Why fusing the epilogue saves real time:** without it, the accumulated
result would have to leave the chip — the GEMM kernel writes raw `C = A@B`
to HBM, a *separate* kernel (bias-add/activation) reads it back from HBM,
computes, writes the final result back to HBM again. Three HBM round trips
for work that only strictly needs one. A fused epilogue transforms the
result while it's still on-chip (registers/shared memory) and writes out
only once — saving 2 of those 3 HBM transactions, plus an entire second
kernel launch (the CPU dispatch overhead tracked throughout these notes).
Same "avoid an extra HBM round trip" logic as flash attention fusing softmax
into the attention matmul instead of materializing the full attention matrix
to HBM — different op, identical motivation: on-chip data is nearly free to
touch, HBM round trips are the expensive resource.

## Terminology check: "overhead-bound" is not "memory-bound"

Easy to conflate, but they're two different axes:

- **Overhead-bound vs. GPU-bound**: does wall-clock time get dominated by
  *CPU dispatch/launch* before the GPU is even busy, or by the *GPU kernel
  itself*? This is the axis we've been measuring throughout (64 vs 4096
  matmul, `reduce-overhead`, etc.).
- **Memory-bound vs. compute-bound** (the roofline axis): *once a kernel is
  actually running*, is it bottlenecked by HBM bandwidth or by FLOP
  throughput?

A kernel can be GPU-bound (dominates the trace) while still being
memory-bound internally, or overhead-bound overall while whatever GPU work
does happen is compute-bound. Don't collapse these into one axis.

## `nn.Linear`, small shape (`--batch 1024 --in_dim 32 --out_dim 64`, eager): overhead-bound

`Self CPU time total: 162.611us`, `Self CUDA time total: 4.672us` — CUDA is
~2.8% of combined self time, same class as the 64x64 matmul case. Dispatch
chain: `aten::linear` → `aten::t` → `aten::addmm` (not `aten::matmul` +
separate `aten::add` — see below for why).

### Why `aten::linear` calls `addmm` directly in eager mode (no compiler needed)

Unlike our own `torch.add(torch.matmul(x, w), b)` in `one.py` (which eager
mode runs as two separate, un-fused ops), `nn.Linear`'s forward is *written*
to call `torch.addmm(bias, input, weight.t())` directly — a single
dispatcher call. `addmm` isn't a compiler optimization; it's a thin wrapper
over the GEMM primitive cuBLAS/cutlass has always supported natively
(`C = beta*C + alpha*(A@B)`, with `beta=1, C=bias`). Eager mode does zero
automatic rewriting — it just runs exactly the ops you call, in order. So
there are two distinct paths to the same fused dispatcher-level outcome:

1. A human (here, the `nn.Linear` author) picks the already-fused primitive
   when writing library code.
2. A compiler (`torch.compile`/Inductor, see the addmm entry above) rewrites
   a naively-expressed two-op graph into that same primitive.

Eager mode alone gives you neither unless you call the fused op yourself.

### The `torch.compile` reflex, and why it does ~nothing here

A common reflex is to reach for `torch.compile` whenever a model feels slow.
For a single GEMM-with-bias, compile has very little to do. This is not a
bug — it's just that compile needs *more than one operation* to possibly do
any fusing. `nn.Linear`'s eager-mode call is already the single fused
`addmm` dispatcher op (see above); there's no second op left for Inductor to
spot and merge it with. Compile's value shows up once there's an actual
*graph* of multiple ops to rewrite (our own two-op `matmul` + `add` in
`one.py`, or a real multi-layer MLP) — not on an op that was already fused
by the library author before compile ever gets involved.

### Why `aten::t` / `aten::transpose` / `aten::as_strided` cost real CPU time but exactly 0.000us CUDA

These are three distinct, dispatcher-registered ops, not redundant naming:

- `t()` — 2D-specific convenience (swaps dims 0/1, ≤2D only). This is
  literally what `nn.Linear`'s source calls (`weight.t()`).
- `transpose(dim0, dim1)` — the general N-D version. `t()`'s own
  implementation calls this — hence they nest as parent→child in the trace;
  it's the real call stack, not duplicated bookkeeping.
- `as_strided()` — the actual primitive: builds a new view (new
  sizes/strides/storage_offset) over the *same* storage, zero copy.

All three show nonzero **CPU** time because even a metadata-only view goes
through the full C++ dispatcher: dispatch-key resolution, shape/stride
legality checks, computing new stride values, allocating a new (lightweight
but real) `TensorImpl`. None of that touches the GPU — same category of cost
as `cuLaunchKernel`/`cudaLaunchKernel` (framework overhead), just for a
CPU-only op instead of a kernel launch.

They show exactly **0.000us CUDA** — not just small — because the transpose
is never materialized as a physically-transposed tensor in GPU memory at
all. Confirmed by checking the exact (untruncated) kernel name for this
trace:

```
cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_tn_align8>
```

The `_tn_` is a GEMM naming convention: `t` = this operand is read
*as transposed*, `n` = normal, baked into which kernel variant cutlass
selects. Compare against `one.py`'s hand-written `torch.matmul(x, w)` (no
transpose in the math), which picked an `_nn_` kernel — same family, same
tile size (`32x32_64x2`), differing only in the transpose flag. The
`as_strided` view (new strides, zero copy) gets passed straight into the
GEMM's launch parameters, and the `_tn_` kernel reads that operand according
to those strides as part of its own memory access pattern *during* the
multiply. No separate transpose kernel is ever launched — the CPU-side
dispatcher chain *is* the entire cost of the transpose, because a metadata
rewrite is the entire work involved.

### Kernel names encode capability, not necessarily what ran

Side finding while checking kernel names: the large-shape run
(`--in_dim 2048 --out_dim 2048`) picked
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8` — "relu"
in the name despite no activation function anywhere in `two.py`. This is a
cutlass template-naming quirk: the template class is named for the most
general epilogue it *supports* (bias-add, optional ReLU, etc.), but which
epilogue actually executes is a runtime `Params` choice, same principle as
the `addmm` bias-epilogue fusion discussion above. Don't read a kernel name
as a literal description of what ran without checking — the name identifies
capability, not necessarily the specific invocation.

## `nn.Linear`, large shape (`--batch 1024 --in_dim 2048 --out_dim 2048`, eager): GPU-bound

`Self CPU time total: 570.493us`, `Self CUDA time total: 546.083us`. Looks
close to 50/50 at first glance — it isn't. Breaking down where the 570.493us
of "CPU self time" actually goes:

```
aten::linear + aten::addmm + aten::t/transpose/as_strided + launches ≈ 156.401us
cudaDeviceSynchronize                                                = 414.092us (72.6%)
```

`cudaDeviceSynchronize` isn't dispatch work — it's the CPU blocked, waiting
for the GPU to finish (same pattern as the 4096x4096 matmul case). Real
dispatch overhead is ~156.401us — nearly identical to the small linear case's
162.611us total. Dispatch cost stayed roughly *constant* across a huge shape
change; only the GPU kernel scaled, from 4.672us to 546.083us (~117x). So
this has clearly flipped to GPU-bound, not a close call — the raw self-time
totals looking similar is misleading unless you break down what's actually
inside "CPU self time."

Open question worth resolving with the roofline check from the sanity-check
list earlier in these notes: GPU-bound doesn't automatically mean
compute-bound in the classical sense (see terminology note above). Still
need to check achieved FLOPs vs. this GPU's peak bf16 tensor-core throughput
for this exact shape (1024x2048 @ 2048x2048) to know whether it's actually
compute-bound or memory-bound internally.

## What are cuBLAS, CUTLASS, and GEMM, exactly?

One operation, two different ways to implement it on NVIDIA GPUs:

- **GEMM (General Matrix Multiply)** — a BLAS Level-3 routine. Not just
  "matmul" — its actual interface is `C = alpha*(A@B) + beta*C`. This is why
  `addmm` (bias-add) isn't a special case bolted on afterward — bias-add is
  just `beta=1, C=bias` in GEMM's *default* signature (see the epilogue
  section above). Every cutlass kernel name seen in these notes has "gemm"
  baked in (`s161616gemm_bf16...`, `s16816gemm_relu_bf16...`) because each
  one *is* a GEMM implementation, just specialized differently. GEMM is the
  operation, not a library — cuBLAS and CUTLASS are two ways to implement it.
- **cuBLAS** — NVIDIA's own closed-source, prebuilt implementation of BLAS
  (including GEMM). Call a function (`cublasGemmEx`, etc.), get a kernel
  NVIDIA hand-tuned for common shapes/dtypes. A black box you call into, not
  something you customize or recompile.
- **CUTLASS (CUDA Templates for Linear Algebra Subroutines)** — NVIDIA's
  open-source, header-only C++ *template* library for building GEMM (and
  related) kernels. Composable building blocks (tile iterators, warp-level
  MMA wrappers, epilogue functors) get instantiated into a specific kernel
  *at compile time*, specialized to an exact shape/dtype/tile-size/epilogue.
  Every `cutlass::Kernel2<...>` name seen in these traces is a literal C++
  template instantiation — the different tile sizes (`32x32_64x2` vs
  `64x64_32x6`), transpose flags (`_nn_` vs `_tn_`), and epilogue capability
  (`relu` vs plain) are different compile-time specializations of the same
  general template, not different libraries.

Real observation from our own traces: every single GEMM kernel pulled out of
a trace so far has been a `cutlass::Kernel2<...>` instantiation — never a
`cublasGemmEx`-style symbol. PyTorch's ATen backend is choosing the CUTLASS
path for these bf16/tensor-core shapes on this hardware/PyTorch version,
consistent with modern PyTorch increasingly preferring CUTLASS for
tensor-core workloads (deep per-shape specialization, and `torch.compile`'s
`max-autotune` needing to *generate and search* kernel variants — not
possible against a closed prebuilt binary).

## "Writing a kernel": the actual compilation stack, and where assembly really shows up

Easy to assume CUTLASS = "GEMM written at the assembly level" vs. cuBLAS
being higher-level. That's the wrong axis. The real compilation pipeline for
*any* CUDA kernel — CUTLASS, cuBLAS, or something hand-written from scratch —
is:

```
CUDA C++  →  PTX  →  SASS
```

- **PTX** (Parallel Thread Execution) — an intermediate, assembly-like
  instruction set. `nvcc` compiles C++ kernels down to this. Portable across
  a range of GPU generations within a compute-capability family.
- **SASS** (Streaming ASSembler) — the actual native machine code for a
  *specific* GPU architecture. PTX gets further compiled (`ptxas`, or JIT'd
  by the driver at load time) into this — what actually runs on the silicon.

Both CUTLASS and cuBLAS go through this same pipeline. CUTLASS is
overwhelmingly C++ (a template header library), not hand-written assembly.
Neither is "more assembly" than the other in general — the real axis
separating them is open-source-and-customizable-at-compile-time (CUTLASS)
vs. closed-source-fixed-prebuilt-binary (cuBLAS), not "high-level vs.
assembly."

Real nuance, though: for a handful of the most hardware-specific
instructions — tensor-core matrix-multiply-accumulate (PTX's `mma.sync`) and
async memory copy (`cp.async`) — there's no clean C++ way to express them, so
both CUTLASS and cuBLAS drop to literal inline PTX assembly for just those
operations, wrapped in a thin C++ function. Narrow and surgical, not "the
kernel is written in assembly" — closer to 99% C++ orchestration with a few
assembly-level primitives at the bottom where the hardware demands it.

"Writing a kernel" is a broader category than "writing CUTLASS." The plain
elementwise `add` kernel from Part 1 (`vectorized_elementwise_kernel`) is
ordinary hand-written CUDA C++ — no CUTLASS templates involved. CUTLASS is
specifically for templated, tunable, linear-algebra-heavy kernels (GEMM,
convolution) where a reusable tiling/epilogue system pays off.

Third path worth knowing, relevant to what's coming up (Liger kernels,
`torch.compile`'s own codegen): **Triton** — a Python-like DSL, not C++,
with its own compiler that also lowers through PTX → SASS. Inductor
generates Triton kernels for a lot of fused ops (not CUTLASS C++), and
hand-tuned kernel libraries increasingly use Triton for the same reason —
most of the performance, without C++ template metaprogramming.
