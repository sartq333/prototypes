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

# Real model: profiling Qwen 0.5B prefill (`profile_qwen_05.py`)

Plan, before any real numbers exist — filling in results as we run them.

## Setup decision: HF wrapper, not a reimplementation

Use `AutoModelForCausalLM.from_pretrained(...)` directly rather than
reimplementing the model. HF's implementation is just PyTorch modules
underneath (`nn.Linear`, attention, RMSNorm) — profiling it gives the real
dispatch chain of the model people actually deploy, not a possibly-different
toy version.

Key setup choice: `attn_implementation` — `"eager"` vs `"sdpa"` vs
`"flash_attention_2"`. This changes what shows up in the trace completely:
`"eager"` should look like `four.py`'s `NaiveAttention` (separate `matmul` →
`mul` → `masked_fill_` → `softmax` → `matmul` ops). `"sdpa"` dispatches to
`torch.nn.functional.scaled_dot_product_attention`, a single fused op that
can route to a flash-attention-style kernel — the whole QKᵀ→softmax→AV chain
done on-chip without ever materializing the full attention score matrix to
HBM. Same "avoid an HBM round trip" epilogue logic as the GEMM bias-add
fusion, applied across a whole attention block instead of one GEMM. Plan:
diff `four.py`'s naive-attention trace against Qwen's `"eager"` trace
(should look similar) and then against `"sdpa"` (should collapse to one op)
— makes the fusion benefit concrete instead of abstract.

Isolate-vs-full-model order stays the same as established earlier: full
model first to find which component actually owns the time budget, then
pull that component out standalone (HF's real shapes) the same way `four.py`
already isolates attention.

## Pitfall: don't call `model.generate()` to profile prefill, even with `max_new_tokens=1`

First draft of `profile_qwen_05.py`'s `step()` called
`model.generate(**model_inputs, max_new_tokens=1)`. This is wrong for
measuring pure prefill, and it's why the trace didn't look like `one.py`'s
/`two.py`'s /`four.py`'s clean single-call traces (and why
`cudaDeviceSynchronize` wasn't where expected).

`generate()` is a heavy Python-level orchestration wrapper around the actual
forward pass — even for a single new token, it validates the generation
config, initializes a KV cache object, runs the logits-processor pipeline,
checks stopping criteria, and does the sampling/argmax step. All of that
happens *inside* the `record_function("prefill_llm")` scope, so "prefill
time" ends up contaminated with orchestration overhead that has nothing to
do with the model's actual `aten::` dispatch chain — exactly the kind of
CPU-side bookkeeping cost this whole project has been about learning to
separate from real compute, just reintroduced by a different, higher-level
API this time.

Fix: call the model directly — `model(**model_inputs)` — one plain forward
pass over the full prompt, nothing else. That's the real, clean prefill.
Reserve `generate()` for when decode/the orchestration loop itself is
deliberately the thing being measured.

### Verified: what `generate()` actually adds (A/B trace diff, `--use_generate` flag)

Added a `--use_generate` flag so both paths write to distinguishable,
non-overwriting trace files (`llm_direct.json` vs `llm_generate_wrong.json`),
then diffed op/kernel names between the two traces directly. Result: **87
distinct op/kernel names appear only in the `generate()` trace, none of them
touching the model's actual compute.** Three real clusters:

1. **A full top-k/top-p sampling pipeline** — corrects the earlier guess of
   "just an argmax." Qwen's default `generation_config` samples rather than
   greedily arg-maxing: `aten::topk`, `aten::sort`, `aten::multinomial`,
   `aten::exponential_`, `aten::aminmax`, `aten::cumsum`, plus a large set of
   CUB/thrust GPU sort kernels (`DeviceRadixSortHistogramKernel`,
   `DeviceRadixSortOnesweepKernel`, `bitonicSortKVInPlace`, `gatherTopK`,
   ...). Sampling requires sorting the *entire vocab logit distribution* on
   GPU every generated token — dozens of kernels for what greedy decoding
   would've done in one reduction. Almost certainly the single largest
   contributor to the extra time/kernel count.
2. **EOS/stopping-criteria checking** — matches the original prediction:
   `aten::eq`, `aten::isin`, `aten::any`, `aten::is_nonzero`, `aten::item`,
   `aten::_local_scalar_dense` (the actual op underneath `.item()`) appear
   only in `generate()`. Confirmed quantitatively: `cudaMemcpyAsync` goes
   from 3 calls (direct) to 54 (generate); `cudaStreamSynchronize` from 3 to
   21. Checking "is this token EOS" needs a real device-to-host sync to get
   a Python bool — can't be avoided while staying in Python-level control
   flow.
3. **Mask/cache bookkeeping** — `masked_fill_`, `scatter`, `full`, `ones`,
   `sub`/`rsub`/`div`: extending the attention mask and recomputing
   cache-position state for a next decode step that never actually runs
   (since `max_new_tokens=1`), but the bookkeeping still executes.

Raw count: `cudaLaunchKernel` goes from 2,985 (direct) to 3,252 (generate),
consistent in direction/magnitude with the original 3201→3468 observation
that started this investigation. Minor unrelated curiosity: `aten::alias`
appears only in the *direct* trace — a small difference in how the two
paths construct their output view, not worth chasing further.

## Experiment design for prefill

Correction to the initial framing ("find the optimal input size"): prefill
doesn't have a single optimal prompt length — there's a **plateau bounded by
two different failure modes**:

- **Too short**: attention/MLP GEMMs per layer are tiny (like the 64x64
  matmul case). CPU dispatch overhead is roughly *constant* per layer
  (established earlier: ~156us regardless of matmul size) and gets paid once
  per transformer layer (~24 layers for a 0.5B model) for very little actual
  compute. Overhead-bound, at whole-model scale.
- **Too long**: attention cost scales O(seq_len²) (QKᵀ and softmax·V both
  grow quadratically) while MLP/projection cost scales O(seq_len) (linear —
  same weight matrices, more tokens). So attention's *share* of total
  prefill time isn't fixed — it grows relative to MLP as seq_len increases,
  dominating past some crossover point. Separately, KV-cache/activation
  memory grows with seq_len too — a *capacity* concern (fits in VRAM? how
  much batch size does it leave elsewhere?), a different axis from the
  bandwidth-based memory-bound-vs-compute-bound roofline distinction from
  earlier. Don't conflate "memory" in these two senses.

**Plan:**

1. Fixed batch=1, bf16, eval + `no_grad`, one prefill forward pass per
   prompt length (not iterated like decode — one call over the full
   `input_ids`), reusing the established `wait`/`warmup`/`active` schedule
   pattern.
2. Sweep seq_len exponentially: 16, 64, 256, 1024, 4096, as far as VRAM
   allows.
3. Per run, pull: total wall time → tokens/sec = seq_len / wall_time; Self
   CPU vs Self CUDA split (watch overhead-bound → GPU-bound transition at
   whole-model scale); CUDA self-time grouped by op category (attention
   kernels vs linear/MLP kernels) to see the O(seq_len²) vs O(seq_len)
   crossover in real numbers; `torch.cuda.max_memory_allocated()` vs seq_len.

That sweep is what should empirically locate the plateau's two edges,
rather than guessing at a single "ideal" size.

### Implementation: attention-vs-MLP split requires structural scoping, not name matching

Realized before writing any sweep code: you cannot separate "attention CUDA
time" from "MLP CUDA time" by op or kernel name. Checked Qwen2's actual
module structure —

```
Qwen2Attention: q_proj, k_proj, v_proj, o_proj   (all nn.Linear)
Qwen2MLP:       gate_proj, up_proj, down_proj    (all nn.Linear)
```

Attention owns four `nn.Linear` layers of its own. Those dispatch through
the identical `aten::linear` → `aten::addmm` → `cutlass::Kernel2<...>` chain
as MLP's projections — indistinguishable by name. Fix: wrap `self_attn` and
`mlp` submodule forwards (for every decoder layer) in a named
`record_function("attn_block")`/`"mlp_block")` region via a forward-patching
hook (`instrument_attn_and_mlp` in `profile_qwen_05.py`), so the trace
carries the attribution structurally regardless of what ops run inside.

### Two aggregation bugs caught before trusting any numbers

Both would have produced numbers off by 3-10x if unnoticed — worth recording
since they'll recur in any future custom trace-aggregation code:

1. **`sum(e.self_device_time_total for e in events)` over-counts massively**
   because it includes `record_function` marker/container events
   (`ProfilerStep*`, `attn_block`, `mlp_block`) alongside real leaf ops.
   `ProfilerStep*` alone showed `self_device_time_total` of 56ms in a run
   whose real wall time was ~12ms — the marker's "self" time is a profiler
   bookkeeping artifact, not compute. Fix: filter `not e.is_user_annotation`
   before summing — this is what the printed "Self CPU/CUDA time total"
   table lines already do internally, which is why plain table-reading never
   hit this but hand-rolled aggregation did.
2. **Every `record_function` scope appears *twice* in `key_averages()`**:
   once as a `DeviceType.CPU` entry (`device_time_total` = real correlated
   child-kernel execution time — what you want) and once as a
   `DeviceType.CUDA` "device timeline span" entry (start-of-first-kernel to
   end-of-last-kernel, including any GPU idle gaps in between — 7x larger in
   one measured case, and not actual compute time). Summing both double/
   over-counts. Fix: filter to `str(e.device_type) == "DeviceType.CPU"`.
   This is the same duplicate-row phenomenon first seen (and left
   unexplained) in the very first `one.py` matmul trace at the start of this
   project — now properly understood.

Caught both by sanity-checking that `self_cpu_ms`/`self_cuda_ms` should be
the same order of magnitude as independently-measured `wall_ms`, and that
`attn_cuda_ms + mlp_cuda_ms` should never exceed total `self_cuda_ms` (a
version of the "does the number make physical sense" sanity check from
earlier in these notes) — both were violated before the fix, by ~10x and
~2x respectively.

### Results: the plateau and the crossover, both empirically confirmed

Full sweep (`--seq_lens 16,64,256,1024,4096,8192,16384`, batch=1, bf16,
`attn_implementation="eager"`, synthetic random token ids so seq_len is
exact — see below) on the RTX 5060 Ti:

| seq_len | tok/s | self_cpu_ms | self_cuda_ms | attn_cuda_ms | mlp_cuda_ms | peak_mem_mb |
|---|---|---|---|---|---|---|
| 16 | 1,367 | 14.0 | 8.9 | 1.39 | 1.83 | 987 |
| 64 | 4,400 | 14.0 | 9.4 | 1.42 | 2.02 | 1,002 |
| 256 | 16,682 | 14.8 | 18.2 | 2.76 | 4.02 | 1,060 |
| 1024 | **18,152 (peak)** | 65.4 | 122.2 | 32.8 | 14.1 | 1,293 |
| 4096 | 6,646 | 792.0 | 1,470.1 | 582.2 | 63.1 | 3,346 |
| 8192 | 3,614 | 2,892.6 | 5,643.4 | 2,474.5 | 148.2 | 10,252 |
| 16384 | OOM | — | — | — | — | — |

- **Plateau located, not guessed**: throughput peaks at seq_len=1024
  (18,152 tok/s) for this model/hardware, falling off on both sides exactly
  as predicted — overhead-bound below it, quadratic-attention-bound above.
- **The O(seq_len²) vs O(seq_len) crossover is visible and matches the
  exponents closely**: attention overtakes MLP as the dominant cost between
  seq_len 256 and 1024. Growth rates confirm the predicted scaling almost
  exactly — attention CUDA time grew ~17.8x from seq_len 1024→4096 (4x
  seq_len increase; O(seq_len²) predicts 4²=16x) and ~4.25x from 4096→8192
  (2x seq_len; predicts 2²=4x). MLP grew ~4.46x and ~2.35x over the same
  intervals — close to the predicted linear 4x/2x.
- **Overhead-bound → GPU-bound transition confirmed at whole-model scale**:
  `self_cpu_ms` stays roughly flat (~14ms) from seq_len 16→256 while
  `self_cuda_ms` grows underneath it; `self_cuda_ms` overtakes `self_cpu_ms`
  by seq_len 256→1024 — same story as the isolated 64x4096 matmul case,
  now reproduced inside a real 24-layer model.
- **Memory capacity wall found**: peak memory jumps ~3x (3.3GB → 10.3GB) for
  only a 2x seq_len increase (4096→8192) — consistent with eager attention
  materializing the full O(seq_len²) score matrix — and OOMs at 16384. This
  is the actual VRAM ceiling for this model, batch=1, eager attention, on
  this GPU.

Artifacts: `traces/llm/prefill_sweep.csv` (raw data), `prefill_sweep.png`
(4-panel matplotlib/seaborn plot: throughput, CPU-vs-CUDA split, attn-vs-MLP
CUDA time on log-log axes, peak memory — all vs seq_len).

### Word count vs. token count

Earlier runs built prompts via `"what is your name? " * N` (phrase
repetition) — an imprecise proxy for seq_len since tokenizer merging at
repeat boundaries doesn't scale token count linearly with N. The sweep
instead constructs `input_ids` directly as synthetic random token ids
(`torch.randint(0, vocab_size, (1, seq_len))`) with an all-ones attention
mask — exact seq_len control, no tokenizer ambiguity. Same "only the shape
matters for compute cost" reasoning `one.py`/`two.py` already used with
`torch.randn` synthetic tensors — real text content doesn't affect
GEMM/attention FLOPs, only the shape does, so there's no need to construct
real language for a pure compute-shape sweep.

## Serving-system context (why prompt length matters beyond one GPU)

Not directly observable from single-GPU local profiling (this is a systems
layer above one forward pass), but the mechanistic curves above are the
actual input these systems are built around:

- **Continuous batching**: servers (vLLM, TGI) dynamically add/remove
  sequences from a running batch every decode step — prefill and decode
  requests end up interleaved on the same GPU, not run in separate phases.
- **Chunked prefill**: exists because of the long-prompt problem above — one
  huge prefill is a big blocking compute burst that would stall other users'
  decode latency, so it gets split into chunks interleaved with other
  sequences' decode steps.
- **Prefix caching**: shared prefixes (e.g. system prompts) get their KV
  cache computed once and reused — reduces *effective* prefill cost
  independent of nominal prompt length.
- **Short prompts**: not a "can't be hosted" problem — it's the
  overhead-bound waste above, at the batch level: many small requests
  processed individually waste proportionally more time on fixed per-request
  overhead unless batched well.

## In-place ops: what they actually save, and what still applies in an eval/no_grad project like this one

**The convention**: any ATen op with a trailing underscore is the in-place
variant — `add_()`, `mul_()`, `copy_()`, `masked_fill_()` (already used in
`four.py`'s `NaiveAttention`: `scores.masked_fill_(mask, float("-inf"))`).
`x.add_(y)` writes the result into `x`'s existing storage; `x = x + y` (or
`torch.add(x, y)`) allocates a brand new tensor for the result.

**What's actually saved — memory, primarily, not raw compute speed.** The
underlying kernel does the same arithmetic either way; an in-place add
doesn't run a "faster add" kernel. What it avoids is the allocation itself:
no new O(N) buffer, no extra call into the caching allocator, less
allocator fragmentation from churn over many calls. Any time savings are
*indirect* — less allocation/deallocation overhead, not less compute. Don't
expect an in-place rewrite to show up as a faster kernel in a trace; expect
it to show up as a lower peak memory number (exactly the `peak_mem_mb`
column the prefill sweep already tracks).

**The usual big caveat — autograd — mostly doesn't apply to this project.**
In-place ops can break `backward()` if the original (pre-modified) value is
needed for a gradient computation; PyTorch's autograd engine tracks a
version counter per tensor and raises at backward time if it detects this.
But everything in this project runs under `torch.no_grad()`/`model.eval()`
for pure inference profiling — no gradients are ever computed, so this
specific restriction isn't a live concern here the way it would be in a
training script.

**The caveat that *does* still apply, independent of autograd: aliasing via
views.** This project has spent a lot of time on exactly this mechanism —
`.t()`/`.transpose()`/slicing/`as_strided` all create a *view* sharing the
same underlying storage as the original tensor, not a copy (see the
`aten::t`/`aten::as_strided` zero-CUDA-time entry above). In-place-modifying
a view also modifies whatever it's a view *of* — if that original tensor is
still needed elsewhere (e.g. held for a residual connection), an in-place
op on a view derived from it is a silent correctness bug, not a crash.
Worth checking whether a tensor is a fresh allocation or a view before
converting an op to its in-place form.

**For a pre-built model like Qwen, you don't control most op choices
directly — this is largely what `torch.compile`/Inductor already
automates.** HF's modeling code chooses `add`/`mul`/`matmul` etc. as library
code; rewriting it by hand to use in-place variants is invasive and fragile
across `transformers` versions. But Inductor's own scheduling passes
already do exactly this kind of buffer-reuse/memory-planning work as part
of graph compilation — recall the `"lite"` compile mode's config
(`torch._inductor.list_mode_options()`, logged earlier) explicitly sets
`allow_buffer_reuse: False` to *disable* this pass for faster compilation,
which implies it's *on* by default in normal compile modes and is
Inductor's automatic version of "make ops in-place/reuse buffers where
safe." Practical path: rather than manually rewriting ops, compare
`peak_mem_mb` between an eager sweep and a `torch.compile`-wrapped sweep at
the same seq_len — the delta (if any) is buffer-reuse working for you
already, no manual rewriting needed. Manual in-place rewriting is really
only relevant for custom code you write yourself (e.g. a hand-rolled decode
loop), not for orchestrating calls into an existing library model.

## What is WMMA? (resolves the `wmma_tensorop` vs `tensorop` open question)

**WMMA = Warp Matrix Multiply-Accumulate.** It's how you get a GPU's
dedicated **Tensor Cores** — specialized hardware for small matrix
multiplies, physically separate from the regular CUDA cores that do
ordinary scalar math — to actually do work.

- **Warp**: 32 CUDA threads execute in lockstep as a group (a warp). WMMA
  isn't per-thread — all 32 threads cooperate, each holding a fragment of
  the input matrices in its own registers, together feeding one Tensor Core
  operation.
- **Multiply-Accumulate**: computes `A@B + C`, not just `A@B` — exactly the
  mechanism a GEMM kernel's mainloop uses to build up a full matmul: load a
  tile of A, a tile of B, multiply-accumulate into the same output
  registers, slide to the next K-tile, accumulate again, repeat. WMMA is the
  literal hardware instruction doing that one accumulation step, tile by
  tile — ties directly to the mainloop/epilogue split in the GEMM epilogue
  entry above, and it's one of the narrow places where even CUTLASS drops
  to literal inline PTX assembly (no clean C++ way to issue a Tensor Core
  instruction) — see the "writing a kernel" compilation-stack entry above.

**Resolves the open question from the very first matmul investigation**
(why cutlass picked `cutlass_80_wmma_tensorop_bf16_..._32x32_64x2_nn_align8`
for the small 64x64 case but `cutlass_80_tensorop_bf16_..._64x64_32x6_tn_...`
for the large 4096x4096/2048x2048 cases). Confirmed via CUTLASS's own GitHub
discussion (NVIDIA/cutlass#1446): `WmmaTensorOp` uses the higher-level
`wmma.mma` PTX instruction — the older, easier, warp-unified API. Plain
`TensorOp` uses the lower-level `mma.sync` PTX instruction directly — newer,
finer control, faster. Small/simple tile configs route through the older
WMMA-based kernel generator; larger, more performance-critical shapes route
through the newer, faster `mma.sync`-based path. That's exactly the split
observed across every trace in this project: small shapes → `wmma_tensorop`,
large shapes → plain `tensorop`.

## tinygrad's equivalent of `cudaLaunchKernel`: HCQ

Every trace in this project shows PyTorch reaching the GPU by calling into
NVIDIA's own provided software layer — `cudaLaunchKernel` (CUDA Runtime API)
or `cuLaunchKernel` (CUDA Driver API). Both are APIs NVIDIA wrote; PyTorch
asks NVIDIA's driver to launch the kernel for you, and every one of those
calls carries the real, measured, non-zero CPU-side launch overhead this
whole project has been about.

tinygrad has an additional, more radical option for NVIDIA/AMD specifically:
**HCQ** (Hardware Command Queue) — talk to the GPU's hardware command queue
directly, skipping the vendor runtime layer entirely. Confirmed via
tinygrad's own docs: "In HCQ, all interactions with devices occur in a
hardware-friendly manner using command queues... commands [are] issued
directly to devices, bypassing runtime overhead such as HIP or CUDA."
tinygrad's `NV=1`/`AMD=1` backends are built on this, explicitly framed as
reducing Python-side/CPU-side launch time, especially for multi-GPU.

Submitting a kernel through HCQ looks like this — and it's plain Python, the
whole way down, even at this "closest to hardware" layer:

```python
HWQueue().wait(signal_to_wait, value_to_wait) \
         .exec(program, args_state, global_dims, local_dims) \
         .signal(signal_to_fire, value_to_fire) \
         .submit(your_device)
```

Same "record now, execute later" laziness as the Tensor API, just one layer
lower: each chained method (`.wait`/`.exec`/`.signal`) just appends an
instruction to a buffer `HWQueue` is building internally — nothing runs yet.
`.wait`/`.signal` are tinygrad's own synchronization primitives, doing the
same job `cudaStreamSynchronize` does in the standard API, just implemented
by tinygrad itself instead of borrowed from NVIDIA. The actual
hardware-touching moment is `.submit()` — Python reaches outside itself via
a low-level foreign-function call (`ctypes`-style) into the OS's GPU driver
interface, writes the command buffer into GPU-visible memory, then pokes a
specific hardware register (a "doorbell") telling the GPU new work is ready.

So even tinygrad's most direct hardware path is Python all the way down to
one narrow FFI call at the very bottom — not fundamentally different in
spirit from how PyTorch's Python call eventually bottoms out in a C++
`cudaLaunchKernel` call, just that tinygrad wrote that bottom layer itself
instead of using NVIDIA's.
