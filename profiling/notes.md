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
