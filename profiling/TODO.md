# Profiling project — open TODOs

## Next up (prefill sweep extensions)

- [ ] Rerun the full sweep with `--attn_implementation sdpa` and diff against
      the eager numbers already in `traces/llm/prefill_sweep.csv`. Prediction
      to check: SDPA's fused kernel shouldn't change attention's O(seq_len²)
      exponent, but should change its constant factor (no materialized
      seq_len×seq_len score matrix) — expect a flatter peak-memory curve
      (maybe clearing the 16384 OOM wall) and a later attn-vs-MLP crossover
      point.
- [ ] Densify `--seq_lens` around the located peak (e.g.
      `512,768,1024,1536,2048`) — the current "peak at 1024" is just the
      nearest point on a coarse exponential grid, not a located optimum.
- [ ] Roofline check at the sweet spot: compute achieved FLOPs (attention
      QKᵀ+AV, MLP projections) vs. the RTX 5060 Ti's peak bf16 tensor-core
      throughput at seq_len≈1024. Settles compute-bound vs. memory-bound in
      the classical sense — GPU-bound (already established) is a different
      axis and doesn't imply this.
- [ ] Batch-size sweep (fixed seq_len, varying batch) as a complementary axis
      to the seq_len sweep — different scaling story, relevant to real
      serving throughput.

## Decode-phase profiling (not started — everything so far is prefill only)

What to actually check here, beyond just "build the script" — decode is a
different enough regime from prefill that it needs its own checklist, not
just a smaller version of the prefill one:

- [ ] **Shape setup**: `q` at seq_len=1, `k`/`v` at the full cached length so
      far (grows by 1 every step). No causal mask needed — a single new
      token attends to all past positions, there's nothing in the future to
      mask.
- [ ] **Schedule sizing**: `wait=2, warmup=3, active=~20` (per notes.md's
      "how many tokens" guidance) — decode shape drifts every step (cache
      keeps growing), so "steady state" needs averaging over more steps than
      prefill's single-shot measurement, and skipping more cold-start steps.
- [ ] **The core thing to verify**: decode ops are GEMV-shaped, not
      GEMM-shaped, and therefore memory-bandwidth-bound, not compute-bound.
      At batch=1, every matmul in the model — attention's cache lookup *and*
      the MLP/projection layers (`gate_proj`/`up_proj`/`down_proj`,
      `q/k/v/o_proj`) — degenerates into a matrix-*vector* product (1 row ×
      weight matrix), not a matrix-matrix product. A GEMV reads the entire
      weight matrix from HBM for a tiny amount of compute per byte read —
      classic low-arithmetic-intensity, bandwidth-bound territory. This is
      the mechanistic reason decode is expensive per-token relative to its
      FLOP count, and it's a genuinely different roofline regime than
      prefill's compute-bound GEMMs. Check: measure achieved HBM bandwidth
      (bytes moved / time) during a decode step and compare against the RTX
      5060 Ti's peak memory bandwidth spec — this is the memory-axis
      counterpart to the FLOPs-based roofline check already planned for
      prefill above.
- [ ] **Why batching exists, verified empirically**: batching multiple
      concurrent sequences' single-token decode steps together turns those
      GEMV ops back into real GEMMs (batch dimension = number of concurrent
      sequences), amortizing the HBM weight-read cost across many sequences'
      worth of compute instead of paying it per sequence. This is the
      mechanistic reason continuous batching (already covered in notes.md's
      serving-system section) exists at all. Verify by comparing decode
      throughput (tokens/sec, summed across sequences) at batch=1 vs. a
      swept batch size — expect a much better FLOPs-to-bandwidth trade as
      batch grows, unlike prefill's seq_len sweep which was bandwidth-cheap
      and compute-hungry in the opposite direction.
- [ ] Apply `torch.compile(mode="reduce-overhead")` to real Qwen decode
      steps. Decode is the overhead-bound *launch* regime (small
      batch-of-1-token matmuls, lots of them, once per layer) — exactly
      where CUDA graphs should help most, per the `one.py` matmul
      investigation. Remember the static-shape requirement: a growing KV
      cache breaks CUDA graph capture, so this needs bucketing/padding to
      fixed shapes first (see notes.md's KV-cache gotcha).

## torch.compile modes not yet tested empirically

- [ ] `max-autotune` / `max-autotune-no-cudagraphs` — only `"default"` and
      `"reduce-overhead"` have been tried so far (on the isolated matmul in
      `one.py`).
- [ ] `"lite"` mode — discussed conceptually (fast-compile, opt-in
      optimizations, `fallback_by_default`) but never actually run.

## Other open threads

- [x] ~~Why does cutlass pick different kernel variants at different sizes
      (`wmma_tensorop` vs plain `tensorop`)?~~ Resolved: `WmmaTensorOp` uses
      the higher-level `wmma.mma` PTX instruction (older, easier, warp-level
      unified API); plain `TensorOp` uses the lower-level `mma.sync` PTX
      instruction directly (newer, faster, finer control). Small shapes
      (64x64 matmul, small `nn.Linear`) picked `wmma_tensorop`; large shapes
      (2048x2048, 4096x4096) picked plain `tensorop` — consistent with
      CUTLASS routing smaller/simpler tile configs through the older WMMA
      path and larger, more performance-critical shapes through the newer
      `mma.sync` path. See notes.md's WMMA entry for the full writeup and
      source.

## From the original learning plan (HF blog Part 2)

- [ ] GeGLU MLP fusion exercise (`nn.Linear` → fused GeGLU MLP).
- [ ] Hand-tuned Liger kernel vs. `torch.compile`-generated kernel
      comparison on the same shape — per CLAUDE.md, expect an ambiguous case
      (static-shape specialization vs. shape-general robustness) worth
      sitting with before resolving.
