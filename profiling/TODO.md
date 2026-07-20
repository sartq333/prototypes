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

- [ ] Build a decode-step profiling script: single new-token forward per
      step, growing KV cache, `q` at seq_len=1 with `k`/`v` at the cached
      length, no causal mask needed. Reuse the wait/warmup/active pattern
      sized for decode (see notes.md's "how many tokens" guidance:
      `wait=2, warmup=3, active=~20`).
- [ ] Apply `torch.compile(mode="reduce-overhead")` to real Qwen decode
      steps. Decode is the overhead-bound regime (small batch-of-1-token
      matmuls) — exactly where CUDA graphs should help most, per the
      `one.py` matmul investigation. Remember the static-shape requirement:
      a growing KV cache breaks CUDA graph capture, so this needs
      bucketing/padding to fixed shapes first (see notes.md's KV-cache
      gotcha).

## torch.compile modes not yet tested empirically

- [ ] `max-autotune` / `max-autotune-no-cudagraphs` — only `"default"` and
      `"reduce-overhead"` have been tried so far (on the isolated matmul in
      `one.py`).
- [ ] `"lite"` mode — discussed conceptually (fast-compile, opt-in
      optimizations, `fallback_by_default`) but never actually run.

## Other open threads

- [ ] Why does cutlass pick different kernel variants at different sizes
      (`wmma_tensorop` vs plain `tensorop`)? Flagged early in notes.md,
      never resolved.

## From the original learning plan (HF blog Part 2)

- [ ] GeGLU MLP fusion exercise (`nn.Linear` → fused GeGLU MLP).
- [ ] Hand-tuned Liger kernel vs. `torch.compile`-generated kernel
      comparison on the same shape — per CLAUDE.md, expect an ambiguous case
      (static-shape specialization vs. shape-general robustness) worth
      sitting with before resolving.
