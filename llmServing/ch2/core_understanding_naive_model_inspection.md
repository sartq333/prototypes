# Core Understanding: Naive Model Inspection

> These questions check if you truly understand the code in `naive_model_inspection.py` — not just what it does, but why.

---

## Q1: You have a model loaded on MPS. After running inference, you call `free_gpu(model)` but don't `del model` at the call site. Will the GPU memory actually be freed? Why or why not?

**Your answer:** No. `free_gpu` will clean up cache but since `del model` is not called it'd still be in MPS memory.

**Exact answer:** No. `del model` inside `free_gpu` deletes the local copy of the reference, not the original. The model object still has 1 reference alive (the caller's `model` variable) — Python's reference counter won't drop to 0, so the object won't be garbage collected and the weights stay in MPS memory. `empty_cache()` only releases memory that PyTorch has already freed back to its allocator — it cannot free memory held by a live object. The root cause is **reference counting**, not just that `del` wasn't called.

---

## Q2: In `simple_inference`, at iteration 50 (prompt length = 10 tokens), how many tokens is the model processing in attention, and what is the computational complexity?

**Your answer:** 50 tokens (missed the prompt length). Complexity is quadratic.

**Exact answer:** **60 tokens** — 10 (prompt) + 50 (generated). Complexity is **O(n²)** where n=60. This is why no-KV-cache inference gets progressively slower — each iteration the sequence grows by 1 and attention cost grows quadratically.

---

## Q3: In `kv_cache_enabled_inference`, after the first iteration you pass `idx[:, -1:]` instead of the full `idx`. Why only the last token, and what would go wrong if you passed the full sequence every iteration even with `use_cache=True`?

**Your answer:** Only the last token is needed because K,V of old tokens and their interactions are already stored in cache. If full sequence is passed with cache enabled, no error would be raised — the code would always go into the `if past_key_values is None` branch and recompute everything, making it look like KV cache is working but it isn't.

**Exact answer:** Correct on both. One important addition: it's actually worse than just "KV cache not working." If you pass the full sequence with `past_key_values` already populated, the model sees the full sequence **plus** the cached K/V — effectively processing the prompt tokens twice. This leads to wrong position encodings and corrupted attention. The bug is silent but the generated text would be nonsensical garbage.

```
Iteration 1 - WRONG (pass full sequence with cache populated):
Input  → [t1, t2, t3, t4, t5, t6]  (full sequence)
Cache  → also loads K,V for t1-t5
Result → t1-t5 seen TWICE, positions wrong, attention corrupted → garbage output
```

---

## Q4: `torch.cuda.synchronize()` is called before measuring time in `kv_cache_enabled_inference`. What would happen to timing measurements if you removed it?

**Your answer:** GPU ops get queued but not finished yet. We'd only measure the time to send the op to the queue, not execute it. Measured time would be lower than actual.

**Exact answer:** Correct. GPU operations are asynchronous — `model(idx_cond)` returns to Python immediately after queuing the work. Without `synchronize()`, `time.time()` fires before the GPU has finished computing. You'd see artificially small per-token times. The fix for MPS is `torch.mps.synchronize()` instead of `torch.cuda.synchronize()`.

---

## Q5: In `attention_visualization`, you do `.float()` before `.numpy()`. Why can't you call `.numpy()` directly on a bfloat16 tensor, and why is this conversion safe?

**Your answer:** NumPy doesn't support bf16. It's safe because we're just doing visualization, not computation.

**Exact answer:** Correct on both. NumPy has no bfloat16 dtype, so calling `.numpy()` directly throws a `TypeError`. Converting to `float32` via `.float()` is safe here because the heatmap colors won't look any different to the human eye — precision loss only matters when doing math, not visualization.

---

## Q6: `torch.multinomial` vs `torch.argmax` — you used `argmax` initially and it immediately hit EOS on the long prompt. Why would `argmax` pick EOS, and what does this tell you about EOS in the probability distribution?

**Your answer:** `argmax` greedily picks the highest probability token which turns out to be EOS. `multinomial` picks randomly weighted by probability. EOS sits at the top of the probability distribution for that prompt.

**Exact answer:** Correct. The long prompt ends with a complete question — a natural stopping point. The model's training data has many examples where such prompts are followed by EOS, so it assigns EOS the highest probability. `argmax` always picks it. `multinomial` avoids this because even if EOS has 40% probability, it's only picked 40% of the time — the other 60% goes to actual content tokens. This is also why `temperature` exists: raising it flattens the distribution so EOS probability drops relative to other tokens.

---

## Q7: `output_attentions=True` is set in `from_pretrained`. What is the performance cost, and how would you fix it?

**Your answer:** Attention weights are stored in memory even when not used. Fix is to only pass `output_attentions=True` for specific forward passes that need it.

**Exact answer:** Correct. Every forward pass — including `simple_inference` and `kv_cache_enabled_inference` — materializes and stores all attention weight tensors even though they're never used there. The fix is to remove it from `from_pretrained` and only pass it in `attention_visualization`:

```python
# Only where needed:
outputs = model(**inputs, output_attentions=True)

# Everywhere else, omit it — default is False
outputs = model(idx_cond)
```

---

## Q8: In `simple_inference`, `idx_cond = idx` is assigned every iteration but never differs from `idx`. Is this line necessary?

**Your answer:** No — it's just there to set up the conceptual distinction between "current context" and "full sequence" before KV cache is introduced. Since the functions are split, the line is redundant.

**Exact answer:** Correct. The author likely wrote it to prepare the reader for `kv_cache_enabled_inference`, where `idx_cond` actually does become different from `idx` (only the last token). In a single function it would have semantic value. In separate functions it is dead code.

---

## Q9: In `kv_cache_enabled_inference`, `past_key_values` grows with each iteration. What is stored in it and how does memory grow?

**Your answer:** K,V values of past tokens are stored — the interactions of each token with other tokens. It grows linearly as output tokens increase.

**Exact answer:** Correct. Per new token added, the memory cost is:
```
K: num_kv_heads × head_dim × bytes_per_param
V: same
Per token per layer = 2 × 2 × 64 × 2 = 512 bytes  (Qwen2.5-0.5B)
Per token across all 24 layers = 512 × 24 = 12,288 bytes (~12KB)
After 500 tokens = ~6MB
```
Linear growth. Small for this model, but for 70B models with 80 layers it can hit tens of GBs — which is exactly why vLLM's PagedAttention was invented.

---

## Q10: `torch.no_grad()` is used in both inference functions. What exactly does it disable, and what happens to memory if you remove it?

**Your answer:** Space for gradients won't be stored. Correct direction but wanted the exact mechanics.

**Exact answer:** `torch.no_grad()` prevents PyTorch from building the **computation graph** — the graph that tracks every operation so gradients can be backpropagated via `loss.backward()`. Without it, all intermediate activations (outputs of every layer, every operation) are kept in memory. In a 500-token generation loop, you'd accumulate graphs for 500 forward passes — memory grows unboundedly until OOM. A single forward pass uses roughly **2-3x more memory** without `no_grad()`. It is not optional for inference — it's a hard requirement for anything beyond tiny sequences.

---

## Q11: Why is `torch.softmax` used instead of simple normalization (divide by sum)?

**Your answer:** Softmax is smooth — wasn't sure of the exact reason.

**Exact answer:** Two reasons simple normalization fails:

1. **Logits can be negative** — dividing by the sum gives negative probabilities, which is meaningless. `exp(x)` is always > 0 regardless of sign, so all output probabilities are valid.

2. **Amplifies differences** — `exp` is exponential, so a logit of 5 vs 4 becomes `e^5/e^4 = e ≈ 2.7x` more probable, not just 1.25x. This sharpens the distribution, making the model more decisive.

This is also why temperature works:
```python
probs = softmax(logits / temperature)
```
- Low temp → logits larger → exp amplifies differences → sharper → more greedy
- High temp → logits smaller → differences shrink → flatter → more random

---

## Q12: `get_model_size` returns 495M. The safetensors file on disk is ~988MB. Why the 2x difference?

**Your answer:** 495M is the count of weights/params. 988MB is the memory required to store them.

**Exact answer:** Correct. The exact reason is dtype:
```
495M parameters × 2 bytes (bfloat16) = 990MB ≈ 988MB
```
`numel()` counts scalar values with no concept of bytes per value. `get_model_size` returns parameter **count**, not memory — which is why `calculate_model_memory` exists and multiplies by `bytes_per_param`. If the model were float32: `495M × 4 = ~1.98GB`.
