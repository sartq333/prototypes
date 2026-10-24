import time
import torch
import tiktoken
from gpt import GPTModel, generate_text_simple

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

def tokens_per_sec(model, idx, num_tokens, use_cache):
    # warm-up: untimed short run so one-off setup costs don't land in the measurement
    generate_text_simple(model, idx, 10, GPT_CONFIG["context_length"], use_cache)
    start = time.time()
    generate_text_simple(model, idx, num_tokens, GPT_CONFIG["context_length"], use_cache)
    return num_tokens / (time.time() - start)

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    idx = torch.tensor(tokenizer.encode("welcome to my world")).unsqueeze(0)

    # Reference numbers (not measured here), Mac Mini M4, CPU:
    #   tokens/sec without KV cache: 27
    #   tokens/sec with KV cache:    144
    num_bench_tokens = 100
    for use_cache in (False, True):
        tps = tokens_per_sec(model, idx, num_bench_tokens, use_cache)
        print(f"use_cache={use_cache}: {tps:.1f} tokens/sec")
