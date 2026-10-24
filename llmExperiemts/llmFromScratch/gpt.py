import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:1+max_length]
            target_chunk = token_ids[i+1:i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size, max_length, stride, shuffle, drop_last, num_workers):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader 

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear( 4*cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out 
        self.num_heads = num_heads 
        self.head_dim = d_out // num_heads 
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x, use_cache=False, cached_kv=None):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        if use_cache and cached_kv is not None:
            keys = torch.cat((cached_kv[0], keys), dim=1)
            values = torch.cat((cached_kv[1], values), dim=1)

        if use_cache:
            cached_kv = (keys, values)
        num_keys = keys.shape[1]
        keys = keys.view(b, num_keys, self.num_heads, self.head_dim)
        values = values.view(b, num_keys, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)        

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries@keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_keys, :num_keys][-num_tokens:]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights@values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec, cached_kv

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention( 
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ffn = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False, cached_kv=None):
        shortcut = x 
        x = self.norm1(x)
        x, cached_kv = self.attention(x, use_cache, cached_kv)
        x = self.drop_shortcut(x)
        x += shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x += shortcut
        return x, cached_kv

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.cached_kv = None   # one (keys, values) per layer, built on the first use_cache call

    def reset_kv_cache(self):
        self.cached_kv = None

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape

        # tokens already in the cache = positions the new tokens must continue from
        past_len = 0
        if use_cache and self.cached_kv is not None:
            past_len = self.cached_kv[0][0].shape[1]   # layer 0 keys: (b, tokens_so_far, d_out)

        if use_cache and self.cached_kv is None:
            self.cached_kv = [None] * len(self.trf_blocks)

        tok_embeds = self.tok_emb(in_idx)
        # prefill: seq_len = prompt length, positions 0..seq_len-1
        # decode:  seq_len = 1, so this is just the single position past_len
        pos_embeds = self.pos_emb(torch.arange(past_len, past_len + seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for i, block in enumerate(self.trf_blocks):
            x, layer_kv = block(x, use_cache, self.cached_kv[i] if use_cache else None)
            if use_cache:
                self.cached_kv[i] = layer_kv
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size, use_cache):
    model.reset_kv_cache()   # every generation starts with an empty cache
    for _ in range(max_new_tokens):
        if use_cache and model.cached_kv is not None:
            idx_cond = idx[:, -1:]
        else:
            idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond, use_cache)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx 

if __name__ == "__main__":
    GPT_CONFIG = {
        "vocab_size": 50257, 
        "context_length": 1024, 
        "emb_dim": 768, 
        "n_heads": 12, 
        "n_layers": 12, 
        "drop_rate": 0.1, 
        "qkv_bias": False,
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "welcome to my world"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print(f"Input text: {start_context}")
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG["context_length"],
        use_cache=True
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(out)
    print(f"output text: {decoded_text}")

"""
import torch
import torch.nn as nn

context_length = 10 # 256, kept it 10 just for representation purposes
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print("mask: ", mask)

num_keys = 4 # total length - including the current one
num_tokens = 1
mask_bool_1 = mask.bool()[:num_keys, :num_keys][-num_tokens:]
mask_bool_2 = mask.bool()[:num_keys, :num_keys]
print("mask_bool_1: ", mask_bool_1)
print("mask_bool_2: ", mask_bool_2)

mask:  tensor([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
mask_bool_1:  tensor([[False, False, False, False]])
mask_bool_2:  tensor([[False,  True,  True,  True],
                      [False, False,  True,  True],
                      [False, False, False,  True],
                      [False, False, False, False]])
"""