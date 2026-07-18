import argparse
import math 
import os 
import torch 
import torch.nn as nn 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class NaiveAttention(nn.Module):

    def __init__(self, head_dim):
        super().__init__()
        self.scale = 1/math.sqrt(head_dim)

    def forward(self, q, k, v, mask):
        # q, k, v: [batch, heads, seq, head_dim]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = torch.mul(scores, self.scale)
        # scores = scores.masked_fill(mask, float("-inf"))
        scores = scores.masked_fill_(mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        return output
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--trace_dir", default=os.path.join(SCRIPT_DIR, "traces/attention"))
    args = p.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    shape = (args.batch, args.heads, args.seq, args.head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    mask = torch.triu(torch.ones(args.seq, args.seq, device=device, dtype=torch.bool), diagonal=1)

    attn = NaiveAttention(args.head_dim).to(device=device, dtype=dtype)

    if args.compile:
        fwd = torch.compile(attn)
        compile_tag = "compile"
    else:
        fwd = attn
        compile_tag = "eager"

    def step():
        with torch.profiler.record_function("attn_forward"), torch.no_grad():
            return fwd(q, k, v, mask)
        
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    tag = f"{args.batch}_{args.heads}_{args.seq}_{args.head_dim}_{compile_tag}"
    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
        schedule=schedule, 
        record_shapes=False, 
        profile_memory=False, 
        with_stack=False
    ) as prof:
        for _ in range(5):
            step()
            prof.step()
    torch.cuda.synchronize()
    print(f"saving traces: {trace_path}")
    prof.export_chrome_trace(trace_path)

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    print("table: ", table)

    with open(table_path, "w") as f:
        f.write(table)

if __name__ == "__main__":
    main()