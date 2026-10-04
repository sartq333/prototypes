import argparse
import os 
import torch 
import torch.nn as nn 
from torch.nn import functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class SimpleGeGLUMLP(nn.Module):
    
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = F.gelu(g, approximate="tanh")
        m = h*u
        y = self.down_proj(m)
        return y 
    
def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--batch", type=int, default=64)
    arg.add_argument("--seq", type=int, default=128)
    arg.add_argument("--dim", type=int, default=768)
    arg.add_argument("--hidden", type=int, default=3072)
    arg.add_argument("--compile", action="store_true")
    arg.add_argument("--trace_dir", default=os.path.join(SCRIPT_DIR, "traces/mlpGLU"))
    args = arg.parse_args()
    
    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda"
    x = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=torch.bfloat16)
    model = SimpleGeGLUMLP(args.dim, args.hidden).to(device=device, dtype=torch.bfloat16)
    model.eval()

    if args.compile:
        fwd = torch.compile(model)
        compile_tag = "compile"
    else:
        fwd = model
        compile_tag = "eager"
    tag = f"{args.batch}_{args.seq}_{args.dim}_{args.hidden}_{compile_tag}"

    def step():
        with torch.profiler.record_function("mlp_fwd"), torch.no_grad():
            return fwd(x)
    
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU, 
            torch.profiler.ProfilerActivity.CUDA
        ],
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
    print(table)
    
    with open(table_path, "w") as f:
        f.write(table)

if __name__ == "__main__":
    main()