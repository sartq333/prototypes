import argparse 
import os 
import torch 
from torch import nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--in_dim", type=int, default=32)
    p.add_argument("--out_dim", type=int, default=64)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--trace_dir", default=os.path.join(SCRIPT_DIR, "traces/mlp"))
    args = p.parse_args()

    os.makedirs(args.trace_dir, exist_ok=True)

    device = "cuda"
    x = torch.randn(args.batch, args.in_dim, device=device, dtype=torch.bfloat16)
    linear_layer = nn.Linear(args.in_dim, args.out_dim, bias=True).to(device=device, dtype=torch.bfloat16)
    linear_layer.eval()
    print(linear_layer.weight.shape)
    print(linear_layer.bias.shape)
    
    if args.compile:
        fwd = torch.compile(linear_layer)
    else:
        fwd = linear_layer 

    def step():
        with torch.profiler.record_function("linear_fwd"), torch.no_grad():
            return fwd(x)

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    if args.compile:
        compile_tag = "compile"
    else:
        compile_tag = "eager"

    tag = f"{args.batch}_{args.in_dim}_{args.out_dim}_{compile_tag}"
    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU, 
            torch.profiler.ProfilerActivity.CUDA, 
        ], 
        schedule=schedule, 
        record_shapes=False, 
        profile_memory=False, 
        with_stack=False 
    ) as prof:
        for _ in range(5):
            step()
            prof.step()

    torch.cuda.synchronize() # important 

    print(f"saving traces: {trace_path}")
    prof.export_chrome_trace(trace_path)

    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    main()