import os
import argparse
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def set_device():
    device = "cpu"
    if torch.cuda.is_available(): # setting up to cuda if available else cpu
        device = "cuda"
    print(f"Note: device is set to {device}.")
    return device 

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--trace_dir", default=os.path.join(SCRIPT_DIR, "traces/matrix_multiplication"))
    return p.parse_args()

def main():
    args = parse_arguments()
    
    device = set_device()
    dtype = torch.float32
    if args.dtype=="bf16":
        dtype = torch.bfloat16
    print(f"Note: dtype is set to {dtype}.")

    x = torch.randn(args.size, args.size, device=device, dtype=dtype)
    w = torch.randn(args.size, args.size, device=device, dtype=dtype)
    b = torch.randn(args.size, args.size, device=device, dtype=dtype)

    def matrix_multiplication(x, w, b):
        return torch.add(torch.matmul(x, w), b)

    """
    link: https://docs.pytorch.org/docs/2.13/generated/torch.compile.html
    import torch
    print(torch._inductor.list_mode_options())
    Ouptut: {'default': {}, 'lite': {'fallback_by_default': True, 'selective_decompose': True, 'reorder_for_peak_memory': False, 'reorder_for_compute_comm_overlap': False, 'triton.reorder_for_reducing_graph_partitions': False, 'use_pre_grad_passes': False, 'use_joint_graph_passes': False, 'use_post_grad_passes': False, 'use_dce': False, 'allow_buffer_reuse': False}, 'reduce-overhead': {'triton.cudagraphs': True}, 'max-autotune-no-cudagraphs': {'max_autotune': True, 'coordinate_descent_tuning': True}, 'max-autotune': {'max_autotune': True, 'triton.cudagraphs': True, 'coordinate_descent_tuning': True}}
    """

    if args.compile:
        matrix_multiplication =  torch.compile(matrix_multiplication)
    
    def step():
        with torch.profiler.record_function("matmul_add"):
            return matrix_multiplication(x, w, b)
        
    if args.warmup:
        for _ in range(3):
            step()
        # important 
        torch.cuda.synchronize()
    
    os.makedirs(args.trace_dir, exist_ok=True)
    
    compile_tag = "eager"
    if args.compile:
        compile_tag = "compile"
    warmup_tag = "cold"
    if args.warmup:
        warmup_tag = "warm"
       
    tag = f"{args.size}_{args.dtype}_{warmup_tag}_{compile_tag}"
    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=schedule, 
        record_shapes=False, # adds CPU overhead 
        profile_memory=False, # adds GPU overhead 
        with_stack=False 
    ) as prof:
        for _ in range(5):
            step()
            prof.step()
    torch.cuda.synchronize()
    print(f"Note: saving traces at {trace_path}")
    prof.export_chrome_trace(trace_path)
    
    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    main()