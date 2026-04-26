import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
    a = torch.tensor([1, 2, 3])
    print(torch.utils.get_cpp_backtrace())

print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total"))
prof.export_stacks("stacks.txt", "self_cpu_time_total")
prof.export_chrome_trace("trace.json")