import gc
import time
import torch
from transformers import pipeline 
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def free_gpu(model):
    if model:
        del model

    device = get_device()
    if device=="cuda":
        torch.cuda.empty_cache()
    elif device=="mps":
        torch.mps.empty_cache()

    gc.collect()


def get_model_size(model): 
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    trust_remote_code=True,
    device_map="auto"
)
device = get_device()
model = model.to(device)

# inspection of model configuration: 
config = model.config

print("architecture parameters:")
print(f"hidden size: {config.hidden_size}")
print(f"number of layers: {config.num_hidden_layers}")
print(f"number of attention heads: {config.num_attention_heads}")
print(f"intermediate size: {config.intermediate_size}")

# inspection of tokenizer: 
print(50*"-", "\ntokenizer parameters:")
print(f"vocabulary parameters: {config.vocab_size}")
print(f"max pos embeddings: {config.max_position_embeddings}")

# model size information:
total_params = get_model_size(model)
print(50*"-", f"\nmodel size: {total_params/1000000:.2f} M")

# print(50*"-", "\n model's config:")
# for k, v in config.to_dict().items():
#     if k in ["architectures", "model_type", "dtype"]:
#         print(f"{k}: {v}")

# model architecture information:
print(50*"-", f"\nmodel architecture: {model}") # gives recursive tree of all nested submodules
# print(f"\nmodel architecure using module.named_children(): {dict(model.named_children())}") 
# named children is for when we need to programmatically do something with each layer
# like applying a hook, freezing specific layer. its a traversal tool.
# a hook is a function that can be attached to a layer that automatically runs during 
# forward or backward pass, without modifying the model's code.
free_gpu(model)