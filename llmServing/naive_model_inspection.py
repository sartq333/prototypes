import gc
import time 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def free_gpu():
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

def initialize_model(model_name): 
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        device_map="auto",
        output_attentions=True 
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    # print(help(AutoModelForCausalLM.from_pretrained))
    # print(help(AutoTokenizer.from_pretrained))
    return model, tokenizer

def attention_visualization(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    print(50*"-", f"\ntokenized input: {inputs}")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attention = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # head_view(attention, tokens)
    layer, head = 5, 10
    attn_map = attention[layer][0][head].float().detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(attn_map, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.tight_layout()
    plt.savefig(f"attention_layer_{layer+1}_head_{head+1}.png")

def simple_inference(model, tokenizer, input_prompt=None, max_new_tokens=100):
    if input_prompt is None:
        input_prompt = "India is seventh largest country in the world, "
    idx = tokenizer(input_prompt, return_tensors="pt").input_ids.to(model.device)
    start_time = total_time = time.time()
    times = []
    first_token_generated = False 
    for _ in range(max_new_tokens):
        idx_cond = idx # current context for generation
        with torch.no_grad():
            outputs = model(idx_cond) # generate prediction for next token
            # note: at iteration 0 `idx_cond` has the full input prompt - prefill phase (parallel).
            # all prompt tokens are processed simultaneously in one forward pass through the model.
            logits = outputs.logits 
            # print(f"logits: {logits}")
            # print(f"logits shape: {logits.shape}")
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        # idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
        idx_next = torch.multinomial(probas, num_samples=1)
        print("Next Token is:", tokenizer.decode(idx_next[0], skip_special_tokens=True))
        time_cost = time.time() - start_time
        times.append(time_cost)
        if not first_token_generated:
            print(f"time taken for the generating the first token: {time_cost:.4f} seconds")
            first_token_generated = True 
        else:
            print(f"time taken for generating a token: {time_cost:.4f} seconds")
        start_time = time.time()
        idx = torch.cat((idx, idx_next), dim=1) # append the new token to the input prompt
        if idx_next.item()==tokenizer.eos_token_id:
            print("generation completed - eos token came")
            break 
    generated_text = tokenizer.decode(idx[0], skip_special_tokens=True)
    print(f"total time taken: {time.time()-total_time:.4f} seconds")
    # both having max new tokens set as 500 ->
    # total time taken: 554.7164 seconds (mps backend)
    # total time taken: 26.6806 seconds (cuda backend)
    # cuda is ~20-22 times faster as compared to apple mps backend
    print(f"generated text: {generated_text}")

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(times)), times, color=['red'] + ['blue'] * (len(times) - 1))
    plt.xlabel("Token ID")
    plt.ylabel("Time Spent in Token Generation")
    plt.title("LLM Generation Times for each token")
    plt.savefig(f"token_generation_time.png")

def kv_cache_enabled_inference(model, tokenizer, input_prompt=None, max_new_tokens=100):
    if input_prompt is None:
        input_prompt = "India is seventh largest country in the world, "
    idx = tokenizer(input_prompt, return_tensors="pt").input_ids.to(model.device)
    start_time = total_time = time.time()
    times = []
    first_token_generated = False 
    past_key_values = None
    for _ in range(max_new_tokens):
        if past_key_values is None:
            idx_cond = idx
        else:
            idx_cond = idx[:, -1:]
        with torch.no_grad():
            outputs = model(input_ids=idx_cond,
                            past_key_values=past_key_values, # use kv cache from previous iteration
                            use_cache=True) # enable caching 
            # print("outputs: ", outputs)
            logits = outputs.logits 
            past_key_values = outputs.past_key_values 

            if model.device=="cuda":
                torch.cuda.synchronize()
            elif model.device=="mps":
                torch.mps.synchronize()

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        # idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
        idx_next = torch.multinomial(probas, num_samples=1)
        print("Next Token is:", tokenizer.decode(idx_next[0], skip_special_tokens=True))
        time_cost = time.time() - start_time
        times.append(time_cost)
        if not first_token_generated:
            print(f"time taken for the generating the first token: {time_cost:.4f} seconds")
            first_token_generated = True 
        else:
            print(f"time taken for generating a token: {time_cost:.4f} seconds")
        start_time = time.time()
        idx = torch.cat((idx, idx_next), dim=1) # append the new token to the input prompt
        if idx_next.item()==tokenizer.eos_token_id:
            print("generation completed - eos token came")
            break 
    generated_text = tokenizer.decode(idx[0], skip_special_tokens=True)
    print(f"total time taken: {time.time()-total_time:.4f} seconds")
    # both having max new tokens set as 500 ->
    # total time taken: 13.8067 seconds - with kv cache (mps backend)
    # total time taken: 2.5959 seconds - with kv cache (cuda backend)
    # cuda is ~6-7 times faster as compared to apple mps backend when kv cache is enabled
    print(f"generated text: {generated_text}")

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(times)), times, color=['red'] + ['blue'] * (len(times) - 1))
    plt.xlabel("Token ID")
    plt.ylabel("Time Spent in Token Generation")
    plt.title("LLM Generation Times for each token")
    plt.savefig(f"kv_enabled_token_generation_time.png")

def calculate_model_memory(config, model):
    # Constants
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    # num_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    vocab_size = config.vocab_size

    if model.dtype==torch.bfloat16:
        bytes_per_param = 2
    else:
        bytes_per_param = 4 
    # memory for embeddings
    embedding_memory = vocab_size*hidden_size*bytes_per_param 
    # memory for each transformer block
    qkv_memory = hidden_size*hidden_size*3*bytes_per_param  # q, k, v projections
    attention_output_memory = hidden_size*hidden_size*bytes_per_param  # output projection
    # mlp 
    mlp_input_memory = hidden_size*intermediate_size*bytes_per_param   # first mlp layer
    mlp_output_memory = intermediate_size*hidden_size*bytes_per_param  # second mlp layer
    # layer norms
    norm_memory = hidden_size*bytes_per_param  # layer normalization parameters
    # total memory per layer
    layer_memory = (qkv_memory + attention_output_memory +
                    mlp_input_memory + mlp_output_memory +
                    norm_memory * 2)  # 2 layer norms per block
    # total model memory
    total_memory = (embedding_memory+layer_memory*num_layers)
    # convert to MB
    total_memory_mb = total_memory/(1024*1024)

    return {
        "Embedding Memory (MB)": embedding_memory / (1024 * 1024),
        "Memory per Layer (MB)": layer_memory / (1024 * 1024),
        "Total Model Memory (MB)": total_memory_mb,
        "Total Model Memory (GB)": total_memory_mb / 1024
    }

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B"
    model, tokenizer = initialize_model(model_name=model_name)
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

    print(50*"-", f"\nmodel size and dtype: {total_params/1000000:.2f} M, {config.dtype}")

    # model memory information:
    memory_stats = calculate_model_memory(config=config, model=model)
    print(50*"-", "\nmemory usage analysis")
    for key, value in memory_stats.items():
        print(f"{key}: {value:.2f}")

    # model architecture information:
    print(50*"-", f"\nmodel architecture: {model}") # gives recursive tree of all nested submodules
    # print(f"\nmodel architecure using module.named_children(): {dict(model.named_children())}") 
    # named children is for when we need to programmatically do something with each layer
    # like applying a hook, freezing specific layer. its a traversal tool.
    # a hook is a function that can be attached to a layer that automatically runs during 
    # forward or backward pass, without modifying the model's code.
    # https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html#torch.nn.Module.named_children

    attention_visualization(model=model, tokenizer=tokenizer, input_text="write a short introduction about the US capital city")
    input_prompt = """The history of human communication is a story of innovation. From ancient cave paintings and spoken language to the invention of writing systems, humans have constantly developed new methods to express ideas and share knowledge. The printing press revolutionized the spread of information, enabling books to be produced and distributed at an unprecedented scale. Centuries later, the invention of the telegraph, radio, and television further transformed how we connect with one another. But perhaps no advancement has reshaped communication more profoundly than the internet.
    Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, information overload, and the nature of human interaction.
    Looking ahead, emerging technologies such as virtual reality, brain-computer interfaces, and artificial intelligence promise to once again redefine how we communicate. As we reflect on this history and anticipate the future, one question arises:
    How might the next wave of communication tools shape our relationships, societies, and sense of identity?"""
    # simple_inference(model=model, tokenizer=tokenizer, input_prompt=input_prompt, max_new_tokens=500)
    # interesting observation to note: the time and decoding speed depends on the length of input prompt. if prompt is short then things are consistent.
    # but if prompt is long then new tokens adds incrementally gradually increasing the time.
    # kv_cache_enabled_inference(model=model, tokenizer=tokenizer, input_prompt=input_prompt, max_new_tokens=500)
    free_gpu(model)
    del model 