import os
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelManager:
    
    def __init__(self, model_name):
        self.model_dir = "model_cache"
        self.model_name = model_name 

    def load_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_name, backend="tokenizers")
        except Exception as e:
            raise(f"Error in loading model or tokenizer: {e}.")
        return model, tokenizer

if __name__ == "__main__":
    model_manager = ModelManager(model_name="HuggingFaceTB/SmolLM2-360M")
    model, tokenizer = model_manager.load_model()
    print("Model loaded successfully: ", model)
    print("Tokenizer initialized successfully: ", tokenizer)