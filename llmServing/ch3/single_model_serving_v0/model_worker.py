import sys
import logging 
import torch 
import multiprocessing as mp 
from typing import List, Dict, Any, Generator 
from model_manager import ModelManager 
from utils import get_device, setup_logging

logger = setup_logging()

class ModelWorker:
    
    def __init__(self, model_name: str):
        
        self.device = get_device()
        logger.debug(f"[ModelWorker] init Loading model: {model_name} on device: {self.device}.")
        self.model, self.tokenizer = ModelManager(model_name="HuggingFaceTB/SmolLM2-360M").load_model()
        self.model = self.model.to(self.device)
        self.tokenizer.padding_side = "left"  # decoder-only models generate by extending the sequence rightward, so padding must be on the left — right-padding would place real tokens after padding, causing the model to attend to padding before content and produce incorrect output
    
    def generate(self, prompts: List[Dict[str, Any]])->List[Dict[str, Any]]: # simple naive generation
        
        logger.debug(f"[ModelWorker] generate Received prompts: {prompts}.")
        
        prompt_texts = []
        request_ids = []

        # extracting prompt and request ids
        for p in prompts:
            prompt_texts.append(p["prompt"])
            request_ids.append(p["id"])

        # tokenize all prompts in one batch
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        inputs = self.tokenizer(
            text=prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        logger.debug(f"[ModelWorker] generate Batch input shape: {inputs.input_ids.shape}.")

        # generate text for all prompts in one batch 
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                num_return_sequences=1,  # number of independent output sequences to generate per prompt; >1 requires sampling (e.g. temperature) to produce variation
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # decode all outputs 
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.debug(f"[ModelWorker] generate Generated texts: {generated_texts}.")

        results = []
        for request_id, generated_text in zip(request_ids, generated_texts):
            results.append({"request_id": request_id, "generated_text": generated_text})
        
        return results         
    
    def run(model_name: str, task_queue: mp.Queue, result_queue: mp.Queue):
        
        worker = ModelWorker(model_name=model_name)
        logger.debug(f"[ModelWorker] run Worker initialized.")

        while True:
            logger.debug(f"[ModelWorker] run Waiting for batch from queue.")
            batch_data = task_queue.get()
            logger.debug(f"[ModelWorker] run Received batch: {batch_data}")

            if batch_data is None:
                logger.debug("[ModelWorker] run batch_data is None, shutting down.")
                break

            batch = batch_data

            result_queue.put(("complete", worker.generate(batch)))  

if __name__ == "__main__":
    model_worker = ModelWorker(model_name="HuggingFaceTB/SmolLM2-360M")
    prompts = [{"id": "1", "prompt": "hello world"}, {"id": "2", "prompt": "hello smolLM"}]
    results = model_worker.generate(prompts=prompts)
    print(results)