from typing import List, Dict, Any
from .workload_manager import WorkloadManager, Sequence
from .model_executor import ModelExecutor
import atexit
import threading
import time

class LLMEngine:
    def __init__(self):
        self.model_executor = ModelExecutor()
        self.workload_manager = WorkloadManager(batch_size=4)
        self.max_tokens = 50
        self.sleeper = 0.5 
        self.model_executor.setup_worker(model_name="HuggingFaceTB/SmolLM2-360M")
        self.thread = threading.Thread(target=self.requests_processing_loop, daemon=True)
        self.thread.start()
        atexit.register(self._cleanup)

    def requests_processing_loop(self):
        while True:
            try:
                active_sequences = self.workload_manager.get_next_batch()
                if not active_sequences:
                    time.sleep(self.sleeper)
                    continue
                prompts = []
                for sequence in active_sequences:
                    prompts.append({"prompt": sequence.prompt, "id": sequence.id})
                prompts_results = self.model_executor.execute_batch(prompts=prompts)
                for result in prompts_results:
                    self.workload_manager.update_sequence_output(result["request_id"], result["generated_text"])
                    self.workload_manager.remove_active_sequence(result["request_id"])
                    
            except Exception as e:
                print(f"[LLMEngine] request_processing_loop | Error in processing loop: {e}.")

    def _cleanup(self):
        pass 

    def _is_batch_finished(self, request_ids: List[str])->bool:
        for id in request_ids:
            if not self.workload_manager.is_sequence_finished(id):
                return False
        return True 
    
    def generate(self, prompts: List[str])->List[str]:
        request_ids = []
        for prompt in prompts:
            request_id = self.workload_manager.add_request(prompt=prompt)
            request_ids.append(request_id)

        while not self._is_batch_finished(request_ids):
            time.sleep(self.sleeper)

        generated_texts = []
        for request_id in request_ids:
            generated_texts.append(self.workload_manager.get_sequence(request_id).output)
            self.workload_manager.remove_finished_sequence(request_id)
        return generated_texts 