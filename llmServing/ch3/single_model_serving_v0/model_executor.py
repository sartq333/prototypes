import multiprocessing as mp
from typing import List, Dict, Any
from .model_worker import ModelWorker
from .utils import setup_logging

logger = setup_logging()

class ModelExecutor:
    def __init__(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.worker_process = None 
        logger.debug("[ModelExecutor] init | Initialized with queues")

    def setup_worker(self, model_name: str):
        logger.debug(f"[ModelExecutor] setup_worker | Setting up worker with model: {model_name}")
        self.worker_process = mp.Process(
            target=ModelWorker.run,
            args=(model_name, self.task_queue, self.result_queue)
        )
        logger.debug("[ModelExecutor] setup_worker | Starting worker process")
        self.worker_process.start()
        logger.debug("[ModelExecutor] setup_worker | Worker process started")

    def execute_batch(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not prompts:
            logger.debug("[ModelExecutor] execute_batch | Empty batch received")
            return []
        
        logger.debug(f"[ModelExecutor] execute_batch | Sending batch to worker: {prompts}")
        # send batch to worker
        self.task_queue.put(prompts)
        
        # getting results
        logger.debug("[ModelExecutor] execute_batch | Waiting for results from worker")
        results = self.result_queue.get()
        logger.debug(f"[ModelExecutor] execute_batch | Received results from worker: {results}")
        return results

    def __del__(self):
        if self.worker_process and self.worker_process.is_alive():
            logger.debug("[ModelExecutor] __del__ | Terminating worker process")
            self.worker_process.terminate()
            self.worker_process.join()
            logger.debug("[ModelExecutor] __del__ | Worker process terminated")