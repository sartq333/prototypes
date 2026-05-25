# single model serving v0 (without streaming and batching)

`ModelManager` - Loads/caches the actual model and tokenizer

`ModelWorker` - Runs in separate process, generates next tokens using the model

`WorkloadManager` - Orchestrates the pipeline: decides which sequences go in next batch, tracks progress, knows when to pull more from queue

`ModelExecutor` - Manages the worker process itself (spawn it, communicate with it via queues)

`LLMEngine` - Top-level wrapper that coordinates everything together

LLMEngine (llm.py)

    ↓

    └─→ ModelExecutor (model_executor.py)
            
            ↓

            ├─→ Initializes/manages ModelWorker process

            ├─→ Sends batches to ModelWorker via queues

            └─→ Receives generated tokens back

                    ↓

                    ModelWorker (model_worker.py)

                            ↓

                            ├─→ Uses ModelManager to load model/tokenizer

                            └─→ Generates next tokens

WorkloadManager (workload_manager.py)

    ├─→ Tracks incoming requests (incoming_queue)

    ├─→ Manages active sequences being processed

    ├─→ Decides when to form the next batch (via get_next_batch)
    
    └─→ Tracks when sequences are finished