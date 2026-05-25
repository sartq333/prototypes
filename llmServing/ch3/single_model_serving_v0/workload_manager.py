import uuid 
from typing import List, Dict, Any, Optional
from queue import Queue

class Sequence:
    def __init__(self, seq_id: str, prompt: str):
        self.id = seq_id
        self.prompt = prompt
        self.output = []
        self.finished = False

class WorkloadManager:
    def __init__(self, batch_size: int):
        self.incoming_queue: Queue[Sequence] = Queue()
        self.active_sequences: List[Sequence] = []
        self.batch_size = batch_size 
        self.sequence_map: Dict[str, Sequence] = {}
    
    # for basic and batch generate 
    def add_request(self, prompt: str)->str:
        request_id = str(uuid.uuid4())
        sequence = Sequence(request_id, prompt)
        self.incoming_queue.put(sequence)
        self.sequence_map[request_id] = sequence
        return request_id

    def get_next_batch(self)->List[Sequence]:
        while len(self.active_sequences)<self.batch_size and not self.incoming_queue.empty():
            sequence = self.incoming_queue.get()
            self.active_sequences.append(sequence)
        return self.active_sequences
    
    def remove_active_sequence(self, seq_id: str):
        if seq_id in self.sequence_map:
            sequence = self.sequence_map[seq_id]
            if sequence in self.active_sequences:
                self.active_sequences.remove(sequence)

    def remove_finished_sequence(self, seq_id: str):
        if seq_id in self.sequence_map:
            self.remove_active_sequence(seq_id)
            del self.sequence_map[seq_id]
    
    def is_sequence_finished(self, seq_id: str) -> bool:
        if seq_id in self.sequence_map:
            sequence = self.sequence_map[seq_id]
            return sequence.finished
        return False
    
    def get_sequence(self, seq_id: str) -> Optional[Sequence]:
        return self.sequence_map.get(seq_id)
    
    def update_sequence_output(self, seq_id: str, generated_text: str):
        if seq_id in self.sequence_map:
            sequence = self.sequence_map[seq_id]
            sequence.output.append(generated_text)
            sequence.finished = True
        return None 