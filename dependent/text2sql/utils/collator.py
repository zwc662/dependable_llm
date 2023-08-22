from transformers import DataCollatorForLanguageModeling
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch

@dataclass
class DataCollator(DataCollatorForLanguageModeling):
    def __init__(self, 
        tokenizer, 
        max_length,
        label_pad_token_id 
    ):
        super(DataCollator, self).__init__(
            tokenizer, 
            mlm=False
            )
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        response_token_id = self.tokenizer.encode(" query")[-1]
        #print(response_token_id)
        batch = {"input_ids": torch.tensor([example["input_ids"][-self.max_length:] for example in examples]), 
                 "attention_mask": torch.tensor([example["attention_mask"][-self.max_length:] for example in examples])}     
        #print(self.tokenizer.batch_decode(batch['input_ids'][:1], skip_special_tokens=True))
        labels = torch.clone(batch["input_ids"])
        batch["labels"] = labels
    
        return batch