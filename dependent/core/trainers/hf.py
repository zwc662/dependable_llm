from typing import Any, Dict, List, Optional, Set, Union, Literal, Callable, ClassVar


from dependent.core.utils.register import register
from dependent.core.utils.config import HFTrainConfig
from dependent.core.trainers import BaseTrainer

import transformers
from transformers import TrainingArguments

from torch.utils.data import Dataset  

@register
class HFTrainer(BaseTrainer):

    def __init__(
            self, 
            train_args: TrainingArguments = ...,
            data_collator: Callable = ...
    ):
        self.train_args = train_args
        self.data_collator = data_collator

    

    @classmethod
    def from_config(cls, data_collator: Callable, config: HFTrainConfig):
        train_args = TrainingArguments(
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                learning_rate=config.learning_rate,
                fp16=config.fp16,
                logging_steps=config.logging_steps,
                output_dir=config.output_dir,
                optim=config.optim,
                full_determinism = False,
                sharded_ddp = '',
                fsdp = '',
                fsdp_config = None
            )
        return cls(train_args = train_args, data_collator = data_collator)
    
    def train(
            self, 
            dataset: Dataset,
            model: Callable,  
            save_dir: Optional[str] = None
            ):

        model.train()
        print(type(self.train_args))
        print(self.train_args.full_determinism)
        trainer = transformers.Trainer(
            model=model,
            train_dataset=dataset,
            args=self.train_args,
            data_collator=self.data_collator
        )
        trainer.train()
        if save_dir is not None:
            trainer.save_model(save_dir)
 