from typing import Any, Dict, List, Optional, Set, Union, Literal, ClassVar, Callable
from pydantic import BaseModel, Field, validator
from abc import ABC, abstractmethod

 
from dependent.core.utils.config import DPConfig
from dependent.core.utils.datasplit import DataSplit
from dependent.core.trainers import HFTrainer

# ==============================Change Path=======================================
#import os, sys
#sys.path.append(os.path.os.path.dirname(__file__))
import third_party
import local_datasets
# ================================================================================

import transformers
from transformers import AutoConfig, AutoTokenizer, pipeline, TrainingArguments, AutoModel
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig


import torch
import torch.nn as nn

import importlib

import logging
logger = logging.getLogger(__name__)

 
class dependentBase(ABC):
    def __init__(
            self, 
            config: DPConfig, 
            model: Any = ...,
            tokenizer: Callable = ...,
            pipeline: Callable = ...,
            trainer: Callable = ...,
            data_splits: DataSplit = ...
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.trainer = trainer
        self.data_splits = data_splits
    


    @classmethod
    def from_config(cls, config: DPConfig):
        model = cls.get_model(config)
        tokenizer = cls.get_tokenizer(config)
        pipeline = cls.get_pipeline(model, tokenizer, config)
        trainer = cls.get_trainer(tokenizer, config)
        data_splits = cls.get_dataset_splits(tokenizer, config)
        return cls(
            config = config, 
            model = model, 
            tokenizer = tokenizer, 
            pipeline = pipeline, 
            trainer = trainer, 
            data_splits = data_splits
            )
    
    @classmethod
    def get_pipeline(cls, model: Callable, tokenizer: Callable, config: DPConfig):
        #config.llm.pipeline.eos_token_id = getattr(tokenizer, config.llm.eos_token_id),
        #config.llm.pipeline.pad_token_id = getattr(tokenizer, config.llm.eos_token_id),
        return pipeline(
                model=model,
                tokenizer=tokenizer,
                **config.llm.pipeline.to_dict()
        )
 
    @classmethod
    def get_trainer(cls, tokenizer: Callable, config: DPConfig):
        if config.train.name == 'huggingface':
            trainer = HFTrainer.from_config(
                data_collator = cls.get_collator(tokenizer, config),
                config = config.train
                )
            return trainer
   
    @classmethod
    def get_model(cls, config: DPConfig):
        model_class = getattr(transformers, config.llm.model.model_class)
        model_kwargs = {
            #'torch_dtype': self.config.llm.model.torch_dtype,
            'device_map': config.llm.model.device_map,
            'trust_remote_code': config.llm.model.trust_remote_code
        }
        
        if hasattr(config.llm.model, 'adapter'):
            config = model_class.from_pretrained(
                config.llm.model.id,
                **model_kwargs    
            )
 
            model = PeftModel.from_pretrained(model, config.llm.model.adapter_name, is_trainable=True)

        elif config.llm.quantization is not None:
            logger.info("Loading quantization model")
            quantization_config = BitsAndBytesConfig(**config.llm.quantization.to_dict())
            model_kwargs['quantization_config'] = quantization_config
        
            model = model_class.from_pretrained(
                config.llm.model.id,
                **model_kwargs    
            )

            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)

            if config.llm.lora is not None:
                lora_config = LoraConfig(**config.llm.lora.to_dict())
                model = get_peft_model(model, lora_config)
 
        else:
            model = model_class.from_pretrained(
                config.llm.model.id,
                **model_kwargs    
            )
        return model
 