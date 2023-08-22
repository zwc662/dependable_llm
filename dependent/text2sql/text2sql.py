
from typing import Any, Dict, List, Optional, Set, Union, Literal, ClassVar
from pydantic import BaseModel, Field, validator
from functools import partial

from dependent.core.utils.config import DPConfig
from dependent.core.utils.datasplit import DataSplit
from dependent.core.base import dependentBase

from dependent.text2sql.utils.tokenization import tokenization, tokenization_se, tokenization_sparc, tokenization_sparc_se
from dependent.text2sql.utils.collator import *

from dependent.text2sql.dataset_adapter.utils.dataset import DataArguments
from dependent.text2sql.dataset_adapter.utils.dataset_loader import load_dataset

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


class Text2SQL(dependentBase):
    def __init__(self, *args, **kwargs):
        super(Text2SQL, self).__init__(*args, **kwargs)

    
    @classmethod    
    def get_dataset_splits(cls, tokenizer: Callable, config: DPConfig): 
        train_splits = None
        for dataset_name in config.data.dataset:
            data_args = DataArguments(
                preprocessing_num_workers = config.data.num_workers,  
                dataset = dataset_name,
                cache_dir = './transformer_cache/',
                test_sections = 'validate',
                schema_serialization_type = 'custom'
            )
            data_args.do_train = True
            data_args.do_predict = True
            data_args.do_eval = False
            dataset = load_dataset(data_args)[1].train_split.dataset
            cols2rmv = set(dataset.column_names)
            for col in config.data.columns:
                if col not in ['input_ids', 'label', 'attention_mask']:
                    cols2rmv.remove(col)
            
            tokenization_fn = tokenization(config = config.llm.tokenizer)(tokenizer)
            
            train_data = dataset.map(tokenization_fn).remove_columns(list(cols2rmv))
        
            if train_splits is None:
                train_splits = DataSplit.split_dataset(train_data, config.data.num_splits)
            else:
                train_splits.append(DataSplit.split_dataset(train_data, config.data.num_splits))

        return train_splits   
        train_data=torch.utils.data.ConcatDataset([train_data1, train_data2, train_data3, train_data4])
        test_data=torch.utils.data.ConcatDataset([test_data1, test_data2, test_data3, test_data4])
    

    @classmethod
    def get_collator(cls, tokenizer: Callable, config: DPConfig):
        data_collator=DataCollator(
            tokenizer = tokenizer, 
            max_length = config.llm.tokenizer.max_length,
            label_pad_token_id = getattr(tokenizer, config.llm.pipeline.pad_token_id)
            )
        return data_collator
        
    
    @classmethod
    def get_tokenizer(cls, config: DPConfig):
        tokenizer_class = getattr(transformers, config.llm.tokenizer.tokenizer_class)
        tokenizer = AutoTokenizer.from_pretrained(config.llm.tokenizer.id)
        tokenizer.padding_side = config.llm.tokenizer.padding_side
        tokenizer.pad_token = getattr(tokenizer, config.llm.tokenizer.padding_token)
        tokenizer.eos_token = config.llm.tokenizer.eos_token
        return tokenizer

    def run(self, episodes: int = 1, final_train: bool = True):
        # Run the agent to get experiences
        # Build model dataset 
        # K-split the model dataset and train the detector for multiple rounds 
        # Return the mean metric 
        best_score = None
        best_exps = None
        best_loss_fn = None
        best_validation_info = None
        best_dataset = None
        #with mlflow.start_run as run:
        if True:
            for epoch in range(1, episodes + 1):
                best_score = None
                best_exps = None
                 
                suffix_split = self.data_splits
                prefix_split = None
                for _ in range(1, self.config.data.num_splits):
                    validation_set = suffix_split.head
                    suffix_split = suffix_split.tail
                    if prefix_split is None:
                        train_set = suffix_split.compose()
                    else:
                        train_set = DataSplit.concatenate(prefix_split, suffix_split).compose()
                    self.data_splits = None
    
                    #self.logger.epoch_info("Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch))
                    train_info = self.train(train_set, self.model)
                    print(train_info)
                    #for k, v in train_info.items():
                    #    mlflow.log_metric(k, v, step = epoch)
                    #validation_info = self.learner.evaluate(self.logger, validation_set, metrics_fn)
                    #for k, v in validation_info.items():
                    #    mlflow.log_metric(k, v, step = epoch)
                    
                    #score = validation_info.get(self.config.algorithm.metrics[0])
                    #if best_score is None or best_score < score:
                    #    best_score, best_exps, best_validation_info, best_dataset = score, exps, validation_info, dataset
            #if final_train:
                #final_info = self.learner.train(self.logger, best_dataset, best_loss_fn, self.optimizer)
                #for k, v in final_info.items():
                #    mlflow.log_metric(k, v, step = self.epochs + 1)
            #mlflow.end_run()
            #mlflow.log_artifacts(self.result_dir, artifact_path="configure_events")

    
  
        