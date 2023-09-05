from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union, Literal, Callable, ClassVar
from pydantic import BaseModel, Extra, Field, root_validator

import yaml
import json
from abc import ABC, abstractmethod

import torch


import logging
logger = logging.getLogger(__name__)


def merge(base: Dict, update: Dict, updated: Set) -> Dict:
    "Recursively updates a nested dictionary with new values"
    for k, v in base.items():
        if k in update and isinstance(v, dict):
            base[k] = merge(v, update[k], updated)
            updated.add(k)
        elif k in update:
            base[k] = update[k]
            updated.add(k)

    return base

def _merge_dicts(base: Dict, update: Dict) -> Dict:
    "Merge two dictionaries recursively, returning a new dictionary."

    base = deepcopy(base)

    for k, v in update.items():
        if isinstance(v, dict):
            base[k] = _merge_dicts(base.get(k, {}), v)
        else:
            base[k] = v

    return base


class BaseConfig(BaseModel, ABC):
    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def update(cls, baseconfig: Dict, config: Dict):
        update = {}
        # unflatten a string variable name into a nested dictionary
        # key1.key2.key3: value -> {key1: {key2: {key3: value}}}
        for name, value in config.items():
            if isinstance(value, dict):
                update[name] = value
            else:
                *layers, var = name.split(".")
                if layers:
                    d = update.setdefault(layers[0], {})
                    for layer in layers[1:]:
                        d = d.setdefault(layer, {})
                    d[var] = value

        if not isinstance(baseconfig, Dict):
            baseconfig = baseconfig.to_dict()

        updates = set()
        merged = merge(baseconfig, update, updates)

        for param in update:
            if param not in updates:
                raise ValueError(f"parameter {param} is not present in the config (typo or a wrong config)")

        return cls.from_dict(merged)
    
      
    

class DataConfig(BaseConfig):
    """
    Config for an data.

    :param dataset_paths: dataset paths
    :type dataset_paths: Dict[str, str]

    :param num_workers: Number of works for data preprocessing
    :type name: str
    
    :param overwrite_cache: whether overwrite cach during data preprocessing
    :type overwrite_cache: bool

    :param columns: list of column names 
    :type columns: List[str]

    :param type: type of data ['torch', 'tf', ...]
    :type type: str

    :param num_split : number of splits
    :type num_split : int

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]

    
    """
    dataset: Union[str, List[str]] = ...
    num_workers: str = 1
    overwrite_cache: bool = True
    type: ClassVar[Any]= ...
    num_splits: int = 3
    columns: List[str] = ...
    
    
    

class AlgorithmConfig(BaseConfig):
    """
    Config for an optimizer.

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """

    task: Literal['RLHF', 'SFT'] = 'SFT'

    
    def __post_init__(self, **kwargs):
        for k, v in kwargs:
            if type(v) is dict:
                setattr(self, k, AlgorithmConfig.from_dict(v))
            else:
                setattr(self, k, v)

class PipelineConfig(BaseConfig):
    task: Literal[
        'audio-classification', 
        'automatic-speech-recognition', 
        'conversational', 
        'depth-estimation', 
        'document-question-answering', 
        'feature-extraction', 
        'fill-mask', 
        'image-classification', 
        'image-segmentation', 
        'image-to-text', 'mask-generation', 
        'ner', 'object-detection', 'question-answering', 
        'sentiment-analysis', 'summarization', 'table-question-answering', 
        'text-classification', 'text-generation', 'text-to-audio', 
        'text-to-speech', 'text2text-generation', 'token-classification', 
        'translation', 'video-classification', 'visual-question-answering', 
        'vqa', 'zero-shot-audio-classification', 'zero-shot-classification', 
        'zero-shot-image-classification', 'zero-shot-object-detection', 
        'translation_XX_to_YY'
        ] = "text-generation"
    
    use_cache: bool = True
    device_map: str ="auto"
    do_sample: bool =True
    top_k: int =10
    num_return_sequences: int = 1
    eos_token_id: Optional[Union[str, int]] = None
    pad_token_id: Optional[Union[str, int]] = None


class QuantizationConfig(BaseConfig):
    """
    Config for quantization
 
    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """
    load_in_4bit: bool =True,
    bnb_4bit_compute_dtype: ClassVar[torch.dtype] = torch.float16,
    bnb_4bit_quant_type: str="nf4",
    bnb_4bit_use_double_quant: bool=True,

class LORAConfig(BaseConfig):
    r: int =8, 
    lora_alpha: int =32, 
    target_modules: List[str]=["query_key_value"], 
    lora_dropout: float =0.05, 
    bias: str="none", 
    task_type: str ="CAUSAL_LM"

class ModelConfig(BaseConfig):
    """
    Config for an Model

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """
    id: str = ...
    model_class: str = ...
    torch_dtype: ClassVar[torch.dtype] = ...
    device_map: str = "auto"
    trust_remote_code: bool = True
    
    def __post_init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)
   
class TokenizerConfig(BaseConfig):
    """
    Config for an Model

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """
    id: str
    tokenizer_class: str
    padding_side: str
    padding_token: str
    eos_token: str
    max_length: int = 512
    
    def __post_init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)
    
class LLMConfig(BaseConfig):
    """
    Config for multiple models
    """
    model: ModelConfig = ...
    tokenizer: TokenizerConfig = ...
    pipeline: PipelineConfig = ...
    quantization: Optional[QuantizationConfig] = None
    lora: Optional[LORAConfig] = None
    
    class Config:
        extra = Extra.allow # or 'allow' str
   
    @classmethod
    def from_dict(cls, config: Dict):
        """
        Convert dictionary to DPConfig.
        """
        
        obj = cls(
            model = ModelConfig.from_dict(config.get("model")),
            tokenizer = TokenizerConfig.from_dict(config["tokenizer"]),
            pipeline = PipelineConfig.from_dict(config["pipeline"]) 
        )
        if 'quantization' in config:
            obj.quantization = QuantizationConfig.from_dict(config['quantization'])
        if 'lora' in config:
            obj.lora = LORAConfig.from_dict(config['lora'])
        return obj



  

class BaseTrainConfig(BaseConfig, ABC):
    """
    Config for learn job on model.

    :param episodes: Total number of learning episodes
    :type episodes: int
 
    :param batch_size: Batch size for learning
    :type batch_size: int

    :param tracker: Tracker to use for logging. Default: "wandb"
    :type tracker: str

    :param checkpoint_interval: Save model every checkpoint_interval steps.
        Each checkpoint is stored in a sub-directory of the `LearnerConfig.checkpoint_dir`
        directory in the format `checkpoint_dir/checkpoint_{step}`.
    :type checkpoint_interval: int

    :param eval_interval: Evaluate model every eval_interval steps
    :type eval_interval: int

    :param pipeline: Pipeline to use for learning. One of the registered pipelines present in trlx.pipeline
    :type pipeline: str

    :param learner: learner to use for learning. One of the registered learners present in trlx.learner
    :type learner: str

    :param learner_kwargs: Extra keyword arguments for the learner
    :type learner: Dict[str, Any]

    :param project_name: Project name for wandb
    :type project_name: str

    :param entity_name: Entity name for wandb
    :type entity_name: str

    :param group_name: Group name for wandb (used for grouping runs)
    :type group_name: str

    :param checkpoint_dir: Directory to save checkpoints
    :type checkpoint_dir: str
 
    :param save_best: Save best model based on mean reward
    :type save_best: bool

    :param seed: Random seed
    :type seed: int
     
    """
    

class HFTrainConfig(BaseTrainConfig):
    name: str = 'huggingface'
    epochs: int
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 200
    learning_rate: float = 2e-4
    fp16: bool = True
    logging_steps: int = 1
    output_dir: str = "outputs"
    optim: str = "paged_adamw_8bit"
    warmup_steps: int = 2

class OptimizerConfig(BaseConfig):
    name: str = ...
    def __post_init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)

class SchedulerConfig(BaseConfig):
    name: str = ...
    def __post_init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)

class MosaicTrainConfig(BaseTrainConfig):
    max_duration: str = ...
    optimizer: OptimizerConfig = ...
    schedulers: List[SchedulerConfig] = ...
    device: str
    train_subset_num_batch: int
    prevision: str
    seed: int

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        config['optimizer'] = OptimizerConfig(config['optimizer'])
        if 'scheduler' in config:
            config['schedulers'] = []
            for scheduler in config['scheduler']:
                config['schedulers'].append(SchedulerConfig(scheduler))
        return cls(**config)
            

class TrainConfig(HFTrainConfig, MosaicTrainConfig):
    @classmethod
    def from_dict(cls, config: Dict):
        if 'name' not in config or config['name'] == 'huggingface':
            return HFTrainConfig.from_dict(config)
        elif config['name'] == 'mosaic':
            return MosaicTrainConfig.from_dict(config)
        else:
            raise NotImplementedError


class DPConfig(BaseConfig):
    algorithm: AlgorithmConfig
    llm: LLMConfig 
    train: BaseTrainConfig
    data: DataConfig
    
    @classmethod
    def load_file(cls, fp: str):
        """
        Load file as DPConfig
        
        :param fp: Path to file
        : type fp: str
        """
        if fp.split('.')[-1] == 'json':
            return cls.load_json(fp)
        elif fp.split('.')[-1] == 'yaml' or 'yml':
            return cls.load_yaml(fp)
    @classmethod
    def load_json(cls, json_fp: str):
        """
        Load json file as DPConfig
        
        :param json_fp: Path to json file
        : type json_fp: str
        """
        with open(json_fp, mode='r') as file:
            config = json.safe_load(file)
        return cls.from_dict(config)

    @classmethod
    def load_yaml(cls, yml_fp: str):
        """
        Load yaml file as DPConfig.

        :param yml_fp: Path to yaml file
        :type yml_fp: str
        """
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)

    def to_dict(self):
        """
        Convert DPConfig to dictionary.
        """
        config = {
            "algorithm": self.algorithm.__dict__,
            "llm": self.llm.__dict__,
            "trainer": self.train.__dict__,
            "data": self.data.__dict__
        }
        return config
    
    @classmethod
    def from_dict(cls, config: Dict):
        """
        Convert dictionary to DPConfig.
        """
        return cls(
            algorithm = AlgorithmConfig.from_dict(config.get("algorithm")),
            llm = LLMConfig.from_dict(config["llm"]), 
            train = TrainConfig.from_dict(config["train"]),
            data = DataConfig.from_dict(config['data'])
        )


    def __str__(self):
        """Returns a human-readable string representation of the config."""
        import json

        return json.dumps(self.to_dict(), indent=4)