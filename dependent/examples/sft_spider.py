from dependent.core.utils.config import DPConfig
from dependent.text2sql import Text2SQL

import tqdm
import torch
 
import logging
logger = logging.getLogger(__name__)


algorithm_config = {
    'task': 'SFT'
} 
model_config = {
    'id': 'gpt2', #"vilsonrodrigues/falcon-7b-instruct-sharded",
    'model_class': 'GPT2LMHeadModel', #'AutoModelForCausalLM',
    "torch_dtype": torch.bfloat16,
    'device_map': 'auto',
    "trust_remote_code": True,
}
quantization_config = {
    'load_in_4bit': True,
    'bnb_4bit_quant_type': "nf4",
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_compute_dtype': torch.float16
}
lora_config = {
    'r': 8, 
    'lora_alpha': 32, 
    'target_modules': ["query_key_value"], 
    'lora_dropout' : 0.05, 
    'bias': "none", 
    'task_type': "CAUSAL_LM"
}

tokenizer_config = {
    'id': 'gpt2', #"vilsonrodrigues/falcon-7b-instruct-sharded",
    'tokenizer_class': 'GPT2Tokenizer',
    'padding_side': 'left',
    'padding_token': 'eos_token',
    'eos_token': 'eos_token',
    'max_length': 512,
}
pipeline_config = {
    'eos_token_id': 'eos_token_id',
    'pad_token_id': 'eos_token_id'
}
optimizer_config = {}
train_config = {
    'name': 'huggingface',
    'epochs': 10,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 4,
    'max_steps': 200,
    'learning_rate': 2e-4,
    'fp16': True,
    'logging_steps': 1,
    'output_dir': 'outputs',
    'optim': "paged_adamw_8bit"
}
    
data_config = {
    'dataset': ['spider'],
    'num_workders': 8, 
    'num_splits': 5,
    'columns': ['input_ids', 'attention_mask', 'label'],
    'type': torch
}

dependent_config = {
    'algorithm': algorithm_config,
    'llm': {
        'model': model_config,
        'tokenizer': tokenizer_config,
        #'quantization': quantization_config,
        'pipeline': pipeline_config,
        #'lora': lora_config
    }, 
    'train': train_config,
    'data': data_config
}

log_interval = 10
eval_interval = 100

text2sql_config = DPConfig.from_dict(dependent_config)
agent = Text2SQL.from_config(text2sql_config)

for i in tqdm.tqdm(range(1, agent.config.train.epochs)):
    train_info = agent.run(episodes = 1)
    if i % log_interval == 0:
        for k, v in train_info.items():
            logger.info(f'training/{k}', v, i)

 