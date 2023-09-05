from dependent.core.utils.config import DPConfig
from dependent.text2sql import Text2SQL

import tqdm
import torch
 
import logging
logger = logging.getLogger(__name__)

from golden_eval_examples import examples


# Set up configurations

### Training algorithm configuration
# 'task': choose training algorithm, e.g., supervised fine-tuning (SFT)

algorithm_config = {
    'task': 'SFT'
}

### Model configuration
# 'id': the link of the model on huggingface 
# 'model_class': the class of the model on huggingface
# 'torch_dtype': the model storage type, e.g., fp16, bf16
# 'device_map': select the device to run the model, e.g., 'auto', 'cuda:0'
# 'trust_remote_code": an argument from the huggingface API

model_config = {
    'id': 'codellama/CodeLlama-7b-hf', 
    #'gpt2', #"vilsonrodrigues/falcon-7b-instruct-sharded",
    'model_class': 'AutoModelForCausalLM', #'GPT2LMHeadModel', #'AutoModelForCausalLM',
    "torch_dtype": torch.bfloat16,
    'device_map': 'auto',
    "trust_remote_code": True,
}

### Quantization configuration
# The arguments are required by the `bitsandbytes` library

quantization_config = {
    'load_in_4bit': True,
    'bnb_4bit_quant_type': "nf4",
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_compute_dtype': torch.float16
}

### LORA configuration
# The arguments are required by the `peft` library

lora_config = {
    'r': 8, 
    'lora_alpha': 32, 
    'target_modules': ["query_key_value"], 
    'lora_dropout' : 0.05, 
    'bias': "none", 
    'task_type': "CAUSAL_LM"
}

### Tokenization configuration
# 'id': the link of the model on huggingface 
# 'tokenizer_class': the class of the tokenizer on huggingface
# other arguments: refer to the huggingface tokenizer API

tokenizer_config = {
    'id': "codellama/CodeLlama-7b-hf", 
    #'gpt2', #"vilsonrodrigues/falcon-7b-instruct-sharded",
    'tokenizer_class': 'AutoTokenizer',
    #'GPT2Tokenizer',
    'padding_side': 'left',
    'padding_token': 'eos_token',
    'eos_token': 'eos_token',
    'max_length': 512,
}

### Pipeline configuration
# 'eos_token_id': the token id of the string that marks the end of the text,
# 'pad_token_id': the token id of the string that pads the string to the max length

pipeline_config = {
    'eos_token_id': 'eos_token_id',
    'pad_token_id': 'eos_token_id'
}
### Optimizer configuration
# So far not configured because Huggingface trainer does not provide freedom in configuring the optimizer
# Will have more freedom in configuration if using Mosaic, Zero, ...

optimizer_config = {}

### Trainer configuration
# 'name': the name of the training library, e.g. huggingface, mosaic, ...
# 'epoch': the number of epochs for training
# other arugments: refer to the training library's API

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

### Dataset configuration
# 'dataset': the name of the dataset. Need local support. Currently only support spider, cosql, and sparc
# 'num_workers': the number of workers for preprocessing the dataset. An argument required by pyarrow.Dataset API.
# `num_splits': number of splits in the dataset for k-fold cross validation
# `columns`: the columns to keep in the training set
# 'type': the data type of the training set

data_config = {
    'dataset': ['spider'],
    'num_workers': 8, 
    'num_splits': 1,
    'columns': ['input_ids', 'attention_mask', 'label'],
    'type': torch
}

### Dependent configuration
# 'algorithm': the algorithm configuration
# 'llm': the ensemble of the model, tokenizer, quantization, and pipeline configurations
# 'train': the trainer configuration
# 'data': the data configuration


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

# Create the trainig pipeline configuration
text2sql_config = DPConfig.from_dict(dependent_config)

# Create a training pipeline
agent = Text2SQL.from_config(text2sql_config)


# Run the training pipeline for certain epochs
for i in tqdm.tqdm(range(1, agent.config.train.epochs)):
    # Evaluate the model with example
    for example in examples:
        print(agent.pipeline(example))
    # Run fine-tuning for one epoch
    train_info = agent.run(episodes = 1)
    
    # Record training info
    if i % log_interval == 0:
        for k, v in train_info.items():
            logger.info(f'training/{k}', v, i)

# Evaluate the model with example
for example in examples:
    print(agent.pipeline(example))