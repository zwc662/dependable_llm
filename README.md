## dependent-llm
A LLM training and validation pipeline

## Installation

`pip install -e .` to install the library

## Example
* Train an LLAMA-2 7b model with text2sql datasets 

```
python -m dependent.dependent.examples.sft_spider
```
 

## Usage 

* Refer to [how the text2sql example set up the configuration](dependent/examples/sft_spider.py)
* Steps: 
    1. **Configuration stage**: create a fine-tune pipeline configuration dict/yaml/json and wrap it with [`DPConfig` class](dependent/core/utils/config.py).
        * The configuration has 4 sections: `algorithm`, `llm`, `train`, and `data`. 
        * Each section is wrapped with the corresponding configuration class.
        * The configuration classes are designed to maximize flexibity in composing different LLM libraries.
        * New arguments required by some LLM library can be easily added in the configuration stage.
    2. **Create fine-tuning pipeline**: create a customized fine-tuning pipeline with the `DPConfig` configuration. 
        * Fine-tuning pipeline for different task can be different. An example is [text2sql pipeline](dependent/text2sql/text2sql.py)
        * For different tasks, dataset adapter may be needed. An example is [text2sql dataset adapter](dependent/text2sql/dataset_adapter)
        * As aforementioned, `DPConfig` class can help add new arguments easily.
    3. **Fine-tunning**: run the fine-tuning pipeline.
        * K-split training: use [`DataSplit` class](dependent/core/utils/datasplit.py) can split, compose, concatenate datasets with ease. 
        * Trainer library: use different [trainers](dependent/core/trainers) for fine-tuning.
        
## TODOs

* Add support for hyperparameter tuning based on the [`DPConfig` class](dependent/core/utils/config.py).  
* Add support for more training APIs in [trainers](dependent/core/trainers) 
* Add `mlflow` to manage the pipeline
* Add `wandb` to monitor the training and evaluation process
* Add RLHF support by referring to [trlx](https://trlx.readthedocs.io/en/latest/)