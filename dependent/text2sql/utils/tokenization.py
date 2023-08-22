from dependent.core.utils.config import TokenizerConfig
from dependent.text2sql.utils.prompts import Prompt
from pydantic import BaseModel

class tokenization(BaseModel):
    config: TokenizerConfig
    prompt: Prompt = Prompt()
    
    def __call__(self, tokenizer):
        def _preprocess_fn(example):
            nonlocal tokenizer
            inputs = self.prompt(example["serialized_schema"], example["question"], example["query"])
            model_inputs = tokenizer(inputs,
                    max_length=self.config.max_length,
                    padding='max_length',
                    #truncation=True,
                    return_overflowing_tokens=False)
            for k in list(model_inputs.keys()):
                model_inputs[k]=model_inputs[k][:][-self.config.max_length:]
            return model_inputs
        return _preprocess_fn


class tokenization_se(tokenization):
    def __init__(self, config: TokenizerConfig):
        super().__init__(self, config = config)
    def __call__(self, tokenizer):
        def _preprocess_fn(example):
            nonlocal tokenizer
            inputs = self.prompt(example["serialized_schema"], example["question"], example["query"])
            model_inputs = tokenizer(inputs,
                    max_length=self.config.max_length,
                    padding='max_length',
                    #truncation=True,
                    return_overflowing_tokens=False)
            for k in list(model_inputs.keys()):
                model_inputs[k]=model_inputs[k][:][-self.config.max_length:]
            return model_inputs
            #print(example["question"])
            #input_ids = tokenizer(["question:" + question + "|| schema: " + schema + "|| create_table: " + create_table \
            #     for (question, schema, create_table) in zip(example["question"], example["serialized_schema"], example["create_table"])])
            #labels = tokenizer(example["query"])
            #return [{"input_ids": input_id, "labels": label} for (input_id, label) in zip(input_ids, labels)]
        return _preprocess_fn

class tokenization_sparc(tokenization):
    def __init__(self, config: TokenizerConfig, prompt: Prompt):
        super().__init__(self, config, prompt)
    def __call__(self, tokenizer):
        def _preprocess_fn(example):
            nonlocal tokenizer
            kwargs = self.config.to_dict()
            inputs = "given a database schema( " + example["create_table"] + ") | wirte a sql script to answer(" + '- '.join(example["utterances"]) + ") | query: " + example["query"]
            model_inputs = tokenizer(inputs,
                    max_length=self.config.max_length,
                    padding='max_length',
                    #truncation=True,
                    return_overflowing_tokens=False)
            for k in list(model_inputs.keys()):
                model_inputs[k]=model_inputs[k][:][-self.config.max_length,:]
            return model_inputs
            #print(example["question"])
            #input_ids = tokenizer(["question:" + question + "|| schema: " + schema + "|| create_table: " + create_table \
            #     for (question, schema, create_table) in zip(example["question"], example["serialized_schema"], example["create_table"])])
            #labels = tokenizer(example["query"])
            #return [{"input_ids": input_id, "labels": label} for (input_id, label) in zip(input_ids, labels)]
        return _preprocess_fn

class tokenization_sparc_se(tokenization):
    def __init__(self, config: TokenizerConfig, prompt: Prompt):
        super().__init__(self, config, prompt)
    def __call__(self, tokenizer):
        def _preprocess_fn(example):
            nonlocal tokenizer
            kwargs = self.config.to_dict()
            inputs = "given a database schema( " + example["serialized_schema"] + ") | wirte a sql script to answer(" + '- '.join(example["utterances"]) + ") | query: " + example["query"]
            model_inputs = tokenizer(inputs,
                    max_length=self.config.max_length,
                    padding='max_length',
                    #truncation=True,
                    return_overflowing_tokens=False)
            for k in list(model_inputs.keys()):
                model_inputs[k]=model_inputs[k][:][-self.config.max_length,:]
            return model_inputs
            #print(example["question"])
            #input_ids = tokenizer(["question:" + question + "|| schema: " + schema + "|| create_table: " + create_table \
            #     for (question, schema, create_table) in zip(example["question"], example["serialized_schema"], example["create_table"])])
            #labels = tokenizer(example["query"])
            #return [{"input_ids": input_id, "labels": label} for (input_id, label) in zip(input_ids, labels)]
        return _preprocess_fn
