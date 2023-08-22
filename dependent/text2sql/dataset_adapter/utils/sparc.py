import json
import numpy as np
from typing import Optional, List
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .dataset import DataArguments, normalize, serialize_schema
from .consts import (
    QUESTION_KEY,
    SCHEMA_KEY,
    RESPONSE_KEY, END_KEY
)

def sparc_get_input(
    utterances: List[str],
    serialized_schema: str,
    prefix: str,
    sep: str = " | ",
) -> str:
    # "[prefix] [utterance n] [serialized schema] || [utterance n-1] | [utterance n-2] | ..."
    # if len(utterances) > 1:
    #     reversed_utterance_head = (utterance.strip() for utterance in reversed(utterances[:-1]))
    #     serialized_reversed_utterance_head = " || " + sep.join(reversed_utterance_head)
    # else:
    #     serialized_reversed_utterance_head = ""
    # return prefix + utterances[-1].strip() + " " + serialized_schema.strip() + serialized_reversed_utterance_head

    # "[prefix] [utterance 1] | [utterance 2] | ... [utterance n] [serialized schema]" truncted in head
    if len(utterances) > 1:
        utterance_head = (utterance.strip() for utterance in utterances[:-1])
        serialized_utterance_head = sep.join(utterance_head) + " | "
    else:
        serialized_utterance_head = ""
    return prefix + QUESTION_KEY + serialized_utterance_head + utterances[-1].strip() + " " + SCHEMA_KEY + serialized_schema.strip() + RESPONSE_KEY 
 
  

def sparc_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def sparc_add_serialized_schema(
    ex: dict,
    data_args: DataArguments,
) -> dict:
    if ex["create_table"] == "" or data_args.schema_serialization_type != 'raw':
        serialized_schema = serialize_schema(
            question=" | ".join(ex["utterances"]),
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            schema_serialization_type= 'custom' if data_args.schema_serialization_type == 'raw' else data_args.schema_serialization_type,
            schema_serialization_randomized=data_args.schema_serialization_randomized,
            schema_serialization_with_db_id=data_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=data_args.schema_serialization_with_db_content,
            normalize_query=data_args.normalize_query,
        )
        return {"serialized_schema": serialized_schema}
    else:
        return {"serialized_schema": ex["create_table"].strip()}

def sparc_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_args: DataArguments,
    tokenizer: PreTrainedTokenizerBase,
    preprocess_type: str
) -> dict:
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    inputs = [
        sparc_get_input(utterances=utterances, serialized_schema=serialized_schema, prefix=prefix)
        for utterances, serialized_schema in zip(batch["utterances"], batch["serialized_schema"])
    ]

    targets = [
        sparc_get_target(
            query=query,
            db_id=db_id,
            normalize_query=data_args.normalize_query,
            target_with_db_id=data_args.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

  
    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding='max_length',
        #truncation=True,
        return_overflowing_tokens=False,
    )
    
    for k in list(model_inputs.keys()):
        for i in range(len(model_inputs[k])):
            model_inputs[k][i] = model_inputs[k][i][-max_source_length:].copy()
            #print(len(model_inputs[k][i]))
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding='max_length',
            #truncation=True,
            return_overflowing_tokens=False
        )
        for k in list(labels.keys()):
            for i in range(len(labels[k])):
                labels[k][i] = labels[k][i][-max_target_length:].copy()
                #print(len(labels[k][i]))
    if preprocess_type == 'causalml':
        model_inputs["input_ids"] = [input_id + label for (input_id, label) in zip(model_inputs['input_ids'], labels["input_ids"])] 
        model_inputs["labels"] = model_inputs["input_ids"].copy()
    elif preprocess_type == 'seq2seq':
        model_inputs["labels"] = labels["input_ids"]
    #print([len(input_id) for input_id in model_inputs['input_ids']])

    return model_inputs
 