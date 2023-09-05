from distutils.util import split_quoted
from typing import Optional, List, Dict, Callable
from functools import partial
from dataclasses import dataclass, field
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
 
from .bridge_content_encoder import get_database_matches

import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from legion.text2sql.dataset_adapter.preprocess.choose_dataset import preprocess_by_dataset
import re
import random

import inspect
import local_datasets

__LOCAL_DATASET_PATH__ =  os.path.dirname(inspect.getfile(local_datasets))
__SPIDER_LOCAL_DATASET_PATH__ = os.path.join(__LOCAL_DATASET_PATH__, "spider")
__SPARC_LOCAL_DATASET_PATH__ = os.path.join(__LOCAL_DATASET_PATH__, "sparc")
__COSQL_LOCAL_DATASET_PATH__ = os.path.join(__LOCAL_DATASET_PATH__, "cosql")


def __post_init__(self):
    if self.val_max_target_length is None:
        self.val_max_target_length = self.max_target_length

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation or test examples to this "
            "value if set."
        },
    ) 
    schema_serialization_type: str = field(
        default="raw",
        metadata={"help": "Choose between ``verbose`` and ``peteshaw`` schema serialization."},
    )
    schema_serialization_randomized: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomize the order of tables."},
    )
    schema_serialization_with_db_id: bool = field(
        default=True,
        metadata={"help": "Whether or not to add the database id to the context. Needed for Picard."},
    )
    schema_serialization_with_db_content: bool = field(
        default=True,
        metadata={"help": "Whether or not to use the database content to resolve field matches."},
    )
    normalize_query: bool = field(default=True, metadata={"help": "Whether to normalize the SQL queries."})
    target_with_db_id: bool = field(
        default=True,
        metadata={"help": "Whether or not to add the database id to the target. Needed for Picard."},
    )
    wandb_project_name : Optional[str] = field(
        default="text-to-sql",
        metadata={"help": "Base path to the lge relation dataset."})
    dataset: str = field(
        default=None,
        metadata={"help": "The dataset to be used. Choose between ``spider``, ``cosql``, or ``cosql+spider``."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": __SPIDER_LOCAL_DATASET_PATH__,
            "sparc": __SPARC_LOCAL_DATASET_PATH__,
            "cosql": __COSQL_LOCAL_DATASET_PATH__,
        },
        metadata={"help": "Paths of the dataset modules."},
    )
    metric_config: str = field(
        default="both",
        metadata={"help": "Choose between ``exact_match``, ``test_suite``, or ``both``."},
    )
    metric_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics/spider"),
            "sparc": os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics/sparc"),
            "cosql": os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics/cosql"),
        },
        metadata={"help": "Paths of the metric modules."},
    )
    test_suite_db_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test-suite databases."})
    data_config_file : Optional[str] = field(
        default=None,
        metadata={"help": "Path to data configuration file (specifying the database splits)"}
    )
    test_sections : Optional[List[str]] = field(
        default=None,
        metadata={"help": "Sections from the data config to use for testing"}
    )
    data_base_dir : Optional[str] = field(
        default="./dataset_files/",
        metadata={"help": "Base path to the lge relation dataset."})
    split_dataset : Optional[str] = field(
        default="",
        metadata={"help": "The dataset name after spliting."})
    


@dataclass
class TrainSplit(object):
    dataset: Dataset
    schemas: Dict[str, dict]


@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset
    schemas: Dict[str, dict]


@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_splits: Optional[Dict[str, EvalSplit]]
    schemas: Dict[str, dict]


def _get_schemas(examples: Dataset) -> Dict[str, dict]:
    schemas: Dict[str, dict] = dict()
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = {
                "db_table_names": ex["db_table_names"],
                "db_column_names": ex["db_column_names"],
                "db_column_types": ex["db_column_types"],
                "db_primary_keys": ex["db_primary_keys"],
                "db_foreign_keys": ex["db_foreign_keys"],
            }
    return schemas


def _prepare_train_split(
    dataset: Dataset,
    data_args: DataArguments,
    add_serialized_schema: Callable[[dict], dict]
) -> TrainSplit:
    schemas = _get_schemas(examples=dataset)
    dataset = dataset.map(
        add_serialized_schema,
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    
    if data_args.max_train_samples is not None:
        dataset = dataset.select(range(data_args.max_train_samples))
    column_names = dataset.column_names
     
    return TrainSplit(dataset=dataset, schemas=schemas)


def _prepare_eval_split(
    dataset: Dataset,
    data_args: DataArguments,
    add_serialized_schema: Callable[[dict], dict]
) -> EvalSplit:
    if (data_args.max_val_samples is not None 
            and data_args.max_val_samples < len(dataset)):
        eval_examples = dataset.select(range(data_args.max_val_samples))
    else:
        eval_examples = dataset
    schemas = _get_schemas(examples=eval_examples)
    eval_dataset = eval_examples.map(
        add_serialized_schema,
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    column_names = eval_dataset.column_names

    return EvalSplit(dataset=eval_dataset, examples=eval_examples, schemas=schemas)


def prepare_splits(
    dataset_dict: DatasetDict,
    data_args: DataArguments,
    add_serialized_schema: Callable[[dict], dict]
) -> DatasetSplits:
    train_split, eval_split, test_splits = None, None, None
 
    if data_args.do_train:
        train_split = _prepare_train_split(
            dataset_dict["train"],
            data_args = data_args,
            add_serialized_schema=add_serialized_schema,
        )

    if data_args.do_eval:
        eval_split = _prepare_eval_split(
            dataset_dict["validation"],
            data_args = data_args,
            add_serialized_schema=add_serialized_schema 
        )

    if data_args.do_predict:
        test_splits = {
            section: _prepare_eval_split(
                dataset_dict[section],
                data_args = data_args,
                add_serialized_schema=add_serialized_schema 
            )
            for section in ["validation"] #data_args.test_sections
        }
        test_split_schemas = {}
        for split in test_splits.values():
            test_split_schemas.update(split.schemas)
     
    schemas = {
        **(train_split.schemas if train_split is not None else {}),
        **(eval_split.schemas if eval_split is not None else {}),
        **(test_split_schemas if test_splits is not None else {}),
    }

    return DatasetSplits(
        train_split=train_split, 
        eval_split=eval_split, 
        test_splits=test_splits, 
        schemas=schemas
    )


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))

 
    
def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    schema_serialization_type: str = "peteshaw",
    schema_serialization_randomized: bool = False,
    schema_serialization_with_db_id: bool = True,
    schema_serialization_with_db_content: bool = False,
    normalize_query: bool = True,
) -> str:
    
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    elif schema_serialization_type == "custom":
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} [ {values} ]"
        column_str_without_values = "{column}"
        value_sep = " ; "
    else:
        raise NotImplementedError

    def get_column_str(table_name: str, column_name: str) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                return column_str_with_values.format(column=column_name_str, values=value_sep.join(matches))
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(table_name=table_name, column_name=y[1]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema
