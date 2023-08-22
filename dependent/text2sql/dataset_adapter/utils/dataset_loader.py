import json
from typing import Callable, Tuple, Dict
from functools import partial
import logging
import datasets.load
from datasets.dataset_dict import DatasetDict
from datasets.metric import Metric
from datasets.arrow_dataset import Dataset, concatenate_datasets
from .dataset import (
    DataArguments,
    DatasetSplits,
    TrainSplit,
    _prepare_train_split,
    prepare_splits,
)
from .spider import spider_add_serialized_schema 
from .sparc import sparc_add_serialized_schema 
from .cosql import cosql_add_serialized_schema
 

logger = logging.getLogger(__name__)

"""
    Dataset adapter provides a unified pipeline for loading sparc, spider and cosql datasets.
    The adapter takes the dataset name as input, preprocess the data from the dataset, and 
    instantiate a dataset class
"""
def load_dataset(data_args: DataArguments)-> Tuple[Metric, DatasetSplits]: 

    _spider_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths['spider'], cache_dir=data_args.cache_dir
    )
    _spider_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["spider"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )
    _spider_add_serialized_schema = lambda ex: spider_add_serialized_schema(
        ex=ex,
        data_args=data_args,
    )
    
    _sparc_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["sparc"], cache_dir=data_args.cache_dir
    )
    _sparc_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["sparc"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )
    _sparc_add_serialized_schema = lambda ex: sparc_add_serialized_schema(
        ex=ex,
        data_args=data_args,
    )
    

    _cosql_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["cosql"], cache_dir=data_args.cache_dir
    )
    _cosql_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["cosql"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )
    _cosql_add_serialized_schema = lambda ex: cosql_add_serialized_schema(
        ex=ex,
        data_args=data_args,
    )
    
    if data_args.dataset == "spider":
        metric = _spider_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_spider_dataset_dict(),
            data_args=data_args,
            add_serialized_schema=_spider_add_serialized_schema,
        )
    elif data_args.dataset == "cosql":
        metric = _cosql_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_cosql_dataset_dict(),
            data_args=data_args,
            add_serialized_schema=_cosql_add_serialized_schema,
        )
    elif data_args.dataset == "sparc":
        metric = _sparc_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_sparc_dataset_dict(),
            data_args=data_args,
            add_serialized_schema=_sparc_add_serialized_schema,
        )
    elif data_args.dataset == "cosql+spider":
        metric = _cosql_metric()
        data_args.split_dataset = "cosql"
        cosql_dataset_splits = prepare_splits(
            dataset_dict=_cosql_dataset_dict(),
            data_args=data_args,
            add_serialized_schema=_cosql_add_serialized_schema,
        )
        data_args.split_dataset = "spider"
        spider_training_split = (
            _prepare_train_split(
                dataset=_spider_dataset_dict()["train"],
                data_args=data_args,
                add_serialized_schema=_spider_add_serialized_schema
            )
            if data_args.do_train
            else None
        )
        if cosql_dataset_splits.train_split is None and spider_training_split is None:
            train_split = None
        elif cosql_dataset_splits.train_split is None:
            train_split = spider_training_split
        elif spider_training_split is None:
            train_split = cosql_dataset_splits.train_split
        else:
            dataset: Dataset = concatenate_datasets(
                dsets=[cosql_dataset_splits.train_split.dataset, spider_training_split.dataset]
            )
            train_split = TrainSplit(
                dataset=dataset,
                schemas={**spider_training_split.schemas, **cosql_dataset_splits.train_split.schemas},
            )
        schemas = {
            **cosql_dataset_splits.schemas,
            **(spider_training_split.schemas if spider_training_split is not None else {}),
        }
        dataset_splits = DatasetSplits(
            train_split=train_split,
            eval_split=cosql_dataset_splits.eval_split,
            test_splits=cosql_dataset_splits.test_splits,
            schemas=schemas,
        )
    elif data_args.dataset == "sparc+spider":
        data_args.split_dataset = "sparc"
        metric = _sparc_metric()
        sparc_dataset_splits = prepare_splits(
            dataset_dict=_sparc_dataset_dict(),
            data_args=data_args,
            add_serialized_schema=_sparc_add_serialized_schema
        )
        data_args.split_dataset = "spider"
        spider_training_split = (
            _prepare_train_split(
                dataset=_spider_dataset_dict()["train"],
                data_args=data_args,
                add_serialized_schema=_spider_add_serialized_schema,
            )
            if data_args.do_train
            else None
        )
        if sparc_dataset_splits.train_split is None and spider_training_split is None:
            train_split = None
        elif sparc_dataset_splits.train_split is None:
            train_split = spider_training_split
        elif spider_training_split is None:
            train_split = sparc_dataset_splits.train_split
        else:
            dataset: Dataset = concatenate_datasets(
                dsets=[sparc_dataset_splits.train_split.dataset, spider_training_split.dataset]
            )
            train_split = TrainSplit(
                dataset=dataset,
                schemas={**spider_training_split.schemas, **sparc_dataset_splits.train_split.schemas},
            )
        schemas = {
            **sparc_dataset_splits.schemas,
            **(spider_training_split.schemas if spider_training_split is not None else {}),
        }
        dataset_splits = DatasetSplits(
            train_split=train_split,
            eval_split=sparc_dataset_splits.eval_split,
            test_splits=sparc_dataset_splits.test_splits,
            schemas=schemas,
        )
    else:
        raise NotImplementedError()

     
    return metric, dataset_splits