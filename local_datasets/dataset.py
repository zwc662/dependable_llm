
import json
from typing import List, Generator, Any, Dict, Tuple
from third_party.spider.preprocess.get_tables import dump_db_json_schema
import datasets
from local_datasets import utils
from abc import ABC, abstractmethod
logger = datasets.logging.get_logger(__name__)

_URL, _DESCRIPTION, _HOMEPAGE, _LICENSE, _CITATION = None, None, None, None, None
del _URL, _DESCRIPTION, _HOMEPAGE, _LICENSE, _CITATION

class DataSet(datasets.GeneratorBasedBuilder, ABC):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = None
    del BUILDER_CONFIGS
    
    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        self.create_table_cache = dict()
        print(self.BUILDER_CONFIGS)
    
    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
                "create_table": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    @abstractmethod
    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [
                        downloaded_filepath + "",
                        downloaded_filepath + "",
                    ]
                    if self.include_train_others
                    else [downloaded_filepath + ""],
                    "db_path": downloaded_filepath + "",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [downloaded_filepath + ""],
                    "db_path": downloaded_filepath + "",
                },
            ),
        ]
    
    @abstractmethod
    def _generate_examples(
        self, data_filepaths: List[str], db_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                spider = json.load(f)
                for idx, sample in enumerate(spider):
                    db_id = sample["db_id"]
                    if db_id not in self.schema_cache:
                        self.schema_cache[db_id] = dump_db_json_schema(
                            db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                        )
                        self.create_table_cache[db_id] = utils.get_create_table(db_path + "/" +  db_id + "/schema.sql", db_id)
                    schema = self.schema_cache[db_id]
                    create_table = self.create_table_cache[db_id]
                    question = sample["question"]
                    question = question.replace('``', "\"").replace("''", "\"")

                    yield idx, {
                        "query": sample["query"],
                        "question": question,
                        "db_id": db_id,
                        "db_path": db_path,
                        "db_table_names": schema["table_names_original"],
                        "db_column_names": [
                            {"table_id": table_id, "column_name": column_name}
                            for table_id, column_name in schema["column_names_original"]
                        ],
                        "db_column_types": schema["column_types"],
                        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                        "db_foreign_keys": [
                            {"column_id": column_id, "other_column_id": other_column_id}
                            for column_id, other_column_id in schema["foreign_keys"]
                        ],
                        "create_table": create_table
                    }
