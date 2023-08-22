
from utils.dataset_loader import load_dataset
from utils.dataset import DataArguments



data_args = DataArguments(
    preprocessing_num_workers = 1,  
    dataset = 'spider',
    cache_dir = './transformer_cache/',
    test_sections = 'validate',
    schema_serialization_type = 'custom'
)


data_args.do_train = True
data_args.do_predict = True
data_args.do_eval = False
 

dataset = load_dataset(data_args)
