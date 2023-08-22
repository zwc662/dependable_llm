from typing import Any, Dict, List, Optional, Set, Union, Literal, Callable


from dependent.core.utils.register import register
from dependent.core.trainers import BaseTrainer

import transformers
from torch.utils.data import Dataset  

@register
class MosaicTrainer(BaseTrainer):
    def __post_init__(self):
        raise "NotImplementedError"
    