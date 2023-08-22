from abc import ABCMeta, abstractmethod
from typing import ClassVar, List, Set

class BaseTrainer(metaclass=ABCMeta):
    __registry__: ClassVar[Set[str]] = []

    @classmethod
    def register(cls, name):
        if name in cls.__registry__:    
            raise NameError(f"Class {name} already defined")
        cls.__registry__.append(name)

    @abstractmethod
    def train(self, data):
        pass

    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}