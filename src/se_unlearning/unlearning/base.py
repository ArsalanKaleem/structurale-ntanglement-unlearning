from abc import ABC, abstractmethod

class UnlearningMethod(ABC):
    @abstractmethod
    def update(self, model, tokenizer, rows, device, config, logger=None):
        raise NotImplementedError
