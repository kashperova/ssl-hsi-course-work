from typing import Dict, Any


class BaseTrainConfig:
    epochs: int = None
    train_batch_size: int = None
    eval_batch_size: int = None
    train_test_split: float = None

    @property
    def params(self) -> Dict[str, Any]:
        attrs = set(self.__class__.__dict__.keys()) - set(self.__dict__.keys())
        return {attr: getattr(self, attr) for attr in attrs}
