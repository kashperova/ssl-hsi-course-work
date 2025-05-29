from typing import Any

import wandb

from utils.metrics import Metrics


class WBLogger:
    def __init__(self, train_metrics: Metrics, val_metrics: Metrics, alias: str):
        self._train_metrics = train_metrics
        self._val_metrics = val_metrics
        self.alias = alias

    def log_train_epoch(self, **kwargs):
        logs = {**kwargs}
        if self._train_metrics is not None:
            logs.update(
                {f"train {k}": v for k, v in self._train_metrics.to_dict().items()}
            )
        self.__log(logs)

    def log_val_epoch(self, **kwargs):
        logs = {**kwargs}
        if self._val_metrics is not None:
            logs.update(
                {f"valid {k}": v for k, v in self._val_metrics.to_dict().items()}
            )
        self.__log(logs)

    def __log(self, logs: dict[str, Any]):
        logs = {f"{self.alias} {k}": v for k, v in logs.items()}
        wandb.log(logs)
