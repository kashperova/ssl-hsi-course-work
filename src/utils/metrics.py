from enum import Enum
from typing import Dict, Optional, List

import torch
from torch import Tensor
from torchmetrics import Metric, Accuracy, F1Score, Recall, Precision
from torchmetrics.classification import CohenKappa, MulticlassConfusionMatrix


class Task(Enum):
    BINARY_CLASSIFICATION: str = "binary_classification"
    MULTICLASS_CLASSIFICATION: str = "multiclass_classification"


CLASSIFICATION_TASKS: List[Task] = [
    Task.BINARY_CLASSIFICATION,
    Task.MULTICLASS_CLASSIFICATION,
]


class Metrics:
    def __init__(
        self,
        task: Task,
        average: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        if task not in list(Task):
            raise ValueError(f"Available tasks: {list(Task)}")

        if task in CLASSIFICATION_TASKS:
            if average is None:
                raise ValueError("Select average")

            if num_classes is None and task == Task.MULTICLASS_CLASSIFICATION:
                raise ValueError("Select number of classes")

            if num_classes is None and task == Task.BINARY_CLASSIFICATION:
                num_classes = 2

            self.num_classes = num_classes

        self.task = task
        self.average = average
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = self.get_metrics()

    def get_metrics(self) -> Dict[str, Metric]:
        metrics = {}

        if self.task == Task.BINARY_CLASSIFICATION:
            metrics["accuracy"] = Accuracy(task="binary").to(self.device)
            metrics["recall"] = Recall(task="binary", average=self.average).to(
                self.device
            )
            metrics["precision"] = Precision(task="binary", average=self.average).to(
                self.device
            )
            metrics["f1"] = F1Score(task="binary", average=self.average).to(self.device)

        elif self.task == Task.MULTICLASS_CLASSIFICATION:
            metrics["accuracy"] = Accuracy(
                task="multiclass", num_classes=self.num_classes
            ).to(self.device)
            metrics["recall"] = Recall(
                task="multiclass", average=self.average, num_classes=self.num_classes
            ).to(self.device)
            metrics["precision"] = Precision(
                task="multiclass", average=self.average, num_classes=self.num_classes
            ).to(self.device)
            metrics["f1"] = F1Score(
                task="multiclass", average=self.average, num_classes=self.num_classes
            ).to(self.device)
            metrics["confmat"] = MulticlassConfusionMatrix(
                num_classes=self.num_classes
            ).to(self.device)
            metrics["kappa"] = CohenKappa(
                num_classes=self.num_classes, task="multiclass"
            ).to(self.device)

        return metrics

    def update(self, y: Tensor, y_hat: Tensor):
        for metric in self.metrics.values():
            metric.update(y_hat, y)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def to_dict(self):
        result = {}
        for k, v in self.metrics.items():
            if k == "confmat":
                confmat = v.compute()
                per_class_acc = confmat.diag() / confmat.sum(1)
                result["aa"] = per_class_acc.mean()
            else:
                result[k] = v.compute()

        result["oa"] = self.metrics["accuracy"].compute()
        return result

    def __str__(self) -> str:
        res = "\n"
        computed = self.to_dict()
        for k, v in computed.items():
            if isinstance(v, torch.Tensor):
                res += f"{k.upper()}: {round(v.item(), 4)}\n"
            else:
                res += f"{k.upper()}: {v}\n"
        return res
