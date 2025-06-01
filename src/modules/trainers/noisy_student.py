from copy import deepcopy
from typing import Callable, Optional, Union

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import ConcatDataset, Dataset

from config.train_config import BaseTrainConfig
from modules.trainers.ssl import BaseSLTrainer
from modules.datasets.base import BaseDataset
from utils.annotator import PseudoLabelAnnotator

from utils.metrics import Metrics


class NoisyStudentTrainer(BaseSLTrainer):
    def __init__(
        self,
        teacher_model: Union[nn.Module, Callable],
        student_model: Union[nn.Module, Callable],
        loss_fn: Callable,
        labeled_dataset: BaseDataset,
        unlabeled_dataset: BaseDataset,
        teacher_optimizer: Optimizer,
        student_optimizer: Optimizer,
        teacher_lr_scheduler: LRScheduler,
        student_lr_scheduler: LRScheduler,
        teacher_config: BaseTrainConfig,
        student_config: BaseTrainConfig,
        metrics: Metrics,
        temperature: float = 1.0,
        threshold: float = 0.8,
        ema_decay: float = 0.999,
        save_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            teacher_model=teacher_model,
            student_model=student_model,
            loss_fn=loss_fn,
            labeled_dataset=labeled_dataset,
            unlabeled_dataset=unlabeled_dataset,
            teacher_optimizer=teacher_optimizer,
            student_optimizer=student_optimizer,
            teacher_lr_scheduler=teacher_lr_scheduler,
            student_lr_scheduler=student_lr_scheduler,
            teacher_config=teacher_config,
            student_config=student_config,
            metrics=metrics,
            ema_decay=ema_decay,
            save_dir=save_dir,
        )
        self.temperature = temperature
        self.threshold = threshold

    def set_student_dataset(self) -> Dataset:
        print("Pseudo Label Annotation")
        pseudo_dataset = PseudoLabelAnnotator(
            model=self.teacher_model,
            temperature=self.temperature,
            threshold=self.threshold,
        ).mark(self.unlabeled_dataset)
        labeled_dataset = deepcopy(self.labeled_dataset)

        return ConcatDataset([pseudo_dataset, labeled_dataset])
