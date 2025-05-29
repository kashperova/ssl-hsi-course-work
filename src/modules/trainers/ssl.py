import os
from copy import deepcopy
from typing import Callable, Union, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

from config.train_config import BaseTrainConfig
from modules.datasets.base import BaseDataset
from modules.trainers.supervised import BaseSupervisedTrainer
from utils.ema import EMA
from utils.metrics import Metrics


class BaseSLTrainer:
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
        ema_decay: float = 0.999,
        save_dir: Optional[str] = None,
        save_teacher_name: Optional[str] = "teacher",
        save_student_name: Optional[str] = "student",
    ) -> None:
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_optimizer = teacher_optimizer
        self.student_optimizer = student_optimizer
        self.teacher_lr_scheduler = teacher_lr_scheduler
        self.student_lr_scheduler = student_lr_scheduler
        self.loss_fn = loss_fn
        self.labeled_dataset = self.to_base_dataset(labeled_dataset)
        self.unlabeled_dataset = unlabeled_dataset
        self.teacher_config = teacher_config
        self.student_config = student_config
        self._metrics = metrics
        self.save_dir = os.getcwd() if save_dir is None else save_dir
        self.save_teacher_name = save_teacher_name
        self.save_student_name = save_student_name
        self.teacher_trained = False
        self.ema_updater = EMA(self.teacher_model, decay=ema_decay)

        self.teacher_trainer = BaseSupervisedTrainer(
            model=self.teacher_model,
            loss_fn=self.loss_fn,
            optimizer=self.teacher_optimizer,
            lr_scheduler=self.teacher_lr_scheduler,
            dataset=self.labeled_dataset,
            config=self.teacher_config,
            metrics=deepcopy(self._metrics),
            save_dir=self.save_dir,
            save_name=self.save_teacher_name,
        )
        self.student_trainer = None

    @staticmethod
    def to_base_dataset(dataset: Dataset) -> BaseDataset:
        if isinstance(dataset, BaseDataset):
            return dataset

        inputs, labels = zip(*[dataset[i] for i in range(len(dataset))])
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)

        return BaseDataset(inputs, labels)

    def set_student_dataset(self) -> Dataset:
        raise NotImplementedError

    def train_teacher(self, verbose: Optional[bool] = True):
        self.teacher_trainer.train(verbose=verbose)
        self.teacher_trained = True

    def train_student(self, verbose: Optional[bool] = True):
        if not self.teacher_trained:
            raise ValueError("Teacher should be trained before student training.")

        self.student_dataset = self.set_student_dataset()
        self.student_trainer = BaseSupervisedTrainer(
            model=self.student_model,
            loss_fn=self.loss_fn,
            optimizer=self.student_optimizer,
            lr_scheduler=self.student_lr_scheduler,
            dataset=self.student_dataset,
            config=self.student_config,
            metrics=deepcopy(self._metrics),
            save_dir=self.save_dir,
            save_name=self.save_student_name,
        )
        self.student_trainer.train(verbose=verbose)

        self.ema_updater.apply_to(self.teacher_model)

    def save_teacher(self):
        self.teacher_trainer.save()

    def save_student(self):
        self.student_trainer.save()

    def plot_teacher_losses(self):
        self.teacher_trainer.plot_losses()

    def plot_student_losses(self):
        self.student_trainer.plot_losses()
