import os
from copy import deepcopy
from typing import Callable, Union, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split as sklearn_split


from config.train_config import BaseTrainConfig
from modules.datasets.base import BaseDataset
from modules.logger.wb import WBLogger
from utils.ema import EMA
from utils.metrics import Metrics, CLASSIFICATION_TASKS
from utils.plots import plot_losses


class BaseSupervisedTrainer:
    def __init__(
        self,
        model: Union[nn.Module, Callable],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        config: BaseTrainConfig,
        metrics: Metrics,
        loss_fn: Optional[Callable] = None,
        save_dir: Optional[str] = None,
        save_name: Optional[str] = "model",
        grad_norm: Optional[float] = 1.0,
        alias: Optional[str] = "supervised",
        ema_hook: Optional[EMA] = None,
        dataset: BaseDataset = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn if loss_fn else self.get_loss
        self.config = config
        self.train_metrics = deepcopy(metrics)
        self.eval_metrics = deepcopy(metrics)
        self.save_dir = os.getcwd() if save_dir is None else save_dir
        self.save_name = save_name
        self.grad_norm = grad_norm
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.ema_hook = ema_hook

        self.logger = WBLogger(
            train_metrics=self.train_metrics, val_metrics=self.eval_metrics, alias=alias
        )

        self.train_dataset, self.eval_dataset = self.train_test_split(
            dataset, self.hyperparams["train_test_split"]
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.hyperparams["eval_batch_size"],
            shuffle=False,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hyperparams["train_batch_size"],
            shuffle=True,
        )

        self.train_losses = []
        self.eval_losses = []

    @property
    def hyperparams(self) -> Dict[str, Any]:
        return self.config.params

    def get_loss(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def train_step(self):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()
            running_loss += loss.item()
            self.update_metrics(self.train_metrics, outputs, labels)

            if self.ema_hook:
                self.ema_hook.update(self.model)

        return running_loss / len(self.train_loader)

    def eval_step(
        self, verbose: Optional[bool] = True, training: Optional[bool] = False
    ):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.eval_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                self.update_metrics(self.eval_metrics, outputs, labels)

        if verbose and not training:
            print(f"\nValidation metrics: {str(self.eval_metrics)}")

        return running_loss / len(self.eval_loader)

    def train(self, verbose: Optional[bool] = True):
        epochs, best_loss = self.hyperparams["epochs"], float("inf")

        for i in tqdm(range(epochs), desc="Training"):
            self.train_metrics.reset()
            train_loss = self.train_step()
            self.train_losses.append(train_loss)
            self.logger.log_train_epoch(train_loss=train_loss)

            self.eval_metrics.reset()
            eval_loss = self.eval_step(verbose=verbose, training=True)
            self.eval_losses.append(eval_loss)
            self.logger.log_val_epoch(valid_loss=eval_loss)
            self.lr_scheduler.step(eval_loss)

            if verbose:
                print(
                    f"Epoch [{i + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {eval_loss:.4f}"
                )
                print(f"\nTraining metrics: {str(self.train_metrics)}")
                print(f"\nValidation metrics: {str(self.eval_metrics)}")

            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save()

        return self.load_model()

    def eval(self):
        self.eval_metrics.reset()
        self.eval_step()

    def load_model(self) -> nn.Module:
        self.model.load_state_dict(
            torch.load(os.path.join(self.save_dir, f"{self.save_name}.pth"))
        )
        return self.model

    def save(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, f"{self.save_name}.pth"),
        )

    def plot_losses(self):
        plot_losses(self.train_losses, self.eval_losses)

    @staticmethod
    def update_metrics(metrics: Metrics, outputs: Tensor, labels: Tensor):
        if metrics.task in CLASSIFICATION_TASKS:
            _, predicted = torch.max(outputs, 1)
            metrics.update(labels, predicted)
        else:
            metrics.update(labels, outputs)

    @staticmethod
    def train_test_split(
        dataset: Dataset, split_size: float
    ) -> Tuple[Dataset, Dataset]:
        dataset_size = len(dataset)

        labels = []
        for i in range(dataset_size):
            label =  dataset[i][1]
            labels.append(label)

        indices = np.arange(dataset_size)

        train_indices, valid_indices = sklearn_split(
            indices,
            test_size=1 - split_size,
            stratify=labels,
        )
        train_dataset = Subset(dataset, train_indices.tolist())
        valid_dataset = Subset(dataset, valid_indices.tolist())

        return train_dataset, valid_dataset
