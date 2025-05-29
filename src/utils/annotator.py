import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modules.datasets import BaseDataset


class PseudoLabelAnnotator:
    def __init__(
        self, model: nn.Module, temperature: float = 1.0, threshold: float = 0.8
    ):
        self.model = model
        self.temperature = temperature
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold

        self.model.to(self.device)
        self.model.eval()

    def mark(self, dataset: Dataset) -> BaseDataset:
        inputs, pseudo_labels = [], []
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                outputs = self.model(x) / self.temperature
                probs = torch.softmax(outputs, dim=1)

                confidence, labels = torch.max(probs, dim=1)
                mask = confidence > self.threshold
                inputs.append(x[mask].cpu())
                pseudo_labels.append(labels[mask].cpu())

        inputs = torch.cat(inputs, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        pseudo_labels = pseudo_labels.squeeze(-1)

        return BaseDataset(inputs=inputs, labels=pseudo_labels)
