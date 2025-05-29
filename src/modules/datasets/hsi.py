import numpy as np
import torch

from torch import Tensor
from torch.utils.data import Dataset


class HyperspectralDataset(Dataset):
    def __init__(self, patches: Tensor, labels: Tensor):
        self.patches = patches
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.patches[idx]  # (H, W, C)
        x = np.transpose(x, (2, 0, 1))  # (C, H, W)
        x = np.expand_dims(x, axis=1)  # (C, 1, H, W)
        x = np.transpose(
            x, (0, 2, 3, 1)
        )  # (C, H, W, D) just to double-check shape consistency
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )
