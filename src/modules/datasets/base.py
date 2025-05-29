from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, inputs: Tensor, labels: Tensor):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
