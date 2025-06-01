from torch.utils.data import Dataset
from modules.datasets.base import BaseDataset


class UnlabeledDataset(Dataset):
    def __init__(self, labeled_dataset: BaseDataset):
        self.dataset = labeled_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image
