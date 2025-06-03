import os

import numpy as np
from torch.utils.data import Subset, Dataset

from modules.datasets import BaseDataset

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.io as sio


def subsample_dataset(dataset: BaseDataset, ratio: float) -> Subset:
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    indices = []
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        sample_size = max(1, int(ratio * len(label_indices)))
        selected = np.random.choice(label_indices, sample_size, replace=False)
        indices.extend(selected)
    return Subset(dataset, indices)


def apply_pca(hsi: np.array, num_components: int):
    reshaped = hsi.reshape(-1, hsi.shape[-1])
    pca = PCA(n_components=num_components)
    reduced = pca.fit_transform(reshaped)
    return reduced.reshape(hsi.shape[0], hsi.shape[1], num_components)


def extract_patches(hsi: np.array, labels: np.array, patch_size=15):
    margin = patch_size // 2
    padded_hsi = np.pad(
        hsi, ((margin, margin), (margin, margin), (0, 0)), mode="reflect"
    )
    data, targets = [], []
    for i in range(margin, hsi.shape[0] + margin):
        for j in range(margin, hsi.shape[1] + margin):
            label = labels[i - margin, j - margin]
            if label != 0:
                patch = padded_hsi[
                    i - margin : i + margin + 1, j - margin : j + margin + 1, :
                ]
                data.append(patch)
                targets.append(label - 1)
    return np.array(data), np.array(targets)


def load_hsi_dataset(
    dataset_name: str,
    root_dir: str,
    pca_components: int = 30,
    patch_size: int = 15,
    return_all: bool = False,
):
    dataset_files = {
        "IndianPines": {
            "image": os.path.join(root_dir, "Indian_pines_corrected.mat"),
            "labels": os.path.join(root_dir, "Indian_pines_gt.mat"),
            "image_key": "indian_pines_corrected",
            "label_key": "indian_pines_gt",
        },
        "PaviaUniversity": {
            "image": os.path.join(root_dir, "PaviaU.mat"),
            "labels": os.path.join(root_dir, "PaviaU_gt.mat"),
            "image_key": "paviaU",
            "label_key": "paviaU_gt",
        },
        "KSC": {
            "image": os.path.join(root_dir, "KSC.mat"),
            "labels": os.path.join(root_dir, "KSC_gt.mat"),
            "image_key": "KSC",
            "label_key": "KSC_gt",
        },
    }

    assert dataset_name in dataset_files, f"Unsupported dataset: {dataset_name}"
    files = dataset_files[dataset_name]
    hsi = sio.loadmat(files["image"])[files["image_key"]].astype(np.float32)
    labels = sio.loadmat(files["labels"])[files["label_key"]].astype(np.int32)

    hsi_pca = apply_pca(hsi, pca_components)
    patches, label_vec = extract_patches(hsi_pca, labels, patch_size)

    if return_all:
        return hsi, hsi_pca, labels, patches, label_vec

    return patches, label_vec


def get_stratified_subset(dataset: Dataset, subset_ratio: float) -> Dataset:
    dataset_size = len(dataset)

    labels = []
    for i in range(dataset_size):
        label = dataset[i][1]
        labels.append(label)

    indices = np.arange(dataset_size)

    subset_indices, _ = train_test_split(
        indices,
        train_size=subset_ratio,
        stratify=labels,
    )

    subset_dataset = Subset(dataset, subset_indices.tolist())

    return subset_dataset
