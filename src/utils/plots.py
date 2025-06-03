from typing import List

import plotly.graph_objects as go
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torch import Tensor


def get_colormap_and_names(dataset_name: str):
    if dataset_name == "IndianPines":
        return ListedColormap([
            "#000000", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
            "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7",
            "#ffffb3", "#bebada"
        ]), [
            "Background", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
            "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
            "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
            "Wheat", "Woods", "Buildings-Grass-Trees-Drives"
        ]
    elif dataset_name == "PaviaUniversity":
        return ListedColormap([
            "#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
            "#ffff33", "#a65628", "#f781bf"
        ]), [
            "Background", "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
            "Bare Soil", "Bitumen", "Self-Blocking Bricks"
        ]
    elif dataset_name == "KSC":
        return ListedColormap([
            "#000000", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
            "#e6ab02", "#a6761d", "#666666", "#a6cee3", "#1f78b4", "#b2df8a",
            "#33a02c", "#fb9a99"
        ]), [
            "Background", "Scrub", "Willow swamp", "CP hammock", "Slash pine", "Oak/broadleaf",
            "Hardwood swamp", "Graminoid marsh", "Spartina marsh", "Cattail marsh", "Salt marsh",
            "Mud flats", "Water", "Developed"
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def plot_image(image: Tensor):
    image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.show()


def plot_losses(train_losses: List[float], eval_losses: List[float]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(y=eval_losses, mode="lines", name="Validation Loss"))
    fig.update_layout(
        title="Losses",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        legend=dict(x=0, y=1, traceorder="normal"),
    )
    fig.show()


def plot_maps(rgb, gt, pred, dataset_name):
    cmap, class_names = get_colormap_and_names(dataset_name)
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    axs[0].imshow(rgb)
    axs[0].set_title("False Color Image")
    axs[0].axis("off")

    axs[1].imshow(gt, cmap=cmap)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    im = axs[2].imshow(pred, cmap=cmap)
    axs[2].set_title("Predicted Map")
    axs[2].axis("off")

    unique_classes = np.unique(pred)
    legend_elements = [
        Patch(facecolor=cmap(i), edgecolor="black", label=class_names[i])
        for i in unique_classes if i < len(class_names)
    ]
    axs[2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

    plt.tight_layout()
    plt.show()


def reconstruct_map(preds, labels, h, w, patch_size=15):
    out_map = np.zeros((h, w), dtype=np.uint8)
    margin = patch_size // 2
    idx = 0
    for i in range(margin, h + margin):
        for j in range(margin, w + margin):
            if labels[i - margin, j - margin] != 0:
                out_map[i - margin, j - margin] = preds[idx] + 1
                idx += 1
    return out_map
