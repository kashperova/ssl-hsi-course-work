from typing import List

import plotly.graph_objects as go

from matplotlib import pyplot as plt
from torch import Tensor


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
