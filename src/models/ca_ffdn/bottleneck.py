import torch.nn as nn


class SCBottleneck(nn.Module):
    """Separable Convolution Bottleneck block (SC-bottleneck)"""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        inter_channels = 4 * growth_rate
        # 1x1x1 conv to expand
        self.reduce = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(),
        )

        # depthwise spatial: 3x3x1
        self.depthwise = nn.Sequential(
            nn.Conv3d(
                inter_channels, inter_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)
            ),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(),
        )
        # pointwise spectral: 1x1x3
        self.pointwise = nn.Sequential(
            nn.Conv3d(
                inter_channels, growth_rate, kernel_size=(1, 1, 3), padding=(0, 0, 1)
            ),
            nn.BatchNorm3d(growth_rate),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.reduce(x)
        out = self.depthwise(out)
        out = self.pointwise(out)
        return out
