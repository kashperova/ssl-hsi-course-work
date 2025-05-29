import math
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention block"""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super(ECA, self).__init__()

        super().__init__()
        k = int(abs((math.log2(channels) / gamma) + b))
        k = k if k % 2 else k + 1  # k should be odd
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: B x C x D x H x W
        # squeeze spatial + spectral dims
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, 1, C)

        # 1D conv along channel dim
        y = self.conv(y)
        y = y.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y
