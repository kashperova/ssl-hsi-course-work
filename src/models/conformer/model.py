import torch
import torch.nn as nn
from torch import Tensor

from models.ca_ffdn.bottleneck import SCBottleneck
from models.conformer.conformer import ChannelConformer


class ModModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        growth1: int = 6,
        growth2: int = 12,
        depth: int = 4,
    ):
        super().__init__()
        self.init_conv = nn.Conv3d(
            in_channels, growth1 * 2, kernel_size=3, padding=1, bias=False
        )
        self.blocks = nn.ModuleList()

        for i in range(depth):
            x_ch = growth1 * 2 + i * growth1
            block = nn.ModuleDict(
                {
                    "scb_x": SCBottleneck(x_ch, growth1),
                    "scb_y": SCBottleneck(growth2, growth2),
                    "fusion": ChannelConformer(growth1 + growth2),
                    "trans": nn.Conv3d(
                        growth1 + growth2, growth2, kernel_size=1, bias=False
                    ),
                }
            )
            self.blocks.append(block)

        self.enhance = nn.Sequential(
            nn.Conv3d(growth2, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: Tensor):
        x0 = self.init_conv(x)
        x, y = x0, x0

        for block in self.blocks:
            out_x = block["scb_x"](x)
            out_y = block["scb_y"](y)
            fused = torch.cat([out_x, out_y], dim=1)
            attn = block["fusion"](fused)
            y = block["trans"](attn)
            x = torch.cat([x, out_x], dim=1)

        feat = self.enhance(y).flatten(1)
        return self.classifier(feat)
