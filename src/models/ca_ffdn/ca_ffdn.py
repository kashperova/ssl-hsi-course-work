import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.ca_ffdn.bottleneck import SCBottleneck
from models.ca_ffdn.eca_block import ECA


class CA_FFDN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.init_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)

        self.sc1 = SCBottleneck(64, 32)
        self.eca1 = ECA(32)

        self.sc2 = SCBottleneck(32, 32)
        self.eca2 = ECA(32)

        self.conv_enh1 = nn.Conv3d(32, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(128)

        self.conv_enh2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: Tensor):
        x = self.init_conv(x)

        x = self.sc1(x)
        x = self.eca1(x)

        x = self.sc2(x)
        x = self.eca2(x)

        # enhancement
        x = F.relu(self.bn1(self.conv_enh1(x)))
        x = F.relu(self.bn2(self.conv_enh2(x)))

        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)
