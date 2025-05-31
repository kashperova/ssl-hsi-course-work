import torch.nn as nn
from torch import Tensor


class ChannelConformer(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int | None = None,
        dim_feedforward: int = 128,
        conv_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if heads is None:
            for h in [4, 3, 2, 1]:
                if channels % h == 0:
                    heads = h
                    break

        self.norm1 = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, channels),
            nn.Dropout(dropout),
        )
        self.conv_module = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=conv_kernel,
                padding=conv_kernel // 2,
                groups=channels,
            ),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        B, C, D, H, W = x.shape
        tokens = x.mean(dim=(2, 3, 4)).unsqueeze(1)  # [B, 1, C]
        attn, _ = self.mha(tokens, tokens, tokens)
        tok = tokens + attn
        tok = self.norm1(tok.squeeze(1))  # [B, C]
        conv_out = self.conv_module(tok.unsqueeze(-1)).squeeze(-1)  # [B, C]
        tok = tok + conv_out
        ffn_out = self.ffn(self.norm2(tok))  # [B, C]
        tok2 = tok + ffn_out
        weights = tok2.view(B, C, 1, 1, 1)
        return x * weights
