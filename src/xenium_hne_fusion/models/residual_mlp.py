"""Residual MLP for flat feature-vector inputs (e.g. gene expression).

No vision-model library (timm's ResMLP/MLP-Mixer included) operates on flat
vectors — those assume a patch grid with a conv patch-embed. This is the
tabular-data counterpart: pre-norm residual blocks over a dense vector,
following the "ResNet" architecture from Gorishniy et al. 2021
(https://arxiv.org/abs/2106.11959).
"""

from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class ResidualMLP(nn.Module):
    """Sized by default (depth=12, hidden_dim=2304) to approximately match
    vit_small_patch16_224's capacity (embed_dim=384, ~22M params)."""

    def __init__(self, input_dim: int, output_dim: int = 384, hidden_dim: int = 2304, depth: int = 12, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Linear(input_dim, output_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(output_dim, hidden_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.norm(x)
