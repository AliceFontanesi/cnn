"""Lightweight All-MLP decode head for SegFormer.

Takes multi-scale features from the encoder, unifies their channel
dimensions, upsamples to a common spatial resolution (1/4 of input),
concatenates, and applies a final MLP to predict per-pixel class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPBlock(nn.Module):
    """Linear → BN → GELU projection for a single scale."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)     # (B, N, C)
        x = self.proj(x)                      # (B, N, E)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.norm(x)
        return x


class SegFormerHead(nn.Module):
    """All-MLP decode head.

    Parameters
    ----------
    in_channels : list[int]
        Per-stage channel counts from the encoder (e.g. [32, 64, 160, 256]).
    embed_dim : int
        Unified embedding dimension (256 for B0/B1, 768 for larger).
    num_classes : int
        Number of semantic classes.
    dropout : float
        Dropout before the final classifier.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear_layers = nn.ModuleList(
            [_MLPBlock(ch, embed_dim) for ch in in_channels]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Return logits at 1/4 of the original input resolution."""
        target_size = features[0].shape[2:]
        projected: list[torch.Tensor] = []
        for mlp, feat in zip(self.linear_layers, features, strict=False):
            out = mlp(feat)
            if out.shape[2:] != target_size:
                out = F.interpolate(
                    out, size=target_size,
                    mode="bilinear", align_corners=False,
                )
            projected.append(out)

        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
