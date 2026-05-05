"""Lightweight convolutional encoder for LiDAR-derived inputs.

Accepts a 2-channel tensor (height map + intensity/depth) and produces
four multi-scale feature maps whose spatial dimensions and channel counts
match the camera MiT-B0 encoder outputs.

Design rationale
----------------
* Depth-wise separable convolutions keep parameter count and FLOPs low
  while still capturing local geometric structure.
* Batch-norm + GELU activation mirrors the transformer branch's activation
  profile for smoother fusion.
* Residual connections in every stage for stable gradient flow.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Depth-wise separable conv block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.needs_proj = (in_ch != out_ch) or (stride != 1)

        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

        if self.needs_proj:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x) if self.needs_proj else x
        out = self.act(self.bn1(self.dw(x)))
        out = self.bn2(self.pw(out))
        return self.act(out + identity)


class LiDAREncoder(nn.Module):
    """Four-stage conv encoder producing feature maps aligned with MiT-B0.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 2: height + intensity).
    stage_channels : list[int]
        Output channels per stage, matching the camera encoder.
    """

    DEFAULT_CHANNELS: list[int] = [32, 64, 160, 256]

    def __init__(
        self,
        in_channels: int = 2,
        stage_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        chs = stage_channels or list(self.DEFAULT_CHANNELS)
        self.channels = chs

        # Stage 1: stride-4 to match MiT patch_embed1
        self.stage1 = nn.Sequential(
            _ConvBlock(in_channels, chs[0], stride=2),
            _ConvBlock(chs[0], chs[0], stride=2),
        )
        # Stages 2-4: stride-2 each
        self.stage2 = nn.Sequential(
            _ConvBlock(chs[0], chs[1], stride=2),
            _ConvBlock(chs[1], chs[1], stride=1),
        )
        self.stage3 = nn.Sequential(
            _ConvBlock(chs[1], chs[2], stride=2),
            _ConvBlock(chs[2], chs[2], stride=1),
        )
        self.stage4 = nn.Sequential(
            _ConvBlock(chs[2], chs[3], stride=2),
            _ConvBlock(chs[3], chs[3], stride=1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return four feature maps at 1/4, 1/8, 1/16, 1/32 resolution."""
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]
