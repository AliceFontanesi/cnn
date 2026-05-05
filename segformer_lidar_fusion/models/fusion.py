"""Multi-scale cross-attention fusion module.

Fuses camera (MiT-B0) and LiDAR encoder features at each of the four
hierarchical scales using a lightweight cross-attention gate:

    fused = cam + gate * lidar_proj

where *gate* is computed via sigmoid(query·key^T) and the value comes
from the LiDAR branch.  This keeps the fusion module small and
ONNX/TensorRT-friendly (no variable-length sequences).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FusionGate(nn.Module):
    """Single-scale gated cross-attention fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self, cam: torch.Tensor, lidar: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(cam)
        k = self.k_proj(lidar)
        v = self.v_proj(lidar)

        # Channel-wise attention map
        attn = torch.sum(q * k, dim=1, keepdim=True)
        attn = torch.sigmoid(attn)

        gated_v = self.gate(v) * attn
        return cam + gated_v


class MultiScaleAttentionFusion(nn.Module):
    """Fuse camera and LiDAR features at four hierarchical scales."""

    def __init__(self, channels: list[int] | None = None) -> None:
        super().__init__()
        chs = channels or [32, 64, 160, 256]
        self.fusions = nn.ModuleList([_FusionGate(c) for c in chs])

    def forward(
        self,
        cam_features: list[torch.Tensor],
        lidar_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        fused: list[torch.Tensor] = []
        for gate, cam, lidar in zip(self.fusions, cam_features, lidar_features, strict=False):
            # Handle potential spatial size mismatch
            if cam.shape[2:] != lidar.shape[2:]:
                lidar = F.interpolate(
                    lidar, size=cam.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            fused.append(gate(cam, lidar))
        return fused
