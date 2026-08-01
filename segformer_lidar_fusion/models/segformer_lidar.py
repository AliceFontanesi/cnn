"""Complete SegFormer-B0 + LiDAR Fusion model.

Combines:
1. MiT-B0 camera encoder  → multi-scale visual features
2. LiDAR conv encoder     → multi-scale geometric features
3. Cross-attention fusion  → merged multi-scale features
4. All-MLP decode head     → per-pixel class logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import MultiScaleAttentionFusion
from .lidar_encoder import LiDAREncoder
from .mit import MixTransformerB0
from .segformer_head import SegFormerHead


class SegFormerLiDAR(nn.Module):
    """SegFormer-B0 with LiDAR fusion for Martian terrain segmentation.

    Parameters
    ----------
    num_classes : int
        Number of semantic classes (default 7 for ERC terrain).
    lidar_channels : int
        Number of input channels for LiDAR branch (default 2: height + intensity).
    cam_channels : int
        Number of input channels for camera branch (default 3: RGB).
    embed_dim : int
        Unified embedding dimension in the decode head.
    dropout : float
        Dropout rate in the decode head.
    """

    def __init__(
        self,
        num_classes: int = 7,
        lidar_channels: int = 2,
        cam_channels: int = 3,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.camera_encoder = MixTransformerB0(in_channels=cam_channels)
        self.lidar_encoder = LiDAREncoder(in_channels=lidar_channels)
        self.fusion = MultiScaleAttentionFusion(
            channels=self.camera_encoder.channels,
        )
        self.decode_head = SegFormerHead(
            in_channels=self.camera_encoder.channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
    ) -> torch.Tensor:
        """Run full forward pass.

        Parameters
        ----------
        image : Tensor (B, 3, H, W)
            RGB camera input.
        lidar : Tensor (B, 2, H, W)
            LiDAR-derived height map + intensity.

        Returns
        -------
        logits : Tensor (B, num_classes, H, W)
            Per-pixel class logits upsampled to input resolution.
        """
        cam_features = self.camera_encoder(image)
        lidar_features = self.lidar_encoder(lidar)
        fused = self.fusion(cam_features, lidar_features)
        logits = self.decode_head(fused)
        logits = F.interpolate(
            logits, size=image.shape[2:],
            mode="bilinear", align_corners=False,
        )
        return logits

    def compute_loss(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
        target: torch.Tensor,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> dict[str, torch.Tensor]:
        """Forward + cross-entropy loss (convenience for training)."""
        logits = self.forward(image, lidar)
        loss = F.cross_entropy(
            logits, target,
            weight=class_weights,
            ignore_index=ignore_index,
        )
        return {"loss": loss, "logits": logits}
