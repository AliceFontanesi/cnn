"""Visualisation helpers for semantic segmentation outputs."""

from __future__ import annotations

import numpy as np
import torch

# Default ERC terrain palette (BGR → RGB)
DEFAULT_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),         # 0 background
    (194, 178, 128),   # 1 traversable_soil
    (128, 128, 128),   # 2 bedrock
    (255, 165, 0),     # 3 small_rock
    (255, 0, 0),       # 4 large_boulder
    (255, 255, 0),     # 5 slope
    (64, 0, 128),      # 6 shadow
]


def colorize_mask(
    mask: np.ndarray | torch.Tensor,
    palette: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Convert a class-index mask (H, W) → RGB image (H, W, 3)."""
    pal = palette or DEFAULT_PALETTE
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.int64)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(pal):
        rgb[mask == cls_id] = color
    return rgb


def overlay_segmentation(
    image: np.ndarray,
    mask: np.ndarray | torch.Tensor,
    alpha: float = 0.5,
    palette: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Overlay a coloured segmentation mask on an RGB image."""
    colored = colorize_mask(mask, palette)
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    blended = (alpha * colored.astype(np.float32) +
               (1 - alpha) * image.astype(np.float32))
    return blended.clip(0, 255).astype(np.uint8)
