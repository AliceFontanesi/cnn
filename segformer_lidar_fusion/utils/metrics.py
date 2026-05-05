"""Semantic segmentation evaluation metrics."""

from __future__ import annotations

import torch


def compute_per_class_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> list[float]:
    """Compute per-class Intersection-over-Union.

    Parameters
    ----------
    pred : Tensor (B, H, W)  — predicted class indices.
    target : Tensor (B, H, W) — ground-truth class indices.
    num_classes : int
    ignore_index : int

    Returns
    -------
    list[float] — IoU for each class (NaN if class absent).
    """
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    ious: list[float] = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious.append(intersection / union if union > 0 else float("nan"))
    return ious


def compute_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Mean IoU over classes present in the ground truth."""
    ious = compute_per_class_iou(pred, target, num_classes, ignore_index)
    valid = [v for v in ious if v == v]  # filter NaN
    return sum(valid) / len(valid) if valid else 0.0
