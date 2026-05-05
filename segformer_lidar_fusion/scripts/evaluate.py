"""Evaluate a trained SegFormer-LiDAR model on the validation/test set.

Usage::

    python -m segformer_lidar_fusion.scripts.evaluate \
        --config configs/erc_config.yaml \
        --checkpoint checkpoints/best.pth \
        --split val
"""

from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from segformer_lidar_fusion.data import ERCDataset
from segformer_lidar_fusion.models import SegFormerLiDAR
from segformer_lidar_fusion.utils.metrics import compute_per_class_iou


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(config_path: str, checkpoint: str, split: str = "val") -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegFormerLiDAR(
        num_classes=cfg["model"]["num_classes"],
        lidar_channels=cfg["model"]["lidar_channels"],
        cam_channels=cfg["model"]["cam_channels"],
    ).to(device)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state))
    model.eval()

    data_cfg = cfg["data"]
    root = data_cfg["val_dir"] if split == "val" else data_cfg["test_dir"]
    dataset = ERCDataset(
        root=root,
        image_size=tuple(data_cfg["image_size"]),
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        lidar_mean=data_cfg["lidar_mean"],
        lidar_std=data_cfg["lidar_std"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    num_classes = cfg["model"]["num_classes"]
    ignore_index = cfg["training"]["ignore_index"]
    class_names = [c["name"] for c in cfg["classes"]]

    all_ious: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            lidar = batch["lidar"].to(device)
            mask = batch["mask"]

            logits = model(image, lidar)
            pred = logits.argmax(1).cpu()
            ious = compute_per_class_iou(pred, mask, num_classes, ignore_index)
            all_ious.append(ious)

    if not all_ious:
        print("No samples found — check dataset path")
        return

    # Average per-class IoU
    import numpy as np

    arr = np.array(all_ious)
    mean_ious = [float(np.nanmean(arr[:, c])) for c in range(num_classes)]

    print(f"\n{'Class':<20} {'IoU':>8}")
    print("-" * 30)
    for name, iou in zip(class_names, mean_ious, strict=False):
        print(f"{name:<20} {iou:>8.4f}")
    valid = [v for v in mean_ious if v == v]
    miou = sum(valid) / len(valid) if valid else 0.0
    print("-" * 30)
    print(f"{'mIoU':<20} {miou:>8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/erc_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
