"""Diagnosis helper: run model on validation set and save worst-k samples.

Saves side-by-side images: [original | pred_overlay | gt_overlay] with mIoU label.

Usage:
    python -m segformer_lidar_fusion.scripts.diagnose --config data/ai4mars_train_config.yaml --checkpoint checkpoints/ai4mars_ft/best.pth --output-dir data/ai4mars_hf_dummy/diagnosis --topk 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw

# Diagnostic utility to run the model on the validation set and save the worst-
# performing examples as side-by-side comparison images.

from segformer_lidar_fusion.models import SegFormerLiDAR
from segformer_lidar_fusion.data import ERCDataset
from segformer_lidar_fusion.utils.visualization import overlay_segmentation, colorize_mask
from segformer_lidar_fusion.utils.metrics import compute_miou


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_diagnosis(config_path: str, checkpoint: str, output_dir: str, topk: int = 5) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = cfg["model"]["num_classes"]
    data_cfg = cfg["data"]
    image_size = tuple(data_cfg["image_size"])

    # Palette: derive from config classes if present, otherwise fallback
    if "classes" in cfg:
        palette = [tuple(c.get("color", [0, 0, 0])) for c in cfg.get("classes", [])]
    else:
        palette = None

    model = SegFormerLiDAR(num_classes=num_classes).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state))
    model.eval()

    # Load the validation dataset from the config path.
    dataset = ERCDataset(
        root=data_cfg["val_dir"],
        image_size=image_size,
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
        lidar_mean=data_cfg.get("lidar_mean"),
        lidar_std=data_cfg.get("lidar_std"),
        augment=False,
    )

    results: list[tuple[float, int, dict]] = []

    # Evaluate each sample in the validation set and record its mIoU.
    with torch.no_grad():
        for i in range(len(dataset)):
            ex = dataset[i]
            img = ex["image"].unsqueeze(0).to(device)
            lidar = ex["lidar"].unsqueeze(0).to(device)
            mask = ex["mask"].unsqueeze(0)

            logits = model(img, lidar)
            pred = logits.argmax(1).cpu()

            miou = compute_miou(pred, mask, num_classes, cfg.get("training", {}).get("ignore_index", 255))

            # raw image path is stored in dataset.samples
            sample_path = dataset.samples[i]["image"]
            raw_img = Image.open(sample_path).convert("RGB")

            results.append((miou, i, {"image": raw_img, "pred": pred.squeeze(0).numpy(), "mask": mask.squeeze(0).numpy()}))

    # sort ascending (worst first)
    results.sort(key=lambda x: x[0])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for rank, (miou, idx, data) in enumerate(results[:topk], start=1):
        raw = data["image"]
        pred_mask = data["pred"]
        gt_mask = data["mask"]

        # Ensure raw image as uint8 array
        if isinstance(raw, Image.Image):
            raw_arr = np.array(raw.convert("RGB"))
        else:
            raw_arr = np.asarray(raw, dtype=np.uint8)

        # Resize for visualization
        pil_raw = Image.fromarray(raw_arr).resize(image_size, Image.BILINEAR)

        pred_overlay = overlay_segmentation(np.array(pil_raw), pred_mask, alpha=0.6, palette=palette)
        gt_overlay = overlay_segmentation(np.array(pil_raw), gt_mask, alpha=0.6, palette=palette)

        # Compose side-by-side
        w, h = pil_raw.size
        canvas = Image.new("RGB", (w * 3, h), (0, 0, 0))
        canvas.paste(pil_raw, (0, 0))
        canvas.paste(Image.fromarray(pred_overlay), (w, 0))
        canvas.paste(Image.fromarray(gt_overlay), (w * 2, 0))

        draw = ImageDraw.Draw(canvas)
        caption = f"rank={rank} idx={idx} miou={miou:.4f}"
        draw.text((6, 6), caption, fill=(255, 255, 255), font=font)

        out_path = out_dir / f"diagnosis_rank{rank:02d}_idx{idx:06d}.png"
        canvas.save(out_path)
        print(f"Saved {out_path} — miou={miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="data/ai4mars_hf_dummy/diagnosis")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    run_diagnosis(args.config, args.checkpoint, args.output_dir, args.topk)
