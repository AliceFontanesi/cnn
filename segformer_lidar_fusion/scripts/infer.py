"""Single-image inference with visualisation.

Usage::

    python -m segformer_lidar_fusion.scripts.infer \
        --checkpoint checkpoints/best.pth \
        --image test_image.png \
        --lidar test_lidar.npy \
        --output result.png
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
import yaml

from segformer_lidar_fusion.models import SegFormerLiDAR
from segformer_lidar_fusion.utils.visualization import overlay_segmentation


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def infer(
    checkpoint: str,
    image_path: str,
    lidar_path: str,
    output_path: str,
    image_size: int = 512,
    num_classes: int = 7,
    config_path: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_path is not None:
        cfg = load_config(config_path)
        cfg_image_size = cfg["data"]["image_size"]
        image_size = tuple(cfg_image_size) if isinstance(cfg_image_size, list) else (cfg_image_size, cfg_image_size)
        mean = cfg["data"]["mean"]
        std = cfg["data"]["std"]
        lidar_mean = cfg["data"]["lidar_mean"]
        lidar_std = cfg["data"]["lidar_std"]
        num_classes = cfg["model"]["num_classes"]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        lidar_mean = [0.0, 0.0]
        lidar_std = [1.0, 1.0]

    model = SegFormerLiDAR(num_classes=num_classes).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state))
    model.eval()

    # Load inputs
    img = np.array(Image.open(image_path).convert("RGB"))
    lid = np.load(lidar_path).astype(np.float32)
    if lid.ndim == 2:
        lid = lid[np.newaxis]

    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    lid_t = torch.from_numpy(lid).float().unsqueeze(0)

    # Resize
    img_t = torch.nn.functional.interpolate(
        img_t, size=image_size, mode="bilinear", align_corners=False,
    )
    lid_t = torch.nn.functional.interpolate(
        lid_t, size=image_size, mode="bilinear", align_corners=False,
    )

    # Normalize inputs to match training
    for c in range(3):
        img_t[:, c] = (img_t[:, c] - mean[c]) / std[c]
    for c in range(lid_t.shape[1]):
        lidar_mean_val = lidar_mean[c] if c < len(lidar_mean) else 0.0
        lidar_std_val = lidar_std[c] if c < len(lidar_std) else 1.0
        lid_t[:, c] = (lid_t[:, c] - lidar_mean_val) / lidar_std_val

    with torch.no_grad():
        logits = model(img_t.to(device), lid_t.to(device))
    pred = logits.argmax(1).squeeze(0).cpu().numpy()

    # Resize prediction back to original image size
    vis_img = np.array(
        Image.open(image_path).convert("RGB").resize(
            (image_size, image_size), Image.BILINEAR,
        )
    )
    result = overlay_segmentation(vis_img, pred, alpha=0.5)
    Image.fromarray(result).save(output_path)
    print(f"Result saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--lidar", required=True)
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config with normalization and model settings")
    args = parser.parse_args()
    infer(
        args.checkpoint, args.image, args.lidar, args.output,
        args.image_size, args.num_classes, args.config,
    )


if __name__ == "__main__":
    main()
