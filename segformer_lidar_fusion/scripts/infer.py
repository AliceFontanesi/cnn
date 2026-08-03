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
from PIL import Image, ImageDraw
import yaml

# Single-image inference script for the SegFormer-LiDAR model.
# It loads the checkpoint, applies normalization, forwards the image and lidar
# inputs through the model, and saves a visual overlay with a legend.

from segformer_lidar_fusion.models import SegFormerLiDAR
from segformer_lidar_fusion.utils.visualization import overlay_segmentation

DEFAULT_3_CLASS_NAMES = [
    "background",
    "traversable_soil",
    "bedrock",
]

DEFAULT_3_CLASS_PALETTE = [
    (0, 0, 255),   # 0: blue (background)
    (0, 255, 0),   # 1: green (traversable_soil)
    (255, 0, 0),   # 2: red (bedrock)
]

DEFAULT_3_CLASS_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 1,
    6: 0,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def infer(
    checkpoint: str,
    image_path: str,
    lidar_path: str,
    output_path: str,
    image_size: int = 512,
    num_classes: int | None = None,
    config_path: str | None = None,
    class_names: list[str] | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve config-driven normalization and image size into a consistent tuple
    if config_path is not None:
        cfg = load_config(config_path)
        cfg_image_size = cfg["data"].get("image_size")
        if isinstance(cfg_image_size, (list, tuple)):
            size_tuple = tuple(cfg_image_size)
        elif isinstance(cfg_image_size, int):
            size_tuple = (cfg_image_size, cfg_image_size)
        else:
            size_tuple = (image_size, image_size)
        mean = cfg["data"].get("mean")
        std = cfg["data"].get("std")
        lidar_mean = cfg["data"].get("lidar_mean")
        lidar_std = cfg["data"].get("lidar_std")
        num_classes = num_classes if num_classes is not None else cfg["model"].get("num_classes", 7)
        if class_names is None:
            class_names = [c["name"] for c in cfg.get("classes", []) if "name" in c]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        lidar_mean = [0.0, 0.0]
        lidar_std = [1.0, 1.0]
        size_tuple = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        num_classes = num_classes if num_classes is not None else 7

    if class_names is None:
        # Default class names based on the number of classes used.
        if num_classes == 3:
            class_names = DEFAULT_3_CLASS_NAMES
        else:
            class_names = [
                "background",
                "traversable_soil",
                "bedrock",
                "small_rock",
                "large_boulder",
                "slope",
                "shadow",
            ]

    palette = None
    if num_classes == 3:
        palette = DEFAULT_3_CLASS_PALETTE

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = state.get("model", state)
    checkpoint_num_classes = int(state_dict["decode_head.classifier.weight"].shape[0])

    model_num_classes = checkpoint_num_classes if num_classes == 3 else num_classes
    if num_classes == 3 and checkpoint_num_classes == 3:
        model_num_classes = 3
    elif num_classes == 3 and checkpoint_num_classes != 3:
        model_num_classes = checkpoint_num_classes

    # Build the model with the number of classes expected by the checkpoint.
    model = SegFormerLiDAR(num_classes=model_num_classes).to(device)
    model.load_state_dict(state_dict)
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
        img_t, size=size_tuple, mode="bilinear", align_corners=False,
    )
    lid_t = torch.nn.functional.interpolate(
        lid_t, size=size_tuple, mode="bilinear", align_corners=False,
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

    # Remap model outputs to the 3-class taxonomy if requested.
    if num_classes == 3:
        pred = remap_to_3_classes(pred)

    # Resize prediction back to the configured size used for the model
    vis_img = np.array(
        Image.open(image_path).convert("RGB").resize(
            size_tuple, Image.BILINEAR,
        )
    )
    result = overlay_segmentation(vis_img, pred, alpha=0.5, palette=palette)

    legend = build_legend(class_names, pred, size_tuple)
    result_with_legend = Image.fromarray(result)
    result_with_legend.paste(legend, (0, 0), legend)

    result_with_legend.save(output_path)
    print(f"Result saved → {output_path}")


def remap_to_3_classes(pred: np.ndarray) -> np.ndarray:
    remapped = np.full_like(pred, 0)
    for old, new in DEFAULT_3_CLASS_MAPPING.items():
        remapped[pred == old] = new
    return remapped


def build_legend(class_names: list[str], pred: np.ndarray, size: tuple[int, int]) -> Image.Image:
    legend_width = 260
    legend_height = min(22 * len(class_names) + 10, size[1])
    legend = Image.new("RGBA", (legend_width, legend_height), (0, 0, 0, 128))
    draw = ImageDraw.Draw(legend)
    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size

    y = 5
    for cls_id, count in zip(unique, counts):
        if cls_id >= len(class_names):
            continue
        pct = count / total * 100
        label = f"{cls_id}: {class_names[cls_id]} ({pct:.1f}%)"
        draw.text((10, y), label, fill=(255, 255, 255, 255))
        y += 22
    return legend


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
