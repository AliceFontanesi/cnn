"""Generate a synthetic Martian terrain dataset for training/testing.

Creates procedurally generated RGB images, LiDAR height+intensity maps,
and semantic masks with 7 terrain classes matching the ERC config.

Usage::

    python -m segformer_lidar_fusion.scripts.generate_dataset \
        --output data \
        --train-samples 200 \
        --val-samples 50 \
        --test-samples 30 \
        --image-size 512
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

# ── Terrain class palette (RGB) and base colours ──────────────────────

CLASS_NAMES = [
    "background",        # 0
    "traversable_soil",  # 1
    "bedrock",           # 2
    "small_rock",        # 3
    "large_boulder",     # 4
    "slope",             # 5
    "shadow",            # 6
]

# Approximate Martian colour ranges (R, G, B) per class
CLASS_COLOURS = {
    0: (20, 15, 12),       # dark sky / background
    1: (180, 140, 100),    # sandy soil
    2: (130, 120, 110),    # flat bedrock
    3: (110, 85, 60),      # small rocks (darker)
    4: (90, 70, 50),       # large boulders (darker still)
    5: (160, 130, 95),     # slope (similar to soil, slightly different)
    6: (40, 30, 25),       # shadow (very dark)
}


# ── Noise helpers ─────────────────────────────────────────────────────

def _perlin_like(h: int, w: int, scale: float = 64.0) -> np.ndarray:
    """Simple multi-octave value noise (approximates Perlin)."""
    result = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    freq = 1.0
    for _ in range(5):
        sh = max(2, int(h / (scale / freq)))
        sw = max(2, int(w / (scale / freq)))
        noise = np.random.randn(sh, sw).astype(np.float32)
        # Upsample with bilinear
        from PIL import Image as _Im

        noise_img = _Im.fromarray(noise, mode="F")
        noise_up = np.array(noise_img.resize((w, h), _Im.BILINEAR))
        result += amplitude * noise_up
        amplitude *= 0.5
        freq *= 2.0
    # Normalise to [0, 1]
    result -= result.min()
    mx = result.max()
    if mx > 0:
        result /= mx
    return result


def _add_texture(
    canvas: np.ndarray,
    mask: np.ndarray,
    cls_id: int,
    rng: np.random.Generator,
) -> None:
    """Paint textured colour onto canvas where mask == cls_id."""
    region = mask == cls_id
    if not region.any():
        return
    base = np.array(CLASS_COLOURS[cls_id], dtype=np.float32)
    h, w = mask.shape
    noise = _perlin_like(h, w, scale=rng.uniform(32, 96))
    for c in range(3):
        variation = (noise - 0.5) * 40  # ±20 colour jitter
        channel = canvas[:, :, c].astype(np.float32)
        channel[region] = np.clip(base[c] + variation[region], 0, 255)
        canvas[:, :, c] = channel.astype(np.uint8)


# ── Mask generation ───────────────────────────────────────────────────

def _generate_mask(
    h: int,
    w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a procedural terrain segmentation mask."""
    mask = np.ones((h, w), dtype=np.uint8)  # default: traversable_soil (1)

    # Background strip at top (sky)
    sky_height = rng.integers(h // 8, h // 4)
    horizon_noise = (_perlin_like(1, w, scale=64.0)[0] * sky_height * 0.3).astype(int)
    for x in range(w):
        boundary = max(0, sky_height + horizon_noise[x])
        mask[:boundary, x] = 0  # background

    # Bedrock patches
    n_bedrock = rng.integers(1, 4)
    for _ in range(n_bedrock):
        cx, cy = rng.integers(0, w), rng.integers(sky_height, h)
        rw, rh = rng.integers(w // 8, w // 3), rng.integers(h // 8, h // 4)
        yy, xx = np.ogrid[:h, :w]
        ellipse = ((xx - cx) / max(rw, 1)) ** 2 + ((yy - cy) / max(rh, 1)) ** 2
        mask[ellipse < 1.0] = 2

    # Slope regions
    n_slopes = rng.integers(0, 3)
    for _ in range(n_slopes):
        cx, cy = rng.integers(0, w), rng.integers(h // 3, h)
        rw, rh = rng.integers(w // 6, w // 3), rng.integers(h // 6, h // 3)
        yy, xx = np.ogrid[:h, :w]
        ellipse = ((xx - cx) / max(rw, 1)) ** 2 + ((yy - cy) / max(rh, 1)) ** 2
        region = ellipse < 1.0
        # Only overwrite soil, not bedrock
        mask[(region) & (mask == 1)] = 5

    # Large boulders
    n_boulders = rng.integers(0, 4)
    for _ in range(n_boulders):
        cx, cy = rng.integers(0, w), rng.integers(sky_height + 20, h)
        r = rng.integers(15, min(60, h // 6))
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask[dist < r] = 4

    # Small rocks
    n_small = rng.integers(5, 25)
    for _ in range(n_small):
        cx, cy = rng.integers(0, w), rng.integers(sky_height + 10, h)
        r = rng.integers(3, 12)
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask[dist < r] = 3

    # Shadows (cast from boulders, random dark patches)
    n_shadows = rng.integers(1, 5)
    for _ in range(n_shadows):
        cx, cy = rng.integers(0, w), rng.integers(sky_height, h)
        rw = rng.integers(20, w // 5)
        rh = rng.integers(10, h // 8)
        angle = rng.uniform(-0.3, 0.3)
        yy, xx = np.ogrid[:h, :w]
        dx = (xx - cx) * math.cos(angle) - (yy - cy) * math.sin(angle)
        dy = (xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
        ellipse = (dx / max(rw, 1)) ** 2 + (dy / max(rh, 1)) ** 2
        shadow_region = ellipse < 1.0
        # Shadows go on top of soil / bedrock only
        mask[(shadow_region) & ((mask == 1) | (mask == 2) | (mask == 5))] = 6

    return mask


# ── LiDAR map generation ─────────────────────────────────────────────

def _generate_lidar(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a 2-channel (height, intensity) LiDAR map from a mask."""
    h, w = mask.shape
    height_map = np.zeros((h, w), dtype=np.float32)
    intensity = np.zeros((h, w), dtype=np.float32)

    base_terrain = _perlin_like(h, w, scale=rng.uniform(80, 150))

    # Class-specific height and intensity
    height_map[mask == 0] = 0.0                        # sky
    height_map[mask == 1] = base_terrain[mask == 1] * 0.3       # soil
    height_map[mask == 2] = base_terrain[mask == 2] * 0.2 + 0.1  # bedrock
    height_map[mask == 3] = 0.5 + rng.uniform(0, 0.2)   # small rock
    height_map[mask == 4] = 0.8 + rng.uniform(0, 0.2)   # boulder
    height_map[mask == 5] = base_terrain[mask == 5] * 0.6 + 0.2  # slope
    height_map[mask == 6] = base_terrain[mask == 6] * 0.15       # shadow

    # Intensity correlates loosely with reflectivity
    intensity[mask == 0] = 0.0
    intensity[mask == 1] = 0.5 + rng.uniform(-0.1, 0.1)
    intensity[mask == 2] = 0.6 + rng.uniform(-0.1, 0.1)
    intensity[mask == 3] = 0.4 + rng.uniform(-0.1, 0.1)
    intensity[mask == 4] = 0.35 + rng.uniform(-0.1, 0.1)
    intensity[mask == 5] = 0.45 + rng.uniform(-0.1, 0.1)
    intensity[mask == 6] = 0.2 + rng.uniform(-0.05, 0.05)

    # Add sensor noise
    height_map += rng.normal(0, 0.02, (h, w)).astype(np.float32)
    intensity += rng.normal(0, 0.03, (h, w)).astype(np.float32)

    lidar = np.stack([height_map, intensity], axis=0)  # (2, H, W)
    return lidar.clip(0, None)


# ── RGB image generation ─────────────────────────────────────────────

def _generate_image(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic Martian terrain RGB image from a mask."""
    h, w = mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in range(len(CLASS_NAMES)):
        _add_texture(canvas, mask, cls_id, rng)

    # Global colour cast (Martian reddish tint)
    cast = np.array([15, -5, -10], dtype=np.float32)
    canvas = np.clip(canvas.astype(np.float32) + cast, 0, 255).astype(np.uint8)

    # Slight gaussian blur for realism
    from PIL import ImageFilter

    img = Image.fromarray(canvas)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return np.array(img)


# ── Main generator ────────────────────────────────────────────────────

def generate_split(
    output_dir: Path,
    num_samples: int,
    image_size: int,
    seed_offset: int = 0,
) -> None:
    """Generate one data split (train/val/test)."""
    img_dir = output_dir / "images"
    lid_dir = output_dir / "lidar"
    msk_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    lid_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        rng = np.random.default_rng(seed_offset + i)
        stem = f"frame_{i:05d}"

        mask = _generate_mask(image_size, image_size, rng)
        image = _generate_image(mask, rng)
        lidar = _generate_lidar(mask, rng)

        Image.fromarray(image).save(img_dir / f"{stem}.png")
        np.save(lid_dir / f"{stem}.npy", lidar)
        Image.fromarray(mask).save(msk_dir / f"{stem}.png")

        if (i + 1) % 50 == 0 or i == num_samples - 1:
            print(f"  [{output_dir.name}] {i + 1}/{num_samples}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Martian terrain dataset"
    )
    parser.add_argument("--output", default="data", help="Root output directory")
    parser.add_argument("--train-samples", type=int, default=200)
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--test-samples", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.output)
    print(f"Generating dataset in {root}/")

    print("Generating training set...")
    generate_split(root / "train", args.train_samples, args.image_size, seed_offset=0)

    print("Generating validation set...")
    generate_split(root / "val", args.val_samples, args.image_size, seed_offset=10000)

    print("Generating test set...")
    generate_split(root / "test", args.test_samples, args.image_size, seed_offset=20000)

    total = args.train_samples + args.val_samples + args.test_samples
    print(f"\nDone! Generated {total} samples ({args.image_size}x{args.image_size})")
    print(f"  train: {args.train_samples}")
    print(f"  val:   {args.val_samples}")
    print(f"  test:  {args.test_samples}")


if __name__ == "__main__":
    main()
