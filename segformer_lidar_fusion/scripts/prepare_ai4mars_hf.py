"""Prepare AI4MARS dataset from HuggingFace for SegFormer training.

This script can load the dataset from HuggingFace using `datasets.load_dataset`
and write a local `train/`, `val/`, and optional `test/` directory structure with
images, masks, and optional lidar data.

The output layout is compatible with `segformer_lidar_fusion.data.ERCDataset`:

    out_root/train/images
    out_root/train/masks
    out_root/val/images
    out_root/val/masks

Label remapping is applied to convert the original AI4MARS classes into the
training classes used by the model.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# This script exports the AI4MARS dataset from HuggingFace to a local directory
# layout suitable for training the SegFormer-LiDAR model. It handles label
# remapping, image/mask saving, optional dummy LiDAR generation, and simple
# split handling.


def parse_mapping(mapping: str) -> dict[int, int]:
    """Parse a mapping string like '0:0,1:1,2:2' into a dict."""
    result: dict[int, int] = {}
    for pair in mapping.split(","):
        if not pair:
            continue
        old, new = pair.split(":")
        result[int(old)] = int(new)
    return result


def remap_mask(mask: np.ndarray, mapping: dict[int, int], default: int = 255) -> np.ndarray:
    """Convert mask labels according to the mapping, leaving unknown as 255."""
    if mask.ndim == 3 and mask.shape[2] == 3:
        if np.all(mask[:, :, 0] == mask[:, :, 1]) and np.all(mask[:, :, 0] == mask[:, :, 2]):
            mask = mask[:, :, 0]
    mask = mask.astype(np.int64)
    out = np.full(mask.shape, default, dtype=np.int64)
    for old, new in mapping.items():
        out[mask == old] = new
    return out.astype(np.uint8)


def save_image(image, path: Path) -> None:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif hasattr(image, "to_pil"):
        image = image.to_pil()
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))
    image.convert("RGB").save(path)


def save_mask(mask, path: Path) -> None:
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    Image.fromarray(mask_arr, mode="L").save(path)


def save_lidar(lidar: np.ndarray, path: Path) -> None:
    np.save(path, lidar.astype(np.float32))


def get_column_name(columns: list[str], candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def export_split(
    split_name: str,
    split_dataset,
    output_root: Path,
    mapping: dict[int, int],
    max_examples: int | None = None,
    generate_dummy_lidar: bool = False,
) -> None:
    image_col = get_column_name(split_dataset.column_names, ["image", "img", "rgb", "photo"])
    mask_col = get_column_name(split_dataset.column_names, ["label_mask", "mask", "label", "labels", "segmentation"])
    has_mask_col = get_column_name(split_dataset.column_names, ["has_masks", "has_labels"])

    if image_col is None or mask_col is None:
        raise ValueError(
            f"Unable to detect image/mask columns in split '{split_name}'. "
            f"Available columns: {split_dataset.column_names}"
        )

    # Create output directories for images, masks and lidar maps.
    out_images = output_root / split_name / "images"
    out_masks = output_root / split_name / "masks"
    out_lidars = output_root / split_name / "lidar"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_lidars.mkdir(parents=True, exist_ok=True)

    print(f"Exporting examples for split '{split_name}'...")
    exported = 0
    skipped = 0
    for idx, example in enumerate(split_dataset):
        if has_mask_col is not None and not example[has_mask_col]:
            skipped += 1
            continue

        try:
            image = example[image_col]
            mask = example[mask_col]
            if mask is None:
                skipped += 1
                continue

            name = f"{split_name}_{exported:06d}.png"
            image_path = out_images / name
            mask_path = out_masks / name

            save_image(image, image_path)
            mask_arr = np.asarray(mask)
            remapped = remap_mask(mask_arr, mapping)
            save_mask(remapped, mask_path)

            if generate_dummy_lidar:
                lidar = np.zeros((2, remapped.shape[0], remapped.shape[1]), dtype=np.float32)
                lidar_path = out_lidars / f"{split_name}_{exported:06d}.npy"
                save_lidar(lidar, lidar_path)

            exported += 1
            if exported % 100 == 0:
                print(f"  exported {exported}")

            if max_examples is not None and exported >= max_examples:
                print(f"Reached max_examples={max_examples} for split '{split_name}'")
                break

        except (Image.UnidentifiedImageError, OSError, ValueError, TypeError) as e:
            skipped += 1
            if skipped % 50 == 0:
                print(f"  skipped {skipped} unreadable examples so far (last idx={idx}): {e}")
            continue

    if exported == 0:
        raise ValueError(f"No labeled examples were exported for split '{split_name}'.")
    print(f"Exported {exported} examples, skipped {skipped} unlabeled examples.")


def load_hf_dataset(dataset_name: str):
    """Load a HuggingFace dataset by name."""
    from datasets import load_dataset

    return load_dataset(dataset_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare AI4MARS dataset from HuggingFace")
    parser.add_argument("--dataset-name", type=str, default="hassanjbara/AI4MARS")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--mapping",
        type=str,
        default="0:0,1:0,2:1,3:2",
        help="Old_label:new_label pairs separated by commas",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used only if a train/val split is created from a single split",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.9,
        help="Fraction of data to keep for train when splitting a single dataset split",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="test_min3",
        help="Which validation/test split to use if the dataset exposes multiple test splits",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of exported examples per split for testing/partial exports",
    )
    parser.add_argument(
        "--generate-dummy-lidar",
        action="store_true",
        help="Generate dummy lidar .npy files for each exported sample.",
    )
    args = parser.parse_args(argv)
    dataset = load_hf_dataset(args.dataset_name)
    out_root = Path(args.out)
    mapping = parse_mapping(args.mapping)

    if "train" not in dataset:
        raise ValueError("HuggingFace dataset does not contain a train split")

    valid_split = None
    if "validation" in dataset:
        valid_split = "validation"
    elif args.val_split in dataset:
        valid_split = args.val_split
    else:
        other_splits = [k for k in dataset.keys() if k not in {"train"}]
        valid_split = other_splits[0] if other_splits else None

    export_split(
        "train",
        dataset["train"],
        out_root,
        mapping,
        args.max_examples,
        args.generate_dummy_lidar,
    )
    if valid_split is not None:
        export_split(
            "val",
            dataset[valid_split],
            out_root,
            mapping,
            args.max_examples,
            args.generate_dummy_lidar,
        )
    else:
        split = dataset["train"].train_test_split(test_size=1.0 - args.train_frac, seed=args.seed)
        export_split("train", split["train"], out_root, mapping, args.max_examples)
        export_split("val", split["test"], out_root, mapping, args.max_examples)

    print("Dataset preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
