"""Export the SegFormer-LiDAR model to ONNX.

Usage::

    python -m segformer_lidar_fusion.deploy.export_onnx \
        --checkpoint checkpoints/best.pth \
        --output deploy/model.onnx \
        --image-size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from segformer_lidar_fusion.models import SegFormerLiDAR


def export(
    checkpoint: str | None,
    output: str,
    image_size: int = 512,
    num_classes: int = 7,
    opset: int = 17,
) -> None:
    model = SegFormerLiDAR(num_classes=num_classes)

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))

    model.eval()

    dummy_image = torch.randn(1, 3, image_size, image_size)
    dummy_lidar = torch.randn(1, 2, image_size, image_size)

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_image, dummy_lidar),
        output,
        opset_version=opset,
        input_names=["image", "lidar"],
        output_names=["logits"],
        do_constant_folding=True,
    )
    print(f"ONNX model exported → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="deploy/model.onnx")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export(args.checkpoint, args.output, args.image_size, args.num_classes, args.opset)


if __name__ == "__main__":
    main()
