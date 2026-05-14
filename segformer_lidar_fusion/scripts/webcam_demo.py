"""Live webcam segmentation demo.

Captures frames from the webcam, runs SegFormer-B0 inference, and
displays the segmentation overlay in real time.

Since there is no LiDAR sensor connected, the LiDAR input is filled
with zeros — the model still produces a segmentation based on the
camera image alone (useful for testing and demonstrations).

Usage::

    # With a trained checkpoint
    python -m segformer_lidar_fusion.scripts.webcam_demo \
        --checkpoint checkpoints/best.pth

    # Without a checkpoint (random weights — just to test the pipeline)
    python -m segformer_lidar_fusion.scripts.webcam_demo

Controls:
    q / ESC  — quit
    s        — save current frame to screenshot.png
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from segformer_lidar_fusion.models import SegFormerLiDAR
from segformer_lidar_fusion.utils.visualization import colorize_mask

# Class names for the overlay legend
CLASS_NAMES = [
    "background",
    "traversable_soil",
    "bedrock",
    "small_rock",
    "large_boulder",
    "slope",
    "shadow",
]

CLASS_COLOURS_BGR = [
    (0, 0, 0),
    (128, 178, 194),
    (128, 128, 128),
    (0, 165, 255),
    (0, 0, 255),
    (0, 255, 255),
    (128, 0, 64),
]


def draw_legend(frame: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Draw a class legend with pixel counts on the frame."""
    h, w = frame.shape[:2]
    x0, y0 = 10, 10
    line_h = 22
    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size

    for i, (cls_id, count) in enumerate(zip(unique, counts, strict=False)):
        if cls_id >= len(CLASS_NAMES):
            continue
        pct = count / total * 100
        label = f"{CLASS_NAMES[cls_id]}: {pct:.1f}%"
        color = CLASS_COLOURS_BGR[cls_id]
        y = y0 + i * line_h

        # Background rectangle for readability
        cv2.rectangle(frame, (x0, y), (x0 + 220, y + line_h - 2), (0, 0, 0), -1)
        # Colour swatch
        cv2.rectangle(frame, (x0 + 2, y + 3), (x0 + 16, y + line_h - 5), color, -1)
        # Text
        cv2.putText(
            frame, label, (x0 + 22, y + line_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam segmentation demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional — uses random weights if omitted)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Model input size (default 256 for speed)")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default 0)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay transparency (0=camera only, 1=mask only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = SegFormerLiDAR(num_classes=args.num_classes).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided — using random weights (for pipeline testing)")
    model.eval()

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        print("Make sure your webcam is connected and not used by another app.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n--- Webcam Segmentation Demo ---")
    print("Press 'q' or ESC to quit")
    print("Press 's' to save a screenshot")
    print("--------------------------------\n")

    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    frame_count = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            break

        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (args.image_size, args.image_size))
        img_norm = (resized.astype(np.float32) / 255.0 - mean) / std
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

        # Dummy LiDAR input (zeros — no LiDAR connected)
        lidar_t = torch.zeros(1, 2, args.image_size, args.image_size, device=device)

        # Inference
        with torch.no_grad():
            logits = model(img_t, lidar_t)
        pred = logits.argmax(1).squeeze(0).cpu().numpy()

        # Colorise and overlay
        mask_rgb = colorize_mask(pred)
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        mask_resized = cv2.resize(mask_bgr, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        overlay = cv2.addWeighted(frame, 1 - args.alpha, mask_resized, args.alpha, 0)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()
        else:
            fps = frame_count / max(elapsed, 0.001)

        cv2.putText(overlay, f"FPS: {fps:.1f}", (overlay.shape[1] - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Legend
        pred_fullsize = cv2.resize(pred.astype(np.uint8),
                                   (frame.shape[1], frame.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        overlay = draw_legend(overlay, pred_fullsize)

        # Display
        cv2.imshow("SegFormer-B0 + LiDAR Fusion — Webcam Demo", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord("s"):
            cv2.imwrite("screenshot.png", overlay)
            print("Screenshot saved: screenshot.png")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo terminated.")


if __name__ == "__main__":
    main()
