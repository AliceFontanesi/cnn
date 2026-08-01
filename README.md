# SegFormer-B0 + LiDAR Fusion — ERC Terrain Segmentation

Real-time semantic segmentation of Martian terrain for the **European Rover Challenge (ERC)**, optimised for deployment on the **NVIDIA Jetson Orin Nano**.

## Architecture

```
Camera (RGB) ──► MiT-B0 Encoder ──┐
                                   ├── Cross-Attention Fusion ──► All-MLP Decoder ──► Semantic Mask
LiDAR (H+I)  ──► Conv Encoder  ──┘
```

| Component | Details |
|---|---|
| **Camera encoder** | Mix Transformer B0 — 4 stages, embed dims [32, 64, 160, 256], efficient self-attention with spatial reduction |
| **LiDAR encoder** | Lightweight depth-wise separable conv encoder (4 stages, channel-aligned) |
| **Fusion** | Multi-scale gated cross-attention — camera features as query, LiDAR as key/value |
| **Decoder** | SegFormer all-MLP head — unifies scales, concatenates, classifies |
| **Target latency** | ~20 ms @ 512×512 on Orin Nano (INT8 TensorRT) |

## Terrain Classes

| ID | Class | Colour |
|---|---|---|
| 0 | Background | ⬛ Black |
| 1 | Traversable Soil | 🟫 Sand |
| 2 | Bedrock | ⬜ Grey |
| 3 | Small Rock | 🟧 Orange |
| 4 | Large Boulder | 🟥 Red |
| 5 | Slope | 🟨 Yellow |
| 6 | Shadow | 🟪 Purple |

## Setup

```bash
# Clone
git clone https://github.com/AliceFontanesi/cnn.git
cd cnn

# Install (Python ≥ 3.10)
pip install -e ".[train,deploy,dev]"
```

## Dataset Structure

```
data/train/          # (or val/ or test/)
├── images/          # RGB frames (*.png, *.jpg)
├── lidar/           # Height+intensity maps (*.npy, shape [2, H, W])
└── masks/           # Label masks (*.png, uint8 class IDs)
```

File stems must match across subdirectories (e.g. `frame_0001.png`, `frame_0001.npy`).

## Training

```bash
python -m segformer_lidar_fusion.scripts.train --config segformer_lidar_fusion/configs/erc_config.yaml
```

Key hyperparameters are in `configs/erc_config.yaml`:
- AdamW optimiser (lr=6e-5, weight_decay=0.01)
- Polynomial LR schedule with 5-epoch warmup
- Mixed precision (AMP)
- Class-weighted cross-entropy

## Evaluation

```bash
python -m segformer_lidar_fusion.scripts.evaluate \
    --config segformer_lidar_fusion/configs/erc_config.yaml \
    --checkpoint checkpoints/best.pth \
    --split val
```

## Single-Image Inference

```bash
python -m segformer_lidar_fusion.scripts.infer \
    --checkpoint checkpoints/best.pth \
    --image test.png \
    --lidar test.npy \
    --output result.png
```

## Deployment on Jetson Orin Nano

### 1. Export to ONNX

```bash
python -m segformer_lidar_fusion.deploy.export_onnx \
    --checkpoint checkpoints/best.pth \
    --output deploy/model.onnx
```

### 2. Build TensorRT Engine (INT8)

```bash
python -m segformer_lidar_fusion.deploy.tensorrt_engine \
    --onnx deploy/model.onnx \
    --output deploy/model.engine \
    --precision int8 \
    --calib-dir data/calibration
```

### 3. ROS 2 Node

```bash
ros2 launch segformer_lidar_fusion segmentation.launch.py \
    engine_path:=deploy/model.engine
```

**Topics:**
| Direction | Topic | Type |
|---|---|---|
| Subscribe | `/rover/camera/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/rover/lidar/points` | `sensor_msgs/PointCloud2` |
| Publish | `/rover/segmentation/costmap` | `sensor_msgs/Image` |

## Project Structure

```
segformer_lidar_fusion/
├── configs/erc_config.yaml          # Hyperparameters & class definitions
├── models/
│   ├── mit.py                       # MiT-B0 encoder
│   ├── lidar_encoder.py             # LiDAR conv encoder
│   ├── fusion.py                    # Cross-attention fusion
│   ├── segformer_head.py            # All-MLP decoder
│   └── segformer_lidar.py           # Complete fused model
├── data/erc_dataset.py              # Dataset loader
├── deploy/
│   ├── export_onnx.py               # ONNX export
│   └── tensorrt_engine.py           # TensorRT INT8 builder
├── ros2/
│   ├── segmentation_node.py         # ROS 2 inference node
│   └── launch/segmentation.launch.py
├── scripts/
│   ├── train.py                     # Training
│   ├── evaluate.py                  # Evaluation (mIoU)
│   └── infer.py                     # Single-image inference
└── utils/
    ├── metrics.py                   # mIoU computation
    └── visualization.py             # Overlay & colourisation
```

## Hardware Target

| Spec | Value |
|---|---|
| Platform | NVIDIA Jetson Orin Nano 8 GB |
| JetPack | 6.0 |
| CUDA | 12.2 |
| TensorRT | 8.6 |
| Precision | INT8 (with FP16 fallback) |
| Target FPS | ≥ 30 Hz @ 512×512 |

## License

MIT
