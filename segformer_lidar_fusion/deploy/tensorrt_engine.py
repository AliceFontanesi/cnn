"""Build a TensorRT engine from ONNX with optional INT8 calibration.

Designed for NVIDIA Jetson Orin Nano (JetPack 6.0, TensorRT 8.6+).

Usage::

    python -m segformer_lidar_fusion.deploy.tensorrt_engine \
        --onnx deploy/model.onnx \
        --output deploy/model.engine \
        --precision int8 \
        --calib-dir data/calibration
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    import tensorrt as trt
except ImportError:
    trt = None  # type: ignore[assignment]

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except ImportError:
    cuda = None  # type: ignore[assignment]


TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if trt else None  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# INT8 calibrator
# ---------------------------------------------------------------------------

class _INT8Calibrator:
    """Entropy-based INT8 calibrator for TensorRT.

    Reads pre-processed NumPy arrays (image + lidar pairs) from a
    calibration directory.
    """

    def __init__(
        self,
        calib_dir: str,
        batch_size: int = 1,
        cache_file: str = "deploy/calibration.cache",
    ) -> None:
        if trt is None:
            raise ImportError("tensorrt is required for INT8 calibration")

        self.cache_file = cache_file
        self.batch_size = batch_size

        calib_path = Path(calib_dir)
        self.image_files = sorted(calib_path.glob("image_*.npy"))
        self.lidar_files = sorted(calib_path.glob("lidar_*.npy"))
        self.index = 0
        self.max_batches = min(len(self.image_files), len(self.lidar_files))

        # Allocate device buffers
        sample_img = np.load(self.image_files[0])
        sample_lid = np.load(self.lidar_files[0])
        self.d_image = cuda.mem_alloc(sample_img.nbytes)  # type: ignore[union-attr]
        self.d_lidar = cuda.mem_alloc(sample_lid.nbytes)  # type: ignore[union-attr]

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.index >= self.max_batches:
            return None
        img = np.load(self.image_files[self.index]).astype(np.float32)
        lid = np.load(self.lidar_files[self.index]).astype(np.float32)
        cuda.memcpy_htod(self.d_image, img)  # type: ignore[union-attr]
        cuda.memcpy_htod(self.d_lidar, lid)  # type: ignore[union-attr]
        self.index += 1
        return [int(self.d_image), int(self.d_lidar)]

    def read_calibration_cache(self) -> bytes | None:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------

def build_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "int8",
    workspace_mb: int = 1024,
    calib_dir: str | None = None,
    cache_file: str = "deploy/calibration.cache",
) -> None:
    """Build a TensorRT engine from an ONNX model.

    Parameters
    ----------
    onnx_path : str
        Path to the input ONNX file.
    output_path : str
        Where to write the serialised engine.
    precision : str
        ``"fp32"``, ``"fp16"``, or ``"int8"``.
    workspace_mb : int
        Maximum GPU workspace in MiB.
    calib_dir : str | None
        Directory with calibration data (required for INT8).
    cache_file : str
        Path to read/write the INT8 calibration cache.
    """
    if trt is None:
        raise ImportError(
            "tensorrt is required — install on Jetson via JetPack SDK"
        )

    builder = trt.Builder(TRT_LOGGER)  # type: ignore[arg-type]
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # type: ignore[union-attr]
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)  # type: ignore[arg-type]

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)  # type: ignore[union-attr]
    )

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)  # type: ignore[union-attr]
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)  # type: ignore[union-attr]
        config.set_flag(trt.BuilderFlag.FP16)  # type: ignore[union-attr]  # fall back
        if calib_dir is None:
            raise ValueError("--calib-dir required for INT8 precision")
        calibrator = _INT8Calibrator(calib_dir, cache_file=cache_file)
        config.int8_calibrator = calibrator  # type: ignore[assignment]

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    print(f"TensorRT engine ({precision}) → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TensorRT engine")
    parser.add_argument("--onnx", required=True, help="Input ONNX model")
    parser.add_argument("--output", default="deploy/model.engine")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="int8")
    parser.add_argument("--workspace-mb", type=int, default=1024)
    parser.add_argument("--calib-dir", default=None, help="Calibration data dir")
    parser.add_argument("--cache-file", default="deploy/calibration.cache")
    args = parser.parse_args()

    build_engine(
        args.onnx, args.output, args.precision,
        args.workspace_mb, args.calib_dir, args.cache_file,
    )


if __name__ == "__main__":
    main()
