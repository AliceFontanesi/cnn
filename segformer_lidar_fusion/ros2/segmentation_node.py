"""ROS 2 real-time semantic segmentation node.

Subscribes to synchronised camera + LiDAR topics, runs inference via
TensorRT (or PyTorch fallback), and publishes a coloured segmentation
mask as a costmap overlay.

Launch::

    ros2 launch segformer_lidar_fusion segmentation.launch.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

try:
    import message_filters
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2

except ImportError:
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment, misc]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError:
    trt = None  # type: ignore[assignment]
    cuda = None  # type: ignore[assignment]


class SegmentationNode(Node):  # type: ignore[misc]
    """Real-time SegFormer-LiDAR inference node."""

    def __init__(self) -> None:
        super().__init__("segformer_segmentation")  # type: ignore[call-arg]

        # Parameters
        self.declare_parameter("engine_path", "deploy/model.engine")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("image_size", 512)
        self.declare_parameter("num_classes", 7)
        self.declare_parameter("camera_topic", "/rover/camera/image_raw")
        self.declare_parameter("lidar_topic", "/rover/lidar/points")
        self.declare_parameter("output_topic", "/rover/segmentation/costmap")

        self.image_size = self.get_parameter("image_size").value
        self.num_classes = self.get_parameter("num_classes").value

        engine_path = self.get_parameter("engine_path").value
        checkpoint_path = self.get_parameter("checkpoint_path").value

        self.bridge = CvBridge()  # type: ignore[operator]
        self.engine = None
        self.model = None

        # Try TensorRT first, fall back to PyTorch
        if trt is not None and Path(engine_path).exists():
            self._load_tensorrt(engine_path)
            self.get_logger().info(f"TensorRT engine loaded: {engine_path}")
        elif torch is not None and checkpoint_path:
            self._load_pytorch(checkpoint_path)
            self.get_logger().info(f"PyTorch model loaded: {checkpoint_path}")
        else:
            self.get_logger().warn("No model loaded — publish zeros")

        # Subscribers (synchronised)
        cam_topic = self.get_parameter("camera_topic").value
        lid_topic = self.get_parameter("lidar_topic").value
        out_topic = self.get_parameter("output_topic").value

        cam_sub = message_filters.Subscriber(self, Image, cam_topic)  # type: ignore[operator]
        lid_sub = message_filters.Subscriber(self, PointCloud2, lid_topic)  # type: ignore[operator]
        self.sync = message_filters.ApproximateTimeSynchronizer(  # type: ignore[operator]
            [cam_sub, lid_sub], queue_size=10, slop=0.1,
        )
        self.sync.registerCallback(self._callback)

        self.pub = self.create_publisher(Image, out_topic, 10)  # type: ignore[arg-type]
        self.get_logger().info("Segmentation node ready")

    # ------------------------------------------------------------------
    def _load_tensorrt(self, path: str) -> None:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))  # type: ignore[union-attr]
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def _load_pytorch(self, path: str) -> None:
        from segformer_lidar_fusion.models import SegFormerLiDAR  # noqa: E402

        self.model = SegFormerLiDAR(num_classes=self.num_classes)
        state = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[union-attr]
        self.model.load_state_dict(state.get("model", state))
        self.model.eval()
        if torch.cuda.is_available():  # type: ignore[union-attr]
            self.model.cuda()

    # ------------------------------------------------------------------
    def _callback(self, cam_msg: Image, lid_msg: PointCloud2) -> None:  # type: ignore[override]
        t0 = time.perf_counter()

        image = self.bridge.imgmsg_to_cv2(cam_msg, "rgb8")  # type: ignore[attr-defined]
        lidar_map = self._pointcloud_to_map(lid_msg)

        pred = self._infer(image, lidar_map)

        # Publish as mono8 mask
        mask_msg = self.bridge.cv2_to_imgmsg(pred.astype(np.uint8), "mono8")  # type: ignore[attr-defined]
        mask_msg.header = cam_msg.header  # type: ignore[attr-defined]
        self.pub.publish(mask_msg)

        dt = (time.perf_counter() - t0) * 1000
        self.get_logger().debug(f"Inference: {dt:.1f} ms")

    # ------------------------------------------------------------------
    def _infer(self, image: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        if self.model is not None and torch is not None:
            return self._infer_pytorch(image, lidar)
        # Fallback: zeros
        return np.zeros(image.shape[:2], dtype=np.uint8)

    def _infer_pytorch(self, image: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        img_t = (
            torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # type: ignore[union-attr]
        )
        lid_t = torch.from_numpy(lidar).float().unsqueeze(0)  # type: ignore[union-attr]
        if lidar.ndim == 2:
            lid_t = lid_t.unsqueeze(0)

        if torch.cuda.is_available():  # type: ignore[union-attr]
            img_t = img_t.cuda()
            lid_t = lid_t.cuda()

        with torch.no_grad():  # type: ignore[union-attr]
            logits = self.model(img_t, lid_t)
        pred = logits.argmax(1).squeeze(0).cpu().numpy()
        return pred

    @staticmethod
    def _pointcloud_to_map(msg: PointCloud2) -> np.ndarray:  # type: ignore[override]
        """Convert PointCloud2 to a 2-channel height+intensity map.

        This is a simplified placeholder — the actual implementation
        depends on the LiDAR sensor and mounting configuration.
        """
        # Placeholder: return zeros (512×512, 2 channels)
        return np.zeros((2, 512, 512), dtype=np.float32)


def main() -> None:
    if rclpy is None:
        raise ImportError("rclpy is required — install ROS 2 Humble/Iron")
    rclpy.init()
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
