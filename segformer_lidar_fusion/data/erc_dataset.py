"""ERC Martian terrain dataset.

Expected directory layout::

    root/
    ├── images/          # RGB camera frames  (*.png | *.jpg)
    ├── lidar/           # 2-channel height+intensity maps (*.npy)
    └── masks/           # Semantic label masks (*.png), uint8 class IDs

File stems must match across the three directories so that the loader
can pair them automatically (e.g. ``frame_0001.png``, ``frame_0001.npy``,
``frame_0001.png``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:  # lightweight envs without Pillow
    Image = None  # type: ignore[assignment, misc]

try:
    import torchvision.transforms.functional as TF
except ImportError:
    TF = None  # type: ignore[assignment, misc]


class ERCDataset(Dataset):
    """Paired camera + LiDAR + mask dataset for ERC terrain segmentation.

    Parameters
    ----------
    root : str | Path
        Root directory containing ``images/``, ``lidar/``, ``masks/``.
    image_size : tuple[int, int]
        (H, W) to resize all inputs.
    mean, std : list[float]
        Per-channel normalisation for RGB images.
    lidar_mean, lidar_std : list[float]
        Per-channel normalisation for LiDAR maps.
    augment : bool
        Enable training-time augmentations (flip, jitter).
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (512, 512),
        mean: list[float] | None = None,
        std: list[float] | None = None,
        lidar_mean: list[float] | None = None,
        lidar_std: list[float] | None = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.lidar_mean = lidar_mean or [0.0, 0.0]
        self.lidar_std = lidar_std or [1.0, 1.0]
        self.augment = augment

        self.samples = self._discover_samples()

    # ------------------------------------------------------------------
    def _discover_samples(self) -> list[dict[str, Path]]:
        img_dir = self.root / "images"
        lidar_dir = self.root / "lidar"
        mask_dir = self.root / "masks"

        if not img_dir.exists():
            return []

        stems = sorted(
            p.stem for p in img_dir.iterdir()
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        samples: list[dict[str, Path]] = []
        for stem in stems:
            img_path = next(img_dir.glob(f"{stem}.*"), None)
            lidar_path = lidar_dir / f"{stem}.npy"
            mask_path = next(mask_dir.glob(f"{stem}.*"), None)
            if img_path is not None and lidar_path.exists() and mask_path is not None:
                samples.append({
                    "image": img_path,
                    "lidar": lidar_path,
                    "mask": mask_path,
                })
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # --- Load ---
        image = self._load_image(sample["image"])
        lidar = self._load_lidar(sample["lidar"])
        mask = self._load_mask(sample["mask"])

        # --- Resize ---
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=self.image_size, mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        lidar = torch.nn.functional.interpolate(
            lidar.unsqueeze(0), size=self.image_size, mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(), size=self.image_size,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        # --- Augment ---
        if self.augment:
            image, lidar, mask = self._augment(image, lidar, mask)

        # --- Normalise ---
        for c in range(3):
            image[c] = (image[c] - self.mean[c]) / self.std[c]
        for c in range(lidar.shape[0]):
            lidar[c] = (lidar[c] - self.lidar_mean[c]) / self.lidar_std[c]

        return {"image": image, "lidar": lidar, "mask": mask}

    # ------------------------------------------------------------------
    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        if Image is None:
            raise ImportError("Pillow is required for image loading")
        img = Image.open(path).convert("RGB")
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def _load_lidar(path: Path) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]  # (1, H, W)
        return torch.from_numpy(arr)

    @staticmethod
    def _load_mask(path: Path) -> torch.Tensor:
        if Image is None:
            raise ImportError("Pillow is required for mask loading")
        mask = Image.open(path)
        return torch.from_numpy(np.array(mask)).long()

    @staticmethod
    def _augment(
        image: torch.Tensor,
        lidar: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > 0.5:
            image = image.flip(-1)
            lidar = lidar.flip(-1)
            mask = mask.flip(-1)
        if torch.rand(1).item() > 0.5:
            image = image.flip(-2)
            lidar = lidar.flip(-2)
            mask = mask.flip(-2)
        return image, lidar, mask
