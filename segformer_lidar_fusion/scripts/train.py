"""Training script for SegFormer-B0 + LiDAR Fusion.

Usage::

    python -m segformer_lidar_fusion.scripts.train \
        --config configs/erc_config.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Training script for SegFormer-LiDAR.
# It loads configuration from YAML, builds the datasets and model, and runs
# training/validation loops with checkpointing.

from segformer_lidar_fusion.data import ERCDataset
from segformer_lidar_fusion.models import SegFormerLiDAR
from segformer_lidar_fusion.utils.metrics import compute_miou


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    total_steps = cfg["training"]["epochs"] * steps_per_epoch
    warmup_steps = cfg["training"]["scheduler"]["warmup_epochs"] * steps_per_epoch
    power = cfg["training"]["scheduler"]["power"]
    min_lr = cfg["training"]["scheduler"]["min_lr"]
    base_lr = cfg["training"]["optimizer"]["lr"]

    def lr_lambda(step: int) -> float:
        # Linear warmup followed by polynomial decay.
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr / base_lr, (1.0 - progress) ** power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config_path: str) -> None:
    cfg = load_config(config_path)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not use_cuda:
        print("CUDA not available — training on CPU (this is normal without an NVIDIA GPU)")

    # Model
    model = SegFormerLiDAR(
        num_classes=cfg["model"]["num_classes"],
        lidar_channels=cfg["model"]["lidar_channels"],
        cam_channels=cfg["model"]["cam_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Dataset
    data_cfg = cfg["data"]
    train_ds = ERCDataset(
        root=data_cfg["train_dir"],
        image_size=tuple(data_cfg["image_size"]),
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        lidar_mean=data_cfg["lidar_mean"],
        lidar_std=data_cfg["lidar_std"],
        augment=True,
    )
    val_ds = ERCDataset(
        root=data_cfg["val_dir"],
        image_size=tuple(data_cfg["image_size"]),
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        lidar_mean=data_cfg["lidar_mean"],
        lidar_std=data_cfg["lidar_std"],
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=use_cuda,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=use_cuda,
    )

    # Optimiser
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )

    steps_per_epoch = max(len(train_loader), 1)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    # Convert class weights to a tensor for the loss computation.
    class_weights = torch.tensor(
        cfg["training"]["class_weights"], dtype=torch.float32,
    ).to(device)
    ignore_index = cfg["training"]["ignore_index"]

    use_amp = cfg["training"]["mixed_precision"] and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = cfg["training"]["gradient_clip"]

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_miou = 0.0
    epochs = cfg["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        # --- Training epoch ---
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            lidar = batch["lidar"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model.compute_loss(
                    image, lidar, mask,
                    class_weights=class_weights,
                    ignore_index=ignore_index,
                )
                loss = out["loss"]

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            if step % cfg["training"]["log_interval"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[Epoch {epoch}/{epochs}] step {step}/{steps_per_epoch} "
                    f"loss={loss.item():.4f}  lr={lr:.2e}"
                )

        avg_loss = epoch_loss / steps_per_epoch
        elapsed = time.time() - t0
        print(f"Epoch {epoch} — avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")

        # --- Validation ---
        if epoch % cfg["training"]["val_interval"] == 0:
            model.eval()
            total_miou = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    image = batch["image"].to(device)
                    lidar = batch["lidar"].to(device)
                    mask = batch["mask"]

                    logits = model(image, lidar)
                    pred = logits.argmax(1).cpu()
                    miou = compute_miou(
                        pred, mask, cfg["model"]["num_classes"], ignore_index,
                    )
                    total_miou += miou
                    n_val += 1

            avg_miou = total_miou / max(n_val, 1)
            print(f"  → val mIoU = {avg_miou:.4f}")

            if avg_miou > best_miou:
                best_miou = avg_miou
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "miou": best_miou},
                    ckpt_dir / "best.pth",
                )
                print(f"  → saved best model (mIoU={best_miou:.4f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict()},
                ckpt_dir / f"epoch_{epoch}.pth",
            )

    print(f"\nTraining complete — best mIoU = {best_miou:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/erc_config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
