"""
Training loop with mixed precision, WandB logging, checkpointing, and early stopping.
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from monai.data import CacheDataset, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from src.utils.logging import log_metrics, log_image, print_log
from src.utils.seed import set_seed


class Trainer:
    """
    Full training pipeline with validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        callbacks: list = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.callbacks = callbacks or []

        # Mixed precision
        self.use_amp = cfg["training"].get("mixed_precision", True)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Checkpointing
        self.checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metric for validation
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

        # Post-processing for predictions
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # Sliding window inference roi size (same as training patch size)
        self.val_roi_size = cfg["preprocessing"]["patch_size"]
        self.sw_batch_size = cfg["training"].get("sw_batch_size", 4)
        self.sw_overlap = cfg["training"].get("sw_overlap", 0.5)

        # Early stopping
        es_cfg = cfg["training"].get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", False)
        self.patience = es_cfg.get("patience", 50)
        self.min_delta = es_cfg.get("min_delta", 0.001)

        # State
        self.best_dice = 0.0
        self.epochs_without_improvement = 0
        self.start_epoch = 0

        # Logging
        self.log_images_every = cfg["logging"].get("log_images_every_n_epochs", 10)

    def train(self) -> dict:
        """Run full training loop. Returns final metrics dict."""

        total_epochs = self.cfg["training"]["epochs"]
        print_log(f"Starting training for {total_epochs} epochs")
        print_log(f"  Mixed precision: {self.use_amp}")
        print_log(f"  Early stopping: {self.early_stopping_enabled} (patience={self.patience})")
        print_log(f"  Device: {self.device}")

        history = {"train_loss": [], "val_dice_mean": [], "val_dice_wt": [],
                    "val_dice_tc": [], "val_dice_et": [], "val_accuracy": []}

        for epoch in range(self.start_epoch, total_epochs):
            epoch_start = time.time()

            # --- Training epoch ---
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)

            # --- Validation epoch ---
            val_dice_per_class, val_dice_mean, val_accuracy = self._validate_epoch(epoch)
            history["val_dice_mean"].append(val_dice_mean)
            history["val_dice_wt"].append(val_dice_per_class[0])
            history["val_dice_tc"].append(val_dice_per_class[1])
            history["val_dice_et"].append(val_dice_per_class[2])
            history["val_accuracy"].append(val_accuracy)

            # --- Step scheduler ---
            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Console logging ---
            print_log(
                f"Epoch {epoch+1}/{total_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Dice: WT={val_dice_per_class[0]:.4f} TC={val_dice_per_class[1]:.4f} "
                f"ET={val_dice_per_class[2]:.4f} Mean={val_dice_mean:.4f} | "
                f"Acc: {val_accuracy:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # --- WandB logging ---
            log_metrics({
                "train/loss": train_loss,
                "val/dice_mean": val_dice_mean,
                "val/dice_wt": val_dice_per_class[0],
                "val/dice_tc": val_dice_per_class[1],
                "val/dice_et": val_dice_per_class[2],
                "val/accuracy": val_accuracy,
                "train/lr": current_lr,
                "train/epoch_time_s": elapsed,
            }, step=epoch)

            # --- Checkpointing ---
            self._save_checkpoint(epoch, val_dice_mean, is_last=True)

            if val_dice_mean > self.best_dice + self.min_delta:
                self.best_dice = val_dice_mean
                self._save_checkpoint(epoch, val_dice_mean, is_best=True)
                self.epochs_without_improvement = 0
                print_log(f"  → New best model! Dice={val_dice_mean:.4f}")
            else:
                self.epochs_without_improvement += 1

            # --- Early stopping ---
            if self.early_stopping_enabled and self.epochs_without_improvement >= self.patience:
                print_log(f"Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                break

            # --- Callbacks ---
            for callback in self.callbacks:
                callback(epoch, val_dice_mean)

        print_log(f"Training complete. Best mean Dice: {self.best_dice:.4f}")
        return history

    def _train_epoch(self, epoch: int) -> float:
        """Single training epoch. Returns mean loss."""
        self.model.train()
        epoch_loss = 0.0
        step_count = 0

        for batch in self.train_loader:
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            step_count += 1

        return epoch_loss / max(step_count, 1)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> tuple:
        """Single validation epoch. Returns (dice_per_class, mean_dice, accuracy)."""
        self.model.eval()
        self.dice_metric.reset()

        if len(self.val_loader) == 0:
            num_classes = self.cfg["model"].get("out_channels", 3)
            return [0.0] * num_classes, 0.0, 0.0

        total_correct = 0
        total_voxels = 0

        for batch_idx, batch in enumerate(self.val_loader):
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = sliding_window_inference(
                inputs,
                roi_size=self.val_roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=self.model,
                overlap=self.sw_overlap,
            )

            # Post-process: sigmoid → threshold
            preds = [self.post_pred(o) for o in decollate_batch(outputs)]
            labels_list = decollate_batch(labels)

            self.dice_metric(y_pred=preds, y=labels_list)

            # Voxel-wise accuracy (per-channel match, averaged across channels)
            for p, l in zip(preds, labels_list):
                total_correct += (p == l).sum().item()
                total_voxels += l.numel()

            # Log sample visualizations periodically
            if batch_idx == 0 and (epoch + 1) % self.log_images_every == 0:
                self._log_sample_visualization(inputs[0], labels[0], preds[0], epoch)

        # Aggregate metrics
        dice_per_class = self.dice_metric.aggregate()
        # Flatten to 1-D and convert to Python floats
        dice_per_class = dice_per_class.flatten().tolist()
        # Ensure exactly 3 class scores (WT, TC, ET)
        num_expected = 3
        while len(dice_per_class) < num_expected:
            dice_per_class.append(0.0)
        dice_per_class = dice_per_class[:num_expected]
        mean_dice = sum(dice_per_class) / len(dice_per_class)
        accuracy = total_correct / max(total_voxels, 1)

        return dice_per_class, mean_dice, accuracy

    def _log_sample_visualization(self, image, label, pred, epoch):
        """Log a mid-training segmentation overlay to WandB."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            original_backend = matplotlib.get_backend()
            matplotlib.use("Agg")
            import numpy as np

            # Get center slice of the first modality (FLAIR)
            img_np = image[0].cpu().numpy()  # (D, H, W) — FLAIR channel
            center_slice = img_np.shape[0] // 2

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # MRI slice
            axes[0].imshow(img_np[center_slice], cmap="gray")
            axes[0].set_title("FLAIR")
            axes[0].axis("off")

            # Ground truth overlay
            gt_np = label.cpu().numpy()  # (3, D, H, W)
            gt_overlay = np.zeros((*gt_np.shape[1:][1:], 3))  # (H, W, 3) RGB
            gt_overlay[..., 0] = gt_np[2, center_slice]  # ET → red
            gt_overlay[..., 1] = gt_np[0, center_slice]  # WT → green
            gt_overlay[..., 2] = gt_np[1, center_slice]  # TC → blue
            axes[1].imshow(img_np[center_slice], cmap="gray")
            axes[1].imshow(gt_overlay, alpha=0.4)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            # Prediction overlay
            pred_np = pred.cpu().numpy()
            pred_overlay = np.zeros((*pred_np.shape[1:][1:], 3))
            pred_overlay[..., 0] = pred_np[2, center_slice]
            pred_overlay[..., 1] = pred_np[0, center_slice]
            pred_overlay[..., 2] = pred_np[1, center_slice]
            axes[2].imshow(img_np[center_slice], cmap="gray")
            axes[2].imshow(pred_overlay, alpha=0.4)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.suptitle(f"Epoch {epoch + 1}")
            plt.tight_layout()

            log_image("val/segmentation_overlay", fig,
                       caption=f"Epoch {epoch+1}", step=epoch)
            plt.close(fig)
            matplotlib.use(original_backend)
        except Exception as e:
            print_log(f"Failed to log visualization: {e}", level="WARN")

    def _save_checkpoint(self, epoch, dice, is_best=False, is_last=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_dice": self.best_dice,
            "config": self.cfg,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, path)

        if is_last:
            path = self.checkpoint_dir / "last_model.pth"
            torch.save(checkpoint, path)

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_dice = checkpoint.get("best_dice", 0.0)
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        print_log(f"Resumed from epoch {self.start_epoch}, best dice={self.best_dice:.4f}")
