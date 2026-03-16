"""
Phase 2+3 — Preprocessing + Training

Sets up data pipelines, builds model, and runs training loop.

Usage:
    python scripts/train.py --config dev     # fast iteration (~5 min on GTX 1080)
    python scripts/train.py --config full    # real training run
    python scripts/train.py --config full --resume checkpoints/last_model.pth
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from monai.data import CacheDataset, DataLoader, list_data_collate

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import init_wandb, finish_wandb, print_log
from src.data.dataset import discover_brats_samples, get_monai_file_list
from src.data.splits import create_splits
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet3d import get_model
from src.training.losses import get_loss_function
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for brain tumor segmentation")
    parser.add_argument("--config", type=str, default="dev", choices=["default", "dev", "full"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=" * 60)
    print("PHASE 2+3 — Preprocessing + Training")
    print(f"  Config: {args.config}")
    print("=" * 60)

    # --- Seed ---
    set_seed(cfg["seed"])

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print_log(f"GPU: {torch.cuda.get_device_name(0)}")
        print_log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print_log("WARNING: No GPU detected, training on CPU will be very slow!")

    # --- WandB ---
    run_name = f"brats-3dunet-{args.config}"
    init_wandb(cfg, run_name=run_name)

    # --- Data ---
    max_samples = cfg["data"].get("max_samples")
    samples = discover_brats_samples(cfg["paths"]["data_root"], max_samples=max_samples)

    train_samples, val_samples, test_samples = create_splits(
        samples,
        ratios=cfg["data"]["split_ratios"],
        seed=cfg["seed"],
        splits_dir=cfg["paths"]["splits_dir"],
    )

    # Apply max_samples limit AFTER split (to preserve split integrity)
    if max_samples:
        # For dev mode, we might have fewer total samples than the split expects
        pass

    train_files = get_monai_file_list(train_samples)
    val_files = get_monai_file_list(val_samples)

    print_log(f"Training samples: {len(train_files)}")
    print_log(f"Validation samples: {len(val_files)}")

    # --- Transforms ---
    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    # --- Datasets ---
    # Cache rate controls RAM usage only; all samples are still iterated each epoch.
    cache_cfg = cfg.get("data", {}).get("cache", {})
    train_cache_rate = cache_cfg.get("train_rate", 0.08)
    val_cache_rate = cache_cfg.get("val_rate", 0.15)
    cache_workers = cache_cfg.get("num_workers", 0)

    print_log(
        f"Cache settings: train_rate={train_cache_rate:.2f}, "
        f"val_rate={val_cache_rate:.2f}, workers={cache_workers}"
    )

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=train_cache_rate,
        num_workers=cache_workers,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=val_cache_rate,
        num_workers=cache_workers,
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 2),
        collate_fn=list_data_collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=True,
    )

    # --- Model ---
    model = get_model(cfg)

    # --- Loss ---
    loss_fn = get_loss_function(cfg)

    # --- Optimizer ---
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
    )

    # --- Scheduler ---
    sch_cfg = cfg["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sch_cfg["T_0"],
        T_mult=sch_cfg.get("T_mult", 2),
        eta_min=sch_cfg.get("eta_min", 1e-7),
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    # --- Resume ---
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # --- Train ---
    history = trainer.train()

    # --- Cleanup ---
    finish_wandb()

    print("\n✅ Training complete.")
    print(f"  Best checkpoint: {cfg['paths']['checkpoint_dir']}/best_model.pth")
    print(f"  Last checkpoint: {cfg['paths']['checkpoint_dir']}/last_model.pth")


if __name__ == "__main__":
    main()
