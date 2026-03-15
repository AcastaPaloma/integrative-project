"""
Single-experiment runner — trains one model from experiments.yaml.

Usage:
    python scripts/run_experiment.py --experiment unet_4ch --config full
    python scripts/run_experiment.py --experiment unet_flair --config full --resume
    python scripts/run_experiment.py --experiment cnn_4ch --config full
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
from monai.data import Dataset, DataLoader, list_data_collate

from src.utils.config import load_config, CONFIGS_DIR
from src.utils.seed import set_seed
from src.utils.logging import init_wandb, finish_wandb, print_log
from src.data.dataset import discover_brats_samples, get_monai_file_list
from src.data.splits import create_splits
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet3d import get_model
from src.training.losses import get_loss_function
from src.training.trainer import Trainer


def load_experiments_config() -> dict:
    """Load experiments.yaml."""
    exp_path = CONFIGS_DIR / "experiments.yaml"
    with open(exp_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run a single training experiment")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name from experiments.yaml")
    parser.add_argument("--config", type=str, default="full",
                        choices=["default", "dev", "full", "tuned"])
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    # Load configs
    cfg = load_config(args.config)
    exp_cfg = load_experiments_config()

    if args.experiment not in exp_cfg["experiments"]:
        available = list(exp_cfg["experiments"].keys())
        print(f"Unknown experiment '{args.experiment}'. Available: {available}")
        sys.exit(1)

    experiment = exp_cfg["experiments"][args.experiment]
    modalities = experiment["modalities"]
    architecture = experiment["model"]

    print("=" * 60)
    print(f"EXPERIMENT: {args.experiment}")
    print(f"  Description: {experiment['description']}")
    print(f"  Architecture: {architecture}")
    print(f"  Modalities: {modalities}")
    print(f"  Config: {args.config}")
    print("=" * 60)

    # Override model config for this experiment
    cfg["model"]["architecture"] = architecture
    cfg["model"]["in_channels"] = len(modalities)

    # Experiment-specific paths
    exp_checkpoint_dir = str(Path(cfg["paths"]["checkpoint_dir"]) / args.experiment)
    cfg["paths"]["checkpoint_dir"] = exp_checkpoint_dir
    Path(exp_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    exp_output_dir = str(Path(cfg["paths"]["output_dir"]) / "results" / args.experiment)
    Path(exp_output_dir).mkdir(parents=True, exist_ok=True)

    # Seed
    set_seed(cfg["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print_log(f"GPU: {torch.cuda.get_device_name(0)}")
        print_log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print_log("WARNING: No GPU detected!")

    # WandB
    run_name = f"brats-{args.experiment}-{args.config}"
    init_wandb(cfg, run_name=run_name)

    # Data
    max_samples = cfg["data"].get("max_samples")
    samples = discover_brats_samples(cfg["paths"]["data_root"], max_samples=max_samples)

    train_samples, val_samples, test_samples = create_splits(
        samples,
        ratios=cfg["data"]["split_ratios"],
        seed=cfg["seed"],
        splits_dir=cfg["paths"]["splits_dir"],
    )

    # Build file lists with modality filtering
    train_files = get_monai_file_list(train_samples, modalities=modalities)
    val_files = get_monai_file_list(val_samples, modalities=modalities)

    print_log(f"Training samples: {len(train_files)}")
    print_log(f"Validation samples: {len(val_files)}")
    print_log(f"Modalities: {modalities} ({len(modalities)} channels)")

    # Transforms
    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    # Datasets — plain Dataset: no disk bloat, no Windows multiprocessing issues
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # DataLoaders
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

    # Model
    model = get_model(cfg)

    # Loss
    loss_fn = get_loss_function(cfg)

    # Optimizer
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
    )

    # Scheduler
    sch_cfg = cfg["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sch_cfg["T_0"],
        T_mult=sch_cfg.get("T_mult", 2),
        eta_min=sch_cfg.get("eta_min", 1e-7),
    )

    # Trainer
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

    # Resume
    if args.resume:
        last_ckpt = Path(exp_checkpoint_dir) / "last_model.pth"
        if last_ckpt.exists():
            trainer.resume_from_checkpoint(str(last_ckpt))
        else:
            print_log(f"No checkpoint found at {last_ckpt}, training from scratch")

    # Train
    history = trainer.train()

    # Save training history
    import json
    history_path = Path(exp_output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print_log(f"Training history saved to {history_path}")

    # Cleanup
    finish_wandb()

    print(f"\n✅ Experiment '{args.experiment}' complete.")
    print(f"  Best checkpoint: {exp_checkpoint_dir}/best_model.pth")
    print(f"  Last checkpoint: {exp_checkpoint_dir}/last_model.pth")
    print(f"  History: {history_path}")


if __name__ == "__main__":
    main()
