"""
Hyperparameter tuning via Optuna.

Runs short training trials in dev mode to find optimal hyperparameters,
then saves the best config as configs/tuned.yaml.

Usage:
    python scripts/tune_hyperparams.py --n_trials 25 --epochs_per_trial 15
    python scripts/tune_hyperparams.py --n_trials 10 --epochs_per_trial 10 --max_samples 10
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
from monai.data import CacheDataset, DataLoader, list_data_collate

from src.utils.config import load_config, CONFIGS_DIR
from src.utils.seed import set_seed
from src.utils.logging import print_log
from src.data.dataset import discover_brats_samples, get_monai_file_list
from src.data.splits import create_splits
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet3d import get_model
from src.training.losses import get_loss_function
from src.training.trainer import Trainer


def objective(trial, args, base_cfg, train_samples, val_samples):
    """Optuna objective function — returns negative mean Dice (to minimize)."""
    import copy
    cfg = copy.deepcopy(base_cfg)

    # --- Sample hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.05)

    channels_options = {
        "small": [16, 32, 64, 128],
        "medium": [24, 48, 96, 192],
        "large": [32, 64, 128, 256],
    }
    channels_choice = trial.suggest_categorical("channels", ["small", "medium", "large"])
    channels = channels_options[channels_choice]

    patch_options = {
        "small": [64, 64, 64],
        "medium": [96, 96, 96],
    }
    patch_choice = trial.suggest_categorical("patch_size", ["small", "medium"])
    patch_size = patch_options[patch_choice]

    # Apply sampled HPs to config
    cfg["training"]["optimizer"]["lr"] = lr
    cfg["training"]["optimizer"]["weight_decay"] = weight_decay
    cfg["model"]["dropout"] = dropout
    cfg["model"]["channels"] = channels
    cfg["model"]["strides"] = [2] * (len(channels) - 1)
    cfg["preprocessing"]["patch_size"] = patch_size
    cfg["training"]["epochs"] = args.epochs_per_trial
    cfg["training"]["early_stopping"]["enabled"] = False
    cfg["logging"]["use_wandb"] = False

    # Unique checkpoint dir per trial
    trial_dir = Path(cfg["paths"]["checkpoint_dir"]) / "hp_tuning" / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["checkpoint_dir"] = str(trial_dir)

    print(f"\n{'─' * 50}")
    print(f"Trial {trial.number}: lr={lr:.2e}, wd={weight_decay:.2e}, "
          f"drop={dropout:.2f}, ch={channels_choice}, patch={patch_choice}")
    print(f"{'─' * 50}")

    # Setup
    set_seed(cfg["seed"] + trial.number)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = get_monai_file_list(train_samples)
    val_files = get_monai_file_list(val_samples)

    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=0,
    )
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_rate=1.0, num_workers=0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=0, collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
    )

    model = get_model(cfg)
    loss_fn = get_loss_function(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs_per_trial, T_mult=1, eta_min=1e-7,
    )

    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        scheduler=scheduler, train_loader=train_loader,
        val_loader=val_loader, cfg=cfg, device=device,
    )

    # Train
    try:
        history = trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print_log(f"Trial {trial.number} OOM — pruning")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        raise

    # Return best dice (Optuna minimizes, so negate)
    best_dice = trainer.best_dice
    print(f"Trial {trial.number} result: best Dice = {best_dice:.4f}")

    # Report intermediate values for pruning
    for epoch, dice in enumerate(history.get("val_dice_mean", [])):
        trial.report(dice, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return -best_dice  # Negative because Optuna minimizes


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--epochs_per_trial", type=int, default=15)
    parser.add_argument("--max_samples", type=int, default=20,
                        help="Max training samples per trial")
    parser.add_argument("--config", type=str, default="dev",
                        choices=["default", "dev", "full"])
    args = parser.parse_args()

    # Import optuna here so it fails early with a clear message
    global optuna
    try:
        import optuna
    except ImportError:
        print("ERROR: Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    cfg = load_config(args.config)
    cfg["data"]["max_samples"] = args.max_samples

    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print(f"  Trials: {args.n_trials}")
    print(f"  Epochs per trial: {args.epochs_per_trial}")
    print(f"  Max samples: {args.max_samples}")
    print("=" * 60)

    # Data setup (shared across trials)
    set_seed(cfg["seed"])
    samples = discover_brats_samples(
        cfg["paths"]["data_root"], max_samples=args.max_samples
    )
    train_samples, val_samples, _ = create_splits(
        samples,
        ratios=cfg["data"]["split_ratios"],
        seed=cfg["seed"],
        splits_dir=cfg["paths"]["splits_dir"],
        force=True,  # Force re-split for HP tuning subset
    )

    # Run Optuna study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name="brats-hp-tuning",
    )

    study.optimize(
        lambda trial: objective(trial, args, cfg, train_samples, val_samples),
        n_trials=args.n_trials,
        catch=(RuntimeError,),
    )

    # Print results
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"  Best Dice: {-study.best_trial.value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # Save best config
    best = study.best_trial.params
    channels_map = {
        "small": [16, 32, 64, 128],
        "medium": [24, 48, 96, 192],
        "large": [32, 64, 128, 256],
    }
    patch_map = {
        "small": [64, 64, 64],
        "medium": [96, 96, 96],
    }

    tuned_config = {
        "preprocessing": {
            "patch_size": patch_map[best["patch_size"]],
        },
        "model": {
            "channels": channels_map[best["channels"]],
            "strides": [2] * (len(channels_map[best["channels"]]) - 1),
            "dropout": best["dropout"],
        },
        "training": {
            "optimizer": {
                "lr": best["lr"],
                "weight_decay": best["weight_decay"],
            },
        },
    }

    tuned_path = CONFIGS_DIR / "tuned.yaml"
    with open(tuned_path, "w") as f:
        yaml.dump(tuned_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest config saved to: {tuned_path}")
    print("Use with: python scripts/run_experiment.py --experiment unet_4ch --config tuned")


if __name__ == "__main__":
    main()
