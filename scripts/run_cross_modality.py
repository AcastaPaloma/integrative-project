"""
Cross-modality inference — test a trained model with different modality subsets.

Loads a 4-channel model and tests it with zero-padded modality subsets
to measure "portability" (can one model handle partial input?).

Usage:
    python scripts/run_cross_modality.py --config full
    python scripts/run_cross_modality.py --config dev --checkpoint checkpoints/unet_4ch/best_model.pth
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer
from monai.transforms import Activations, AsDiscrete, Compose

from src.utils.config import load_config, CONFIGS_DIR
from src.utils.seed import set_seed
from src.utils.logging import print_log
from src.data.dataset import discover_brats_samples, get_monai_file_list
from src.data.splits import create_splits
from src.data.transforms import get_val_transforms
from src.models.unet3d import get_model, load_checkpoint
from src.evaluation.metrics import compute_case_metrics, aggregate_metrics, CLASS_NAMES


def load_experiments_config() -> dict:
    exp_path = CONFIGS_DIR / "experiments.yaml"
    with open(exp_path, "r") as f:
        return yaml.safe_load(f)


def zero_pad_batch(batch, active_modalities, all_modalities):
    """
    Zero-pad a batch's image tensor for missing modalities.

    If the model expects 4 channels but we only have 2 modalities,
    the missing channels are filled with zeros.
    """
    image = batch["image"]  # (B, C, D, H, W) — C = num active modalities

    all_mods = all_modalities
    # Build mapping: which channels in the full model correspond to active ones
    active_indices = [all_mods.index(m) for m in active_modalities]

    # Create zero-padded tensor
    B = image.shape[0]
    full_channels = len(all_mods)
    spatial = image.shape[2:]  # (D, H, W)

    padded = torch.zeros(B, full_channels, *spatial, dtype=image.dtype, device=image.device)
    for new_idx, orig_idx in enumerate(active_indices):
        padded[:, orig_idx] = image[:, new_idx]

    batch["image"] = padded
    return batch


def main():
    parser = argparse.ArgumentParser(description="Cross-modality portability tests")
    parser.add_argument("--config", type=str, default="full",
                        choices=["default", "dev", "full", "tuned"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to 4ch model checkpoint (auto-detected if not set)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    exp_cfg = load_experiments_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_modalities = ["flair", "t1", "t1ce", "t2"]

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(Path(cfg["paths"]["checkpoint_dir"]) / "unet_4ch" / "best_model.pth")
    if not Path(checkpoint_path).is_absolute():
        from src.utils.config import PROJECT_ROOT
        checkpoint_path = str(PROJECT_ROOT / checkpoint_path)

    print("=" * 60)
    print("CROSS-MODALITY PORTABILITY TESTS")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Split: {args.split}")
    print("=" * 60)

    # Ensure model is configured for 4 channels
    cfg["model"]["in_channels"] = 4
    cfg["model"]["architecture"] = "UNet"

    # Load model
    model = get_model(cfg)
    load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    # Data
    samples = discover_brats_samples(cfg["paths"]["data_root"])
    _, val_samples, test_samples = create_splits(
        samples,
        ratios=cfg["data"]["split_ratios"],
        seed=cfg["seed"],
        splits_dir=cfg["paths"]["splits_dir"],
    )
    target_samples = val_samples if args.split == "val" else test_samples

    # Inference setup
    patch_size = cfg["preprocessing"]["patch_size"]
    inf_cfg = cfg.get("inference", {})
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=inf_cfg.get("sw_batch_size", 1),
        overlap=inf_cfg.get("overlap", 0.25),
        mode=inf_cfg.get("mode", "gaussian"),
    )
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Output directory
    output_dir = Path(cfg["paths"]["output_dir"]) / "results" / "cross_modality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run each cross-modality test
    all_results = {}

    for test_cfg in exp_cfg.get("cross_modality_tests", []):
        test_mods = test_cfg["test_modalities"]
        test_name = f"test_{'_'.join(test_mods)}"
        description = test_cfg.get("description", test_name)

        print(f"\n{'─' * 50}")
        print(f"Test: {description}")
        print(f"  Active modalities: {test_mods}")
        print(f"  Zero-padded modalities: {[m for m in all_modalities if m not in test_mods]}")
        print(f"{'─' * 50}")

        # Load data with only the active modalities
        file_list = get_monai_file_list(target_samples, modalities=test_mods)
        val_transforms = get_val_transforms(cfg)
        dataset = Dataset(data=file_list, transform=val_transforms)

        case_metrics = []

        for idx in range(len(dataset)):
            data = dataset[idx]
            patient_id = target_samples[idx]["patient_id"]

            image = data["image"].unsqueeze(0)  # (1, C_subset, D, H, W)
            label = data["label"]  # (3, D, H, W)

            # Zero-pad to 4 channels
            batch = {"image": image}
            batch = zero_pad_batch(batch, test_mods, all_modalities)
            image_padded = batch["image"].to(device)

            with torch.no_grad():
                output = inferer(image_padded, model)

            pred = post_pred(output[0])
            pred_np = pred.cpu().numpy().astype(np.uint8)
            label_np = label.cpu().numpy().astype(np.float32)

            metrics = compute_case_metrics(pred_np, label_np)
            case_metrics.append(metrics)

            print_log(f"  {patient_id}: Dice WT={metrics['dice'][0]:.4f} "
                      f"TC={metrics['dice'][1]:.4f} ET={metrics['dice'][2]:.4f}")

        # Aggregate
        agg = aggregate_metrics(case_metrics)
        all_results[test_name] = {
            "description": description,
            "modalities": test_mods,
            "aggregated": {k: v for k, v in agg.items() if "values" not in k},
            "per_case": case_metrics,
        }

        print(f"\n  Mean Dice: {agg['dice/mean_all']:.4f}")
        for c, name in enumerate(CLASS_NAMES):
            print(f"    {name}: {agg[f'dice/{name}/mean']:.4f} ± {agg[f'dice/{name}/std']:.4f}")

    # Save all results
    results_path = output_dir / "cross_modality_results.json"
    # Convert numpy values for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=to_serializable)

    print(f"\n✅ Cross-modality tests complete. Results: {results_path}")


if __name__ == "__main__":
    main()
