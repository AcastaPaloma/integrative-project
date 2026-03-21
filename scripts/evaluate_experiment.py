"""
Evaluate a trained experiment — run inference + metrics + visualization.

Usage:
    python scripts/evaluate_experiment.py --experiment unet_4ch --config full
    python scripts/evaluate_experiment.py --experiment cnn_4ch --config full --split test
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import torch

from src.utils.config import load_config, CONFIGS_DIR
from src.utils.seed import set_seed
from src.utils.logging import print_log
from src.data.dataset import discover_brats_samples, get_monai_file_list
from src.data.splits import create_splits
from src.inference.predict import run_inference
from src.evaluation.metrics import compute_case_metrics, aggregate_metrics, CLASS_NAMES
from src.evaluation.visualize import create_overlay_figure, create_multi_view_figure


def load_experiments_config() -> dict:
    exp_path = CONFIGS_DIR / "experiments.yaml"
    with open(exp_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained experiment")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--config", type=str, default="full")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--num_vis", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = load_experiments_config()
    set_seed(cfg["seed"])

    if args.experiment not in exp_cfg["experiments"]:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)

    experiment = exp_cfg["experiments"][args.experiment]
    modalities = experiment["modalities"]

    # Override config
    cfg["model"]["architecture"] = experiment["model"]
    cfg["model"]["in_channels"] = len(modalities)
    cfg.setdefault("data", {})["modalities"] = modalities

    # Paths
    checkpoint_path = str(Path(cfg["paths"]["checkpoint_dir"]) / args.experiment / "best_model.pth")
    output_dir = Path(cfg["paths"]["output_dir"]) / "results" / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"EVALUATE: {args.experiment}")
    print(f"  {experiment['description']}")
    print(f"  Modalities: {modalities}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)

    # Data
    samples = discover_brats_samples(cfg["paths"]["data_root"])
    _, val_samples, test_samples = create_splits(
        samples,
        ratios=cfg["data"]["split_ratios"],
        seed=cfg["seed"],
        splits_dir=cfg["paths"]["splits_dir"],
    )
    target_samples = val_samples if args.split == "val" else test_samples

    # Run inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Need to update prediction dir per experiment
    cfg["paths"]["predictions_dir"] = str(output_dir / "predictions")

    results = run_inference(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        samples=target_samples,
        device=device,
        modalities=modalities,
    )

    # Compute metrics
    print_log("Computing metrics...")
    all_case_metrics = []
    per_case_dice = []

    for r in results:
        metrics = compute_case_metrics(r["prediction"], r["label"])
        all_case_metrics.append(metrics)
        per_case_dice.append(metrics["dice"])

        print_log(f"  {r['patient_id']}: Dice "
                  f"WT={metrics['dice'][0]:.4f} "
                  f"TC={metrics['dice'][1]:.4f} "
                  f"ET={metrics['dice'][2]:.4f}")

    # Aggregate
    agg = aggregate_metrics(all_case_metrics)

    print_log("\n--- Aggregated ---")
    for c, name in enumerate(CLASS_NAMES):
        print_log(f"  {name}: Dice={agg[f'dice/{name}/mean']:.4f}±{agg[f'dice/{name}/std']:.4f}")
    print_log(f"  Mean Dice: {agg['dice/mean_all']:.4f}")

    # Save results
    eval_results = {
        "experiment": args.experiment,
        "description": experiment["description"],
        "modalities": modalities,
        "split": args.split,
        "per_case_dice": per_case_dice,
        "aggregated": {k: v for k, v in agg.items() if "values" not in k},
    }

    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    eval_path = output_dir / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=to_serializable)
    print_log(f"Results saved to {eval_path}")

    # Visualizations
    import matplotlib.pyplot as plt
    from src.data.transforms import get_val_transforms
    from monai.data import Dataset

    file_list = get_monai_file_list(target_samples, modalities=modalities)
    ds = Dataset(data=file_list, transform=get_val_transforms(cfg))

    for i, r in enumerate(results[:args.num_vis]):
        data = ds[i]
        image = data["image"].numpy()

        fig = create_overlay_figure(
            image=image, gt=r["label"], pred=r["prediction"],
            patient_id=r["patient_id"],
            save_path=str(vis_dir / f"{r['patient_id']}_overlay.png"),
        )
        plt.close(fig)

        fig2 = create_multi_view_figure(
            image=image, gt=r["label"], pred=r["prediction"],
            patient_id=r["patient_id"],
            save_path=str(vis_dir / f"{r['patient_id']}_multiview.png"),
        )
        plt.close(fig2)

    print(f"\n✅ Evaluation complete for '{args.experiment}'")
    print(f"  Results: {eval_path}")
    print(f"  Visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
