"""
Generate publication-quality plots from experiment results.

Usage:
    python scripts/generate_plots.py --config full
    python scripts/generate_plots.py --results_dir outputs/results
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.utils.config import load_config
from src.utils.logging import print_log
from src.evaluation.metrics import CLASS_NAMES


# Publication-quality style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#E91E63", "#8BC34A", "#FF5722", "#673AB7",
]


def load_training_histories(results_dir: Path) -> dict:
    """Load training histories from all experiments."""
    histories = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name in ("cross_modality", "hp_tuning", "comparison", "plots"):
            continue
        history_path = exp_dir / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                histories[exp_dir.name] = json.load(f)
    return histories


def load_evaluation_results(results_dir: Path) -> dict:
    """Load evaluation results from all experiments."""
    results = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        eval_path = exp_dir / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path) as f:
                results[exp_dir.name] = json.load(f)
    return results


def plot_training_curves(histories: dict, output_dir: Path):
    """Plot loss and Dice curves across all experiments."""
    if not histories:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    for i, (name, h) in enumerate(histories.items()):
        if "train_loss" in h:
            epochs = range(1, len(h["train_loss"]) + 1)
            axes[0].plot(epochs, h["train_loss"], label=name,
                        color=COLORS[i % len(COLORS)], linewidth=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Dice curves
    for i, (name, h) in enumerate(histories.items()):
        if "val_dice_mean" in h:
            epochs = range(1, len(h["val_dice_mean"]) + 1)
            axes[1].plot(epochs, h["val_dice_mean"], label=name,
                        color=COLORS[i % len(COLORS)], linewidth=1.5)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Mean Dice")
    axes[1].set_title("Validation Dice Curves")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "training_curves.png"
    fig.savefig(save_path)
    plt.close(fig)
    print_log(f"  Saved: {save_path}")


def plot_per_class_dice_bar(results: dict, output_dir: Path):
    """Bar chart of per-class Dice scores across all models."""
    if not results:
        return

    models = []
    dice_data = {"WT": [], "TC": [], "ET": []}
    dice_err = {"WT": [], "TC": [], "ET": []}

    for name, r in results.items():
        if "aggregated" not in r:
            continue
        agg = r["aggregated"]
        models.append(name)
        for c, cname in enumerate(["WT", "TC", "ET"]):
            full_name = CLASS_NAMES[c]
            mean_key = f"dice/{full_name}/mean"
            std_key = f"dice/{full_name}/std"
            if mean_key in agg:
                dice_data[cname].append(agg[mean_key])
                dice_err[cname].append(agg.get(std_key, 0))
            else:
                dice_data[cname].append(0)
                dice_err[cname].append(0)

    if not models:
        return

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))

    bars1 = ax.bar(x - width, dice_data["WT"], width, yerr=dice_err["WT"],
                   label="Whole Tumor", color="#4CAF50", alpha=0.85, capsize=3)
    bars2 = ax.bar(x, dice_data["TC"], width, yerr=dice_err["TC"],
                   label="Tumor Core", color="#2196F3", alpha=0.85, capsize=3)
    bars3 = ax.bar(x + width, dice_data["ET"], width, yerr=dice_err["ET"],
                   label="Enhancing Tumor", color="#F44336", alpha=0.85, capsize=3)

    ax.set_xlabel("Model")
    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Class Dice Scores Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "dice_comparison_bar.png"
    fig.savefig(save_path)
    plt.close(fig)
    print_log(f"  Saved: {save_path}")


def plot_cross_modality_heatmap(results_dir: Path, output_dir: Path):
    """Heatmap showing cross-modality performance degradation."""
    cross_path = results_dir / "cross_modality" / "cross_modality_results.json"
    if not cross_path.exists():
        print_log("  No cross-modality results found, skipping heatmap")
        return

    with open(cross_path) as f:
        cross_results = json.load(f)

    if not cross_results:
        return

    # Build matrix: rows = test config, cols = WT/TC/ET Dice
    test_names = []
    dice_matrix = []

    for test_name, result in cross_results.items():
        test_names.append(result.get("description", test_name))
        agg = result.get("aggregated", {})
        row = []
        for c, cname in enumerate(CLASS_NAMES):
            mean_key = f"dice/{cname}/mean"
            row.append(agg.get(mean_key, 0))
        dice_matrix.append(row)

    if not dice_matrix:
        return

    dice_matrix = np.array(dice_matrix)

    fig, ax = plt.subplots(figsize=(8, max(4, len(test_names) * 0.5)))

    im = ax.imshow(dice_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["WT", "TC", "ET"])
    ax.set_yticks(range(len(test_names)))
    ax.set_yticklabels(test_names, fontsize=9)

    # Annotate cells
    for i in range(len(test_names)):
        for j in range(3):
            val = dice_matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                   color=color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Dice Score", fraction=0.046, pad=0.04)
    ax.set_title("Cross-Modality Portability — Dice Scores\n(4ch model tested with modality subsets)")
    plt.tight_layout()

    save_path = output_dir / "cross_modality_heatmap.png"
    fig.savefig(save_path)
    plt.close(fig)
    print_log(f"  Saved: {save_path}")


def plot_box_plots(results: dict, output_dir: Path):
    """Box plots of per-patient Dice distributions."""
    if not results:
        return

    data_by_model = {}
    for name, r in results.items():
        if "per_case_dice" in r:
            data_by_model[name] = np.array(r["per_case_dice"]).mean(axis=1)

    if len(data_by_model) < 2:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(data_by_model) * 1.2), 6))

    labels = list(data_by_model.keys())
    data = [data_by_model[name] for name in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)

    ax.set_ylabel("Mean Dice Score (per patient)")
    ax.set_title("Per-Patient Dice Score Distributions")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "dice_boxplots.png"
    fig.savefig(save_path)
    plt.close(fig)
    print_log(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate result visualizations")
    parser.add_argument("--config", type=str, default="full")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(args.results_dir) if args.results_dir else Path(cfg["paths"]["output_dir"]) / "results"

    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING PLOTS")
    print(f"  Results: {results_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Training curves
    print_log("Training curves...")
    histories = load_training_histories(results_dir)
    if histories:
        plot_training_curves(histories, output_dir)
    else:
        print_log("  No training histories found")

    # Dice bar chart
    print_log("Dice comparison bar chart...")
    eval_results = load_evaluation_results(results_dir)
    if eval_results:
        plot_per_class_dice_bar(eval_results, output_dir)
        plot_box_plots(eval_results, output_dir)
    else:
        print_log("  No evaluation results found")

    # Cross-modality heatmap
    print_log("Cross-modality heatmap...")
    plot_cross_modality_heatmap(results_dir, output_dir)

    print(f"\n✅ Plots generated in: {output_dir}")


if __name__ == "__main__":
    main()
