"""
Compare trained models — load results, run statistical tests, generate report.

Usage:
    python scripts/compare_models.py --config full
    python scripts/compare_models.py --config full --results_dir outputs/results
"""

import sys
import json
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils.config import load_config
from src.utils.logging import print_log
from src.evaluation.metrics import CLASS_NAMES
from src.evaluation.statistical_tests import (
    paired_wilcoxon_test,
    friedman_test,
    bootstrap_confidence_interval,
    cohens_d,
    mcnemar_test,
    run_full_comparison,
)


def load_experiment_results(results_dir: Path) -> dict:
    """Load per-patient Dice scores from all experiment result directories."""
    results = {}

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name in ("cross_modality", "hp_tuning", "plots"):
            continue

        # Look for evaluation results JSON
        eval_path = exp_dir / "evaluation_results.json"
        history_path = exp_dir / "training_history.json"

        if eval_path.exists():
            with open(eval_path) as f:
                data = json.load(f)
            if "per_case_dice" in data:
                dice_array = np.array(data["per_case_dice"])  # (n_patients, 3)
                results[exp_dir.name] = {
                    "dice_per_patient": dice_array,
                    "has_evaluation": True,
                }
                print_log(f"  Loaded {exp_dir.name}: {len(dice_array)} patients")
        elif history_path.exists():
            # Can still load training history for curves
            with open(history_path) as f:
                history = json.load(f)
            results[exp_dir.name] = {
                "training_history": history,
                "has_evaluation": False,
            }
            print_log(f"  Loaded {exp_dir.name}: training history only (no evaluation yet)")

    return results


def generate_comparison_tables(comparison: dict, output_dir: Path):
    """Generate CSV comparison tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bootstrap CI table
    ci_path = output_dir / "confidence_intervals.csv"
    with open(ci_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Class", "Mean Dice", "95% CI Lower", "95% CI Upper", "Std"])
        for model, cis in comparison["bootstrap_ci"].items():
            for cls_name, ci in cis.items():
                writer.writerow([
                    model, cls_name,
                    f"{ci['mean']:.4f}",
                    f"{ci['ci_lower']:.4f}",
                    f"{ci['ci_upper']:.4f}",
                    f"{ci['std']:.4f}",
                ])
    print_log(f"  Saved: {ci_path}")

    # 2. Pairwise Wilcoxon table
    wilcoxon_path = output_dir / "pairwise_wilcoxon.csv"
    with open(wilcoxon_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model A", "Model B", "Mean A", "Mean B", "Mean Diff",
                         "Statistic", "p-value", "Significant", "Interpretation"])
        for w in comparison["pairwise_wilcoxon"]:
            writer.writerow([
                w["model_a"], w["model_b"],
                f"{w['mean_a']:.4f}", f"{w['mean_b']:.4f}",
                f"{w['mean_difference']:.4f}",
                f"{w['statistic']:.2f}",
                f"{w['p_value']:.6f}",
                w["significant"],
                w["interpretation"],
            ])
    print_log(f"  Saved: {wilcoxon_path}")

    # 3. Effect sizes
    effect_path = output_dir / "effect_sizes.csv"
    with open(effect_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model A", "Model B", "Cohen's d", "Magnitude"])
        for e in comparison["effect_sizes"]:
            writer.writerow([
                e["model_a"], e["model_b"],
                f"{e['cohens_d']:.4f}", e["magnitude"],
            ])
    print_log(f"  Saved: {effect_path}")

    # 4. Summary table
    summary_path = output_dir / "model_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Mean Dice", "95% CI", "Dice WT", "Dice TC", "Dice ET"])
        for model, cis in comparison["bootstrap_ci"].items():
            overall = cis["overall"]
            wt = cis.get("WT", {})
            tc = cis.get("TC", {})
            et = cis.get("ET", {})
            writer.writerow([
                model,
                f"{overall['mean']:.4f}",
                f"[{overall['ci_lower']:.4f}, {overall['ci_upper']:.4f}]",
                f"{wt.get('mean', 0):.4f}",
                f"{tc.get('mean', 0):.4f}",
                f"{et.get('mean', 0):.4f}",
            ])
    print_log(f"  Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare trained models with statistical tests")
    parser.add_argument("--config", type=str, default="full")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    results_dir = Path(args.results_dir) if args.results_dir else Path(cfg["paths"]["output_dir"]) / "results"

    print("=" * 60)
    print("MODEL COMPARISON — Statistical Analysis")
    print(f"  Results dir: {results_dir}")
    print("=" * 60)

    # Load results
    print_log("Loading experiment results...")
    results = load_experiment_results(results_dir)

    # Filter to only experiments with evaluation data
    eval_results = {k: v for k, v in results.items() if v.get("has_evaluation", False)}

    if len(eval_results) < 2:
        print(f"\nNeed at least 2 evaluated experiments for comparison.")
        print(f"Found {len(eval_results)} with evaluation data: {list(eval_results.keys())}")
        print(f"Run evaluate.py on your experiments first.")
        return

    print_log(f"Found {len(eval_results)} experiments with evaluation data")

    # Run comparisons
    print_log("Running statistical tests...")
    comparison = run_full_comparison(eval_results, class_names=["WT", "TC", "ET"])

    # Generate tables
    output_dir = results_dir / "comparison"
    print_log("Generating comparison tables...")
    generate_comparison_tables(comparison, output_dir)

    # Save full comparison JSON
    comparison_json_path = output_dir / "full_comparison.json"

    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(comparison_json_path, "w") as f:
        json.dump(comparison, f, indent=2, default=to_serializable)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for model in comparison["bootstrap_ci"]:
        ci = comparison["bootstrap_ci"][model]["overall"]
        print(f"  {model}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    if comparison.get("friedman"):
        f = comparison["friedman"]
        print(f"\n  Friedman test: χ²={f['statistic']:.2f}, p={f['p_value']:.6f}")
        print("  Rankings:")
        for r in f["rankings"]:
            print(f"    #{r['avg_rank']:.1f}: {r['model']}")

    print(f"\n  Pairwise comparisons:")
    for w in comparison["pairwise_wilcoxon"]:
        sig = "✓" if w["significant"] else "✗"
        print(f"    {w['model_a']} vs {w['model_b']}: p={w['p_value']:.4f} [{sig}]")

    print(f"\n✅ Comparison complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
