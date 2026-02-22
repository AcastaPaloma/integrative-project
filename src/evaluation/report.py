"""
Generate markdown evaluation report with metrics tables and embedded visualizations.
"""

import json
from pathlib import Path
from typing import Dict, List

from src.evaluation.metrics import CLASS_NAMES


def generate_report(
    aggregated_metrics: Dict,
    case_metrics: List[Dict],
    patient_ids: List[str],
    visualization_dir: str,
    output_path: str,
) -> str:
    """
    Generate a markdown evaluation report.

    Args:
        aggregated_metrics: Output of aggregate_metrics()
        case_metrics: List of per-case metric dicts
        patient_ids: Corresponding patient IDs
        visualization_dir: Directory containing overlay PNGs
        output_path: Where to save the markdown report

    Returns:
        Path to the generated report
    """
    lines = []

    lines.append("# Brain Tumor Segmentation — Evaluation Report\n")
    lines.append("## Summary Metrics\n")

    # Summary table
    lines.append("| Region | Dice (mean ± std) | HD95 (mean ± std) | Precision | Recall |")
    lines.append("|--------|-------------------|--------------------|-----------| -------|")

    for c, name in enumerate(CLASS_NAMES):
        dice_m = aggregated_metrics.get(f"dice/{name}/mean", 0)
        dice_s = aggregated_metrics.get(f"dice/{name}/std", 0)
        hd_m = aggregated_metrics.get(f"hausdorff95/{name}/mean", 0)
        hd_s = aggregated_metrics.get(f"hausdorff95/{name}/std", 0)
        prec_m = aggregated_metrics.get(f"precision/{name}/mean", 0)
        rec_m = aggregated_metrics.get(f"recall/{name}/mean", 0)

        lines.append(
            f"| {name} | {dice_m:.4f} ± {dice_s:.4f} | "
            f"{hd_m:.2f} ± {hd_s:.2f} | {prec_m:.4f} | {rec_m:.4f} |"
        )

    mean_dice = aggregated_metrics.get("dice/mean_all", 0)
    lines.append(f"\n**Mean Dice (all classes):** {mean_dice:.4f}\n")

    # Per-case breakdown
    lines.append("---\n")
    lines.append("## Per-Case Breakdown\n")
    lines.append("| Patient | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET |")
    lines.append("|---------|---------|---------|---------|---------|---------|---------|")

    for i, (pid, metrics) in enumerate(zip(patient_ids, case_metrics)):
        d = metrics["dice"]
        h = metrics["hausdorff95"]
        lines.append(
            f"| {pid} | {d[0]:.4f} | {d[1]:.4f} | {d[2]:.4f} | "
            f"{h[0]:.2f} | {h[1]:.2f} | {h[2]:.2f} |"
        )

    # Visualizations
    vis_dir = Path(visualization_dir)
    vis_files = sorted(vis_dir.glob("*.png")) if vis_dir.exists() else []

    if vis_files:
        lines.append("\n---\n")
        lines.append("## Qualitative Results\n")
        for vf in vis_files:
            lines.append(f"### {vf.stem}\n")
            lines.append(f"![{vf.stem}]({vf.name})\n")

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)

    # Also save raw metrics as JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        # Convert any non-serializable numpy types
        clean_metrics = {}
        for k, v in aggregated_metrics.items():
            if isinstance(v, list):
                clean_metrics[k] = v
            else:
                clean_metrics[k] = float(v)
        json.dump(clean_metrics, f, indent=2)

    print(f"[Report] Saved evaluation report → {output_path}")
    print(f"[Report] Saved raw metrics → {json_path}")

    return str(output_path)
