"""
Phase 5 — Evaluation

Compute metrics on predictions vs ground truth, generate visualizations and report.

Usage:
    python scripts/evaluate.py --config dev
    python scripts/evaluate.py --config full --split test
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import print_log
from src.data.dataset import discover_brats_samples
from src.data.splits import create_splits
from src.inference.predict import run_inference
from src.evaluation.metrics import compute_case_metrics, aggregate_metrics, CLASS_NAMES
from src.evaluation.visualize import create_overlay_figure, create_multi_view_figure
from src.evaluation.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor segmentation")
    parser.add_argument("--config", type=str, default="dev", choices=["default", "dev", "full"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference step (use existing predictions)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    print("=" * 60)
    print("PHASE 5 — Evaluation")
    print("=" * 60)

    # Discover data
    samples = discover_brats_samples(cfg["paths"]["data_root"])
    _, val_samples, test_samples = create_splits(
        samples,
        ratios=cfg["data"]["split_ratios"],
        seed=cfg["seed"],
        splits_dir=cfg["paths"]["splits_dir"],
    )

    target_samples = val_samples if args.split == "val" else test_samples

    # --- Run inference if needed ---
    if not args.skip_inference:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).is_absolute():
            from src.utils.config import PROJECT_ROOT
            checkpoint_path = str(PROJECT_ROOT / checkpoint_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = run_inference(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            samples=target_samples,
            device=device,
        )
    else:
        print_log("Skipping inference — loading existing predictions")
        # Load predictions from disk
        results = []
        pred_dir = Path(cfg["paths"]["predictions_dir"])
        from src.data.transforms import get_val_transforms, ConvertBraTSLabels
        from monai.data import Dataset
        import nibabel as nib

        val_transforms = get_val_transforms(cfg)
        from src.data.dataset import get_monai_file_list
        file_list = get_monai_file_list(target_samples)
        ds = Dataset(data=file_list, transform=val_transforms)

        for idx, data in enumerate(ds):
            pid = target_samples[idx]["patient_id"]
            pred_path = pred_dir / f"{pid}_pred.nii.gz"
            if not pred_path.exists():
                print_log(f"  Prediction not found for {pid}, skipping")
                continue

            # Load prediction and convert to multi-channel
            pred_nii = nib.load(str(pred_path))
            pred_seg = pred_nii.get_fdata().astype(np.uint8)

            # Convert integer labels back to binary channels
            pred_mc = np.zeros((3, *pred_seg.shape), dtype=np.uint8)
            pred_mc[0] = ((pred_seg == 1) | (pred_seg == 2) | (pred_seg == 4)).astype(np.uint8)
            pred_mc[1] = ((pred_seg == 1) | (pred_seg == 4)).astype(np.uint8)
            pred_mc[2] = (pred_seg == 4).astype(np.uint8)

            results.append({
                "patient_id": pid,
                "prediction": pred_mc,
                "label": data["label"].numpy(),
            })

    # --- Compute metrics ---
    print_log("Computing evaluation metrics...")

    all_case_metrics = []
    patient_ids = []

    for r in results:
        pred = r["prediction"]
        gt = r["label"]

        metrics = compute_case_metrics(pred, gt)
        all_case_metrics.append(metrics)
        patient_ids.append(r["patient_id"])

        print_log(f"  {r['patient_id']}: "
                   f"Dice WT={metrics['dice'][0]:.4f} "
                   f"TC={metrics['dice'][1]:.4f} "
                   f"ET={metrics['dice'][2]:.4f}")

    # Aggregate
    agg = aggregate_metrics(all_case_metrics)

    print_log("\n--- Aggregated Metrics ---")
    for c, name in enumerate(CLASS_NAMES):
        print_log(f"  {name}: "
                   f"Dice={agg[f'dice/{name}/mean']:.4f}±{agg[f'dice/{name}/std']:.4f}  "
                   f"HD95={agg[f'hausdorff95/{name}/mean']:.2f}±{agg[f'hausdorff95/{name}/std']:.2f}")
    print_log(f"  Mean Dice (all): {agg['dice/mean_all']:.4f}")

    # --- Generate visualizations ---
    vis_dir = Path(cfg["paths"]["visualizations_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)

    num_vis = cfg["evaluation"].get("num_visualization_cases", 5)

    for i, r in enumerate(results[:num_vis]):
        # Need the image data for visualization
        from src.data.transforms import get_val_transforms
        from src.data.dataset import get_monai_file_list
        from monai.data import Dataset

        file_list = get_monai_file_list([target_samples[i]])
        ds = Dataset(data=file_list, transform=get_val_transforms(cfg))
        data = ds[0]
        image = data["image"].numpy()

        # Overlay figure
        fig = create_overlay_figure(
            image=image,
            gt=r["label"],
            pred=r["prediction"],
            patient_id=r["patient_id"],
            save_path=str(vis_dir / f"{r['patient_id']}_overlay.png"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Multi-view figure
        fig2 = create_multi_view_figure(
            image=image,
            gt=r["label"],
            pred=r["prediction"],
            patient_id=r["patient_id"],
            save_path=str(vis_dir / f"{r['patient_id']}_multiview.png"),
        )
        plt.close(fig2)

        print_log(f"  Saved visualizations for {r['patient_id']}")

    # --- Generate report ---
    report_path = Path(cfg["paths"]["output_dir"]) / "evaluation_report.md"
    generate_report(
        aggregated_metrics=agg,
        case_metrics=all_case_metrics,
        patient_ids=patient_ids,
        visualization_dir=str(vis_dir),
        output_path=str(report_path),
    )

    print(f"\n✅ Evaluation complete.")
    print(f"  Report: {report_path}")
    print(f"  Visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
