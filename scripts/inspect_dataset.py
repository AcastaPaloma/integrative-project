"""
Phase 1 — Dataset Inspection & Validation

Validates dataset structure, checks NIfTI files, computes statistics,
and creates train/val/test split.

Usage:
    python scripts/inspect_dataset.py --config dev
    python scripts/inspect_dataset.py --config full
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import nibabel as nib
from collections import Counter

from src.utils.config import load_config
from src.data.dataset import discover_brats_samples
from src.data.splits import create_splits


def main():
    parser = argparse.ArgumentParser(description="Inspect BraTS dataset")
    parser.add_argument("--config", type=str, default="dev", choices=["default", "dev", "full"])
    parser.add_argument("--force-split", action="store_true", help="Recreate splits even if they exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = cfg["paths"]["data_root"]

    print("=" * 60)
    print("PHASE 1 — Dataset Inspection & Validation")
    print("=" * 60)
    print(f"Data root: {data_root}\n")

    # --- Step 1: Discover samples ---
    max_samples = cfg["data"].get("max_samples")
    samples = discover_brats_samples(data_root, max_samples=None)  # Discover ALL for stats

    if not samples:
        print("[ERROR] No valid samples found. Check data_root path and file naming.")
        sys.exit(1)

    # --- Step 2: Validate NIfTI format ---
    print("\n--- File Format Validation ---")
    sample = samples[0]

    for modality in ["flair", "t1", "t1ce", "t2", "seg"]:
        path = sample[modality]
        try:
            nii = nib.load(path)
            shape = nii.shape
            dtype = nii.get_data_dtype()
            spacing = nii.header.get_zooms()
            print(f"  {modality:5s}: shape={shape}, dtype={dtype}, spacing={spacing}")
        except Exception as e:
            print(f"  {modality:5s}: ERROR — {e}")

    # --- Step 3: Volume statistics ---
    print("\n--- Volume Statistics ---")
    shapes = []
    spacings = []

    for s in samples:
        nii = nib.load(s["flair"])
        shapes.append(nii.shape)
        spacings.append(nii.header.get_zooms())

    shapes = np.array(shapes)
    spacings = np.array(spacings)

    print(f"  Volumes: {len(samples)}")
    print(f"  Shape range: min={shapes.min(axis=0)}, max={shapes.max(axis=0)}")
    print(f"  Most common shape: {Counter(map(tuple, shapes)).most_common(1)[0]}")
    print(f"  Spacing range: min={spacings.min(axis=0).round(3)}, max={spacings.max(axis=0).round(3)}")

    # --- Step 4: Label analysis ---
    print("\n--- Label Analysis ---")
    total_voxels = Counter()

    for i, s in enumerate(samples):
        seg = nib.load(s["seg"]).get_fdata()
        unique, counts = np.unique(seg.astype(int), return_counts=True)
        for u, c in zip(unique, counts):
            total_voxels[int(u)] += c

        if i == 0:
            print(f"  Sample '{s['patient_id']}' unique labels: {sorted(unique.astype(int))}")

    grand_total = sum(total_voxels.values())
    print(f"\n  Class voxel distribution across {len(samples)} volumes:")

    label_names = {0: "Background", 1: "NCR/NET", 2: "Edema", 4: "Enhancing"}
    for label in sorted(total_voxels.keys()):
        count = total_voxels[label]
        pct = 100.0 * count / grand_total
        name = label_names.get(label, f"Unknown({label})")
        print(f"    Label {label} ({name:12s}): {count:>15,} voxels ({pct:6.2f}%)")

    # --- Step 5: Create splits ---
    print("\n--- Train/Val/Test Split ---")
    train_samples, val_samples, test_samples = create_splits(
        samples,
        ratios=cfg["data"]["split_ratios"],
        seed=cfg["seed"],
        splits_dir=cfg["paths"]["splits_dir"],
        force=args.force_split,
    )

    print(f"  Train: {len(train_samples)} patients")
    print(f"    IDs: {[s['patient_id'] for s in train_samples]}")
    print(f"  Val:   {len(val_samples)} patients")
    print(f"    IDs: {[s['patient_id'] for s in val_samples]}")
    print(f"  Test:  {len(test_samples)} patients")
    print(f"    IDs: {[s['patient_id'] for s in test_samples]}")

    # --- Step 6: Save report ---
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_report.md"

    with open(report_path, "w") as f:
        f.write("# BraTS Dataset Report\n\n")
        f.write(f"**Samples:** {len(samples)}\n\n")
        f.write(f"**Volume shape (first sample):** {shapes[0].tolist()}\n\n")
        f.write("## Label Distribution\n\n")
        f.write("| Label | Name | Voxels | Percentage |\n")
        f.write("|-------|------|--------|------------|\n")
        for label in sorted(total_voxels.keys()):
            count = total_voxels[label]
            pct = 100.0 * count / grand_total
            name = label_names.get(label, f"Unknown({label})")
            f.write(f"| {label} | {name} | {count:,} | {pct:.2f}% |\n")
        f.write(f"\n## Split\n\n")
        f.write(f"- Train: {len(train_samples)}\n")
        f.write(f"- Val: {len(val_samples)}\n")
        f.write(f"- Test: {len(test_samples)}\n")

    print(f"\n[Report] Saved dataset report → {report_path}")
    print("\n✅ Phase 1 complete.")


if __name__ == "__main__":
    main()
