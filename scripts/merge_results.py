"""
Merge results from multiple devices into a unified results directory.

When training is split across GTX 1080, RTX 4060, and Colab,
each device produces experiment results in outputs/results/<experiment_name>/.
This script merges them into a single directory.

Usage:
    python scripts/merge_results.py --sources path1 path2 path3 --target outputs/results
    python scripts/merge_results.py --sources /mnt/usb/results ./outputs/results --target outputs/results_merged
"""

import sys
import json
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import print_log


def main():
    parser = argparse.ArgumentParser(description="Merge results from multiple devices")
    parser.add_argument("--sources", type=str, nargs="+", required=True,
                        help="Source result directories to merge")
    parser.add_argument("--target", type=str, required=True,
                        help="Target merged results directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing experiments in target")
    args = parser.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MERGE RESULTS")
    print(f"  Sources: {args.sources}")
    print(f"  Target: {target}")
    print("=" * 60)

    merged_count = 0
    skipped_count = 0

    for source_path in args.sources:
        source = Path(source_path)
        if not source.exists():
            print_log(f"  WARNING: Source not found: {source}")
            continue

        print_log(f"\nProcessing: {source}")

        for exp_dir in sorted(source.iterdir()):
            if not exp_dir.is_dir():
                continue

            target_exp = target / exp_dir.name

            if target_exp.exists() and not args.overwrite:
                print_log(f"  SKIP: {exp_dir.name} (already exists, use --overwrite)")
                skipped_count += 1
                continue

            # Copy experiment directory
            if target_exp.exists():
                shutil.rmtree(target_exp)
            shutil.copytree(exp_dir, target_exp)
            merged_count += 1
            print_log(f"  MERGED: {exp_dir.name}")

            # Validate checkpoint
            best_ckpt = target_exp / "best_model.pth"
            last_ckpt = target_exp / "last_model.pth"
            history = target_exp / "training_history.json"

            status = []
            if best_ckpt.exists():
                status.append(f"best_ckpt={best_ckpt.stat().st_size / 1024**2:.1f}MB")
            if last_ckpt.exists():
                status.append(f"last_ckpt={last_ckpt.stat().st_size / 1024**2:.1f}MB")
            if history.exists():
                with open(history) as f:
                    h = json.load(f)
                epochs = len(h.get("train_loss", []))
                status.append(f"epochs={epochs}")

            if status:
                print_log(f"    {', '.join(status)}")

    print(f"\n✅ Merge complete. {merged_count} merged, {skipped_count} skipped.")
    print(f"  Results directory: {target}")


if __name__ == "__main__":
    main()
