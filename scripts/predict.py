"""
Phase 4 — Inference

Run sliding window inference on validation/test cases and save predictions as NIfTI.

Usage:
    python scripts/predict.py --config dev --checkpoint checkpoints/best_model.pth
    python scripts/predict.py --config full --checkpoint checkpoints/best_model.pth --num_cases 3
    python scripts/predict.py --config full --checkpoint checkpoints/best_model.pth --split test
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import discover_brats_samples
from src.data.splits import create_splits
from src.inference.predict import run_inference


def main():
    parser = argparse.ArgumentParser(description="Run inference on BraTS volumes")
    parser.add_argument("--config", type=str, default="dev", choices=["default", "dev", "full"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--num_cases", type=int, default=None, help="Max cases to process")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Which split to run inference on")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    print("=" * 60)
    print("PHASE 4 — Inference")
    print("=" * 60)

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).is_absolute():
        checkpoint_path = str(Path(cfg["paths"]["checkpoint_dir"]).parent.parent / checkpoint_path)

    if not Path(checkpoint_path).exists():
        # Try project-relative path
        from src.utils.config import PROJECT_ROOT
        checkpoint_path = str(PROJECT_ROOT / args.checkpoint)

    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Split: {args.split}")

    # Discover data and load splits
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
    results = run_inference(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        samples=target_samples,
        device=device,
        num_cases=args.num_cases or len(target_samples),
    )

    print(f"\n✅ Inference complete. {len(results)} predictions saved.")


if __name__ == "__main__":
    main()
