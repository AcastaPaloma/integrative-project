"""
Deterministic train/val/test splitting with JSON persistence.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.model_selection import train_test_split


def create_splits(
    samples: List[Dict[str, str]],
    ratios: List[float],
    seed: int = 42,
    splits_dir: str = "data/splits",
    force: bool = False,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create or load deterministic train/val/test split.

    Args:
        samples: List of sample dicts from dataset discovery
        ratios: [train, val, test] ratios that sum to 1.0
        seed: Random seed for reproducibility
        splits_dir: Directory to save/load split JSON
        force: If True, recreate splits even if JSON exists

    Returns:
        (train_samples, val_samples, test_samples)
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_file = splits_dir / "split.json"

    if split_file.exists() and not force:
        print(f"[Splits] Loading existing split from {split_file}")
        return _load_splits(split_file, samples)

    # Create new split
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"
    train_ratio, val_ratio, test_ratio = ratios

    patient_ids = [s["patient_id"] for s in samples]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_ids, val_test_ids = train_test_split(
        patient_ids, test_size=val_test_ratio, random_state=seed
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_ids, test_ids = train_test_split(
        val_test_ids, test_size=relative_test_ratio, random_state=seed
    )

    # Save split
    split_data = {
        "seed": seed,
        "ratios": ratios,
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"[Splits] Created split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"[Splits] Saved to {split_file}")

    # Build sample lists
    id_to_sample = {s["patient_id"]: s for s in samples}
    train_samples = [id_to_sample[pid] for pid in train_ids]
    val_samples = [id_to_sample[pid] for pid in val_ids]
    test_samples = [id_to_sample[pid] for pid in test_ids]

    return train_samples, val_samples, test_samples


def _load_splits(
    split_file: Path,
    samples: List[Dict[str, str]]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load existing splits from JSON and map back to sample dicts."""

    with open(split_file, "r") as f:
        split_data = json.load(f)

    id_to_sample = {s["patient_id"]: s for s in samples}

    train_ids = split_data["train"]
    val_ids = split_data["val"]
    test_ids = split_data["test"]

    # Filter to only include IDs that exist in current samples
    train_samples = [id_to_sample[pid] for pid in train_ids if pid in id_to_sample]
    val_samples = [id_to_sample[pid] for pid in val_ids if pid in id_to_sample]
    test_samples = [id_to_sample[pid] for pid in test_ids if pid in id_to_sample]

    # If filtering left val or test empty (e.g. dev mode with few samples),
    # re-split the available samples so every set gets at least 1 sample.
    if len(samples) >= 3 and (len(val_samples) == 0 or len(test_samples) == 0):
        print("[Splits] Subset too small for saved split — re-splitting available samples")
        ratios = split_data.get("ratios", [0.7, 0.15, 0.15])
        seed = split_data.get("seed", 42)
        patient_ids = sorted(id_to_sample.keys())

        val_test_ratio = ratios[1] + ratios[2]
        from sklearn.model_selection import train_test_split

        train_ids_new, val_test_ids = train_test_split(
            patient_ids, test_size=max(val_test_ratio, 2 / len(patient_ids)), random_state=seed
        )
        if len(val_test_ids) >= 2:
            relative_test = ratios[2] / (ratios[1] + ratios[2])
            val_ids_new, test_ids_new = train_test_split(
                val_test_ids, test_size=max(relative_test, 1 / len(val_test_ids)), random_state=seed
            )
        else:
            val_ids_new, test_ids_new = val_test_ids, []

        train_samples = [id_to_sample[pid] for pid in train_ids_new]
        val_samples = [id_to_sample[pid] for pid in val_ids_new]
        test_samples = [id_to_sample[pid] for pid in test_ids_new]

    print(f"[Splits] Loaded split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    return train_samples, val_samples, test_samples
