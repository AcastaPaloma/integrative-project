"""
BraTS dataset discovery — scans data directory and builds file lists.

Supports BraTS 2020 naming convention:
    BraTS20_Training_XXX/BraTS20_Training_XXX_{flair,t1,t1ce,t2,seg}.nii
"""

import os
from pathlib import Path
from typing import List, Dict, Optional


def discover_brats_samples(data_root: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Scan data_root for BraTS-format patient directories.

    Returns list of dicts, each with keys: "flair", "t1", "t1ce", "t2", "seg", "patient_id"
    """
    data_root = Path(data_root)
    samples = []

    # Look for patient directories (BraTS20_Training_XXX pattern)
    patient_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and "Training" in d.name
    ])

    if not patient_dirs:
        # Check for HGG / LGG subdirectories (BraTS 2015 format)
        for subdir_name in ["HGG", "LGG"]:
            subdir = data_root / subdir_name
            if subdir.exists():
                patient_dirs.extend(sorted([
                    d for d in subdir.iterdir() if d.is_dir()
                ]))

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        sample = {"patient_id": patient_id}

        # Find NIfTI files for each modality
        nii_files = list(patient_dir.glob("*.nii")) + list(patient_dir.glob("*.nii.gz")) + list(patient_dir.glob("*.mha"))

        for modality in ["flair", "t1ce", "t1", "t2", "seg"]:
            matched = None
            for f in nii_files:
                fname = f.name.lower()
                # Match modality suffix (handle t1 vs t1ce ordering)
                if modality == "t1ce" and "t1ce" in fname:
                    matched = str(f)
                elif modality == "t1" and "t1." in fname.replace("t1ce", ""):
                    # t1 but NOT t1ce
                    matched = str(f)
                elif modality == "t2" and "t2" in fname:
                    matched = str(f)
                elif modality == "flair" and "flair" in fname:
                    matched = str(f)
                elif modality == "seg" and ("seg" in fname or "ot." in fname):
                    matched = str(f)

            if matched is None:
                print(f"[WARNING] Missing {modality} for {patient_id}")
            sample[modality] = matched

        # Only include if all modalities + seg are present
        required = ["flair", "t1", "t1ce", "t2", "seg"]
        if all(sample.get(k) is not None for k in required):
            samples.append(sample)
        else:
            missing = [k for k in required if sample.get(k) is None]
            print(f"[SKIP] {patient_id} — missing: {missing}")

    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]

    print(f"[Dataset] Found {len(samples)} valid samples in {data_root}")
    return samples


def get_monai_file_list(
    samples: List[Dict[str, str]],
    modalities: Optional[List[str]] = None,
    zero_pad_to: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Convert sample dicts to MONAI-compatible format.

    Args:
        samples: List of sample dicts from discovery
        modalities: Which modalities to include (default: all 4).
                    E.g. ["flair", "t2"] for a 2-channel model.
        zero_pad_to: If set, pad the image list to this many channels
                     using None placeholders (handled by transforms).
                     Used for cross-modality inference: train on 4ch,
                     test with subset by zeroing missing channels.

    Returns:
        List of dicts with keys: "image" (list of paths or None),
        "label" (1 path), "patient_id", "modalities"
    """
    all_modalities = ["flair", "t1", "t1ce", "t2"]
    if modalities is None:
        modalities = all_modalities

    file_list = []
    for s in samples:
        if zero_pad_to is not None:
            # Build full-length image list, None for missing modalities
            image_paths = []
            for mod in all_modalities:
                if mod in modalities:
                    image_paths.append(s[mod])
                else:
                    image_paths.append(None)  # will be zero-padded by transform
        else:
            image_paths = [s[mod] for mod in modalities]

        file_list.append({
            "image": image_paths,
            "label": s["seg"],
            "patient_id": s["patient_id"],
            "modalities": modalities,
        })
    return file_list
