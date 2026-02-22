"""
Sliding window inference for full 3D volumes.
Loads a checkpoint, runs inference, and saves predictions as NIfTI.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
import nibabel as nib
from monai.inferers import SlidingWindowInferer
from monai.transforms import Activations, AsDiscrete, Compose
from monai.data import Dataset, DataLoader

from src.data.transforms import get_val_transforms
from src.models.unet3d import get_model, load_checkpoint
from src.utils.logging import print_log


def run_inference(
    cfg: dict,
    checkpoint_path: str,
    samples: List[Dict[str, str]],
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_cases: Optional[int] = None,
) -> List[Dict]:
    """
    Run sliding window inference on a list of samples.

    Args:
        cfg: Full config dict
        checkpoint_path: Path to model checkpoint
        samples: List of sample dicts from dataset
        output_dir: Where to save NIfTI predictions (None = use cfg default)
        device: Torch device
        num_cases: Max number of cases to process

    Returns:
        List of dicts with "patient_id", "prediction_path", "prediction" (numpy)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        output_dir = cfg["paths"]["predictions_dir"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit number of cases
    if num_cases is not None:
        samples = samples[:num_cases]

    # Build model and load checkpoint
    model = get_model(cfg)
    load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    # Transforms (validation — no augmentation)
    val_transforms = get_val_transforms(cfg)

    # Build file list for MONAI
    from src.data.dataset import get_monai_file_list
    file_list = get_monai_file_list(samples)

    # Dataset (no caching for inference)
    dataset = Dataset(data=file_list, transform=val_transforms)

    # Sliding window inferer
    inf_cfg = cfg["inference"]
    patch_size = cfg["preprocessing"]["patch_size"]
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=inf_cfg.get("sw_batch_size", 2),
        overlap=inf_cfg.get("overlap", 0.5),
        mode=inf_cfg.get("mode", "gaussian"),
    )

    # Post-processing
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    results = []

    for idx, data in enumerate(dataset):
        patient_id = samples[idx]["patient_id"]
        print_log(f"Inference [{idx+1}/{len(dataset)}]: {patient_id}")

        image = data["image"].unsqueeze(0).to(device)  # (1, C, D, H, W)
        label = data["label"]  # (3, D, H, W) — ground truth

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg["training"].get("mixed_precision", True)):
            output = inferer(image, model)  # (1, 3, D, H, W)

        # Post-process
        pred = post_pred(output[0])  # (3, D, H, W)
        pred_np = pred.cpu().numpy().astype(np.uint8)

        # Convert back to BraTS integer labels for NIfTI saving
        # WT=ch0, TC=ch1, ET=ch2
        # Reconstruct: ET(4) where ch2=1, NCR/NET(1) where ch1=1 and ch2=0,
        #              ED(2) where ch0=1 and ch1=0, Background(0) elsewhere
        seg_map = np.zeros(pred_np.shape[1:], dtype=np.uint8)  # (D, H, W)
        seg_map[pred_np[0] == 1] = 2   # Whole tumor → edema (label 2)
        seg_map[pred_np[1] == 1] = 1   # Tumor core → NCR/NET (label 1)
        seg_map[pred_np[2] == 1] = 4   # Enhancing → ET (label 4)

        # Save as NIfTI
        # Use the original image's affine for spatial consistency
        ref_nii = nib.load(samples[idx]["flair"])
        pred_nii = nib.Nifti1Image(seg_map, affine=ref_nii.affine, header=ref_nii.header)
        save_path = output_dir / f"{patient_id}_pred.nii.gz"
        nib.save(pred_nii, str(save_path))

        print_log(f"  Saved prediction → {save_path}")

        results.append({
            "patient_id": patient_id,
            "prediction_path": str(save_path),
            "prediction": pred_np,
            "label": label.cpu().numpy(),
        })

    print_log(f"Inference complete. {len(results)} predictions saved to {output_dir}")
    return results
