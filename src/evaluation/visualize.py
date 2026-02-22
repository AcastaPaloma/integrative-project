"""
Visualization utilities for qualitative evaluation.

Generates:
    - MRI + ground truth overlay
    - MRI + prediction overlay
    - Difference heatmap (FP/FN)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Optional


# Color scheme for tumor regions
TUMOR_COLORS = {
    "WT": [0.2, 0.8, 0.2, 0.5],   # Green
    "TC": [0.2, 0.2, 0.9, 0.5],   # Blue
    "ET": [0.9, 0.2, 0.2, 0.5],   # Red
}


def create_overlay_figure(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    patient_id: str,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a 3-panel overlay figure (axial slice):
        1. MRI + Ground Truth
        2. MRI + Prediction
        3. Difference heatmap (green=FN, red=FP)

    Args:
        image: (C, D, H, W) — multi-modal MRI, uses first channel (FLAIR)
        gt: (3, D, H, W) — binary ground truth masks (WT, TC, ET)
        pred: (3, D, H, W) — binary prediction masks
        patient_id: For labeling
        slice_idx: Axial slice to visualize (None = auto-select center of tumor)
        save_path: Optional path to save PNG

    Returns:
        matplotlib Figure
    """
    # Use FLAIR channel for background
    mri = image[0]  # (D, H, W)

    # Auto-select slice with most tumor voxels
    if slice_idx is None:
        tumor_sum = gt[0].sum(axis=(1, 2))  # WT per slice
        if tumor_sum.max() > 0:
            slice_idx = int(np.argmax(tumor_sum))
        else:
            slice_idx = mri.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: MRI + Ground Truth ---
    axes[0].imshow(mri[slice_idx], cmap="gray", interpolation="none")
    _overlay_masks(axes[0], gt[:, slice_idx], alpha=0.45)
    axes[0].set_title("Ground Truth", fontsize=12)
    axes[0].axis("off")

    # --- Panel 2: MRI + Prediction ---
    axes[1].imshow(mri[slice_idx], cmap="gray", interpolation="none")
    _overlay_masks(axes[1], pred[:, slice_idx], alpha=0.45)
    axes[1].set_title("Prediction", fontsize=12)
    axes[1].axis("off")

    # --- Panel 3: Difference heatmap ---
    axes[2].imshow(mri[slice_idx], cmap="gray", interpolation="none")
    _overlay_diff(axes[2], gt[0, slice_idx], pred[0, slice_idx])
    axes[2].set_title("Difference (WT)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(f"{patient_id} — Axial Slice {slice_idx}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_multi_view_figure(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    patient_id: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a 3×3 figure with axial, coronal, and sagittal views.
    Rows: axial, coronal, sagittal
    Cols: Ground Truth, Prediction, Difference
    """
    mri = image[0]  # FLAIR

    # Find center of tumor mass
    tumor_mask = gt[0]  # WT
    if tumor_mask.sum() > 0:
        coords = np.argwhere(tumor_mask)
        center = coords.mean(axis=0).astype(int)
    else:
        center = np.array(mri.shape) // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    view_names = ["Axial", "Coronal", "Sagittal"]
    col_names = ["Ground Truth", "Prediction", "Difference (WT)"]

    for row, (view, idx) in enumerate(zip(view_names, center)):
        if view == "Axial":
            mri_slice = mri[idx]
            gt_slices = gt[:, idx]
            pred_slices = pred[:, idx]
        elif view == "Coronal":
            mri_slice = mri[:, idx]
            gt_slices = gt[:, :, idx]
            pred_slices = pred[:, :, idx]
        else:  # Sagittal
            mri_slice = mri[:, :, idx]
            gt_slices = gt[:, :, :, idx]
            pred_slices = pred[:, :, :, idx]

        # GT overlay
        axes[row, 0].imshow(mri_slice, cmap="gray", interpolation="none")
        _overlay_masks(axes[row, 0], gt_slices, alpha=0.45)

        # Pred overlay
        axes[row, 1].imshow(mri_slice, cmap="gray", interpolation="none")
        _overlay_masks(axes[row, 1], pred_slices, alpha=0.45)

        # Diff
        axes[row, 2].imshow(mri_slice, cmap="gray", interpolation="none")
        _overlay_diff(axes[row, 2], gt_slices[0], pred_slices[0])

        axes[row, 0].set_ylabel(view, fontsize=12, fontweight="bold")

        for col in range(3):
            axes[row, col].axis("off")

    for col, name in enumerate(col_names):
        axes[0, col].set_title(name, fontsize=12)

    plt.suptitle(f"{patient_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _overlay_masks(ax, masks: np.ndarray, alpha: float = 0.4):
    """Overlay 3-channel binary masks with tumor-region colors."""
    h, w = masks.shape[1], masks.shape[2] if len(masks.shape) > 2 else masks.shape[1]

    colors = [TUMOR_COLORS["WT"], TUMOR_COLORS["TC"], TUMOR_COLORS["ET"]]

    for c, color in enumerate(colors):
        mask_slice = masks[c] if len(masks.shape) > 1 else masks
        overlay = np.zeros((*mask_slice.shape, 4))
        overlay[mask_slice > 0] = color
        ax.imshow(overlay, interpolation="none")


def _overlay_diff(ax, gt_slice: np.ndarray, pred_slice: np.ndarray):
    """Overlay false positives (red) and false negatives (green) on the image."""
    fp = (pred_slice > 0) & (gt_slice == 0)  # False positive
    fn = (gt_slice > 0) & (pred_slice == 0)  # False negative
    tp = (pred_slice > 0) & (gt_slice > 0)   # True positive

    overlay = np.zeros((*gt_slice.shape, 4))
    overlay[tp] = [0.0, 0.5, 1.0, 0.4]   # Blue — correct
    overlay[fp] = [1.0, 0.0, 0.0, 0.6]   # Red — false positive
    overlay[fn] = [0.0, 1.0, 0.0, 0.6]   # Green — false negative

    ax.imshow(overlay, interpolation="none")
