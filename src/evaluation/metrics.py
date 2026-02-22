"""
Evaluation metrics for brain tumor segmentation.

Computes per-class and aggregate:
    - Dice coefficient
    - Hausdorff distance 95th percentile
    - Precision and Recall
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.ndimage import label as scipy_label
from scipy.spatial.distance import directed_hausdorff


CLASS_NAMES = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for a single binary channel."""
    intersection = np.sum(pred * gt)
    if np.sum(pred) + np.sum(gt) == 0:
        return 1.0  # Both empty = perfect
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt))


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Precision for a single binary channel."""
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Recall for a single binary channel."""
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    95th percentile Hausdorff distance between binary masks.
    Returns inf if either mask is empty.
    """
    pred_points = np.argwhere(pred)
    gt_points = np.argwhere(gt)

    if len(pred_points) == 0 or len(gt_points) == 0:
        if len(pred_points) == 0 and len(gt_points) == 0:
            return 0.0
        return float("inf")

    # Compute distances from pred to gt and gt to pred
    from scipy.spatial import cKDTree

    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(pred_points)

    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)

    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    hd95 = np.percentile(all_dists, 95)

    return float(hd95)


def compute_case_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, List[float]]:
    """
    Compute all metrics for a single case.

    Args:
        pred: (3, D, H, W) binary prediction
        gt: (3, D, H, W) binary ground truth

    Returns:
        Dict with metric lists per class (indexed 0=WT, 1=TC, 2=ET)
    """
    num_classes = pred.shape[0]

    metrics = {
        "dice": [],
        "hausdorff95": [],
        "precision": [],
        "recall": [],
    }

    for c in range(num_classes):
        p = pred[c].astype(np.float32)
        g = gt[c].astype(np.float32)

        metrics["dice"].append(dice_coefficient(p, g))
        metrics["hausdorff95"].append(hausdorff_distance_95(p, g))
        metrics["precision"].append(precision_score(p, g))
        metrics["recall"].append(recall_score(p, g))

    return metrics


def aggregate_metrics(all_case_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple cases.

    Returns dict with per-class mean ± std for each metric.
    """
    metrics_keys = ["dice", "hausdorff95", "precision", "recall"]
    num_classes = 3

    result = {}

    for metric in metrics_keys:
        per_class = {c: [] for c in range(num_classes)}

        for case in all_case_metrics:
            for c in range(num_classes):
                per_class[c].append(case[metric][c])

        for c in range(num_classes):
            values = np.array(per_class[c])
            key_prefix = f"{metric}/{CLASS_NAMES[c]}"
            result[f"{key_prefix}/mean"] = float(np.mean(values))
            result[f"{key_prefix}/std"] = float(np.std(values))
            result[f"{key_prefix}/values"] = values.tolist()

        # Overall mean across classes
        all_values = [per_class[c] for c in range(num_classes)]
        all_flat = np.array(all_values).flatten()
        result[f"{metric}/mean_all"] = float(np.mean(all_flat))

    return result
