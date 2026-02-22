"""
MONAI transform pipelines for training and validation.

BraTS label mapping:
    Original labels: {0, 1, 2, 4}  (note: no label 3)
    Converted to 3 overlapping binary channels:
        Channel 0 — Whole Tumor (WT): labels 1 + 2 + 4
        Channel 1 — Tumor Core (TC): labels 1 + 4
        Channel 2 — Enhancing Tumor (ET): label 4
"""

from typing import List
from monai import transforms as T


class ConvertBraTSLabels(T.MapTransform):
    """
    Convert BraTS integer labels {0,1,2,4} to 3-channel binary masks:
        WT (whole tumor):      1,2,4 → 1
        TC (tumor core):       1,4   → 1
        ET (enhancing tumor):  4     → 1
    """

    def __init__(self, keys: str = "label"):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]  # shape: (1, D, H, W)

            wt = (label == 1) | (label == 2) | (label == 4)  # Whole tumor
            tc = (label == 1) | (label == 4)                  # Tumor core
            et = (label == 4)                                  # Enhancing tumor

            import torch
            d[key] = torch.cat([wt.float(), tc.float(), et.float()], dim=0)  # (3, D, H, W)

        return d


def get_train_transforms(cfg: dict) -> T.Compose:
    """Build training transform pipeline with augmentation."""

    prep = cfg["preprocessing"]
    aug = cfg["augmentation"]
    patch_size = prep["patch_size"]

    return T.Compose([
        # Load NIfTI files
        T.LoadImaged(keys=["image", "label"], image_only=True),

        # Ensure channel-first: (C, D, H, W)
        T.EnsureChannelFirstd(keys=["image", "label"]),

        # Standardize orientation to RAS
        T.Orientationd(keys=["image", "label"], axcodes=prep["orientation"]),

        # Resample to consistent spacing
        T.Spacingd(
            keys=["image", "label"],
            pixdim=prep["spacing"],
            mode=("bilinear", "nearest"),
        ),

        # Normalize intensities per-channel (nonzero voxels only)
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        # Convert labels to 3-channel binary masks
        ConvertBraTSLabels(keys="label"),

        # Crop to foreground region (remove empty space)
        T.CropForegroundd(keys=["image", "label"], source_key="image", margin=10),

        # Random patch sampling with foreground emphasis
        T.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=3,  # 3:1 foreground:background ratio
            neg=1,
            num_samples=prep.get("num_samples_per_volume", 2),
        ),

        # ----- Augmentation -----
        T.RandFlipd(keys=["image", "label"], prob=aug["random_flip_prob"], spatial_axis=0),
        T.RandFlipd(keys=["image", "label"], prob=aug["random_flip_prob"], spatial_axis=1),
        T.RandFlipd(keys=["image", "label"], prob=aug["random_flip_prob"], spatial_axis=2),

        T.RandRotate90d(keys=["image", "label"], prob=aug["random_rotate90_prob"], max_k=3),

        T.RandShiftIntensityd(
            keys="image",
            offsets=aug["intensity_shift_offset"],
            prob=0.5,
        ),

        T.RandScaleIntensityd(
            keys="image",
            factors=aug["intensity_scale_range"][1] - 1.0,
            prob=0.5,
        ),
    ])


def get_val_transforms(cfg: dict) -> T.Compose:
    """Build validation transform pipeline — NO augmentation."""

    prep = cfg["preprocessing"]

    return T.Compose([
        T.LoadImaged(keys=["image", "label"], image_only=True),
        T.EnsureChannelFirstd(keys=["image", "label"]),
        T.Orientationd(keys=["image", "label"], axcodes=prep["orientation"]),
        T.Spacingd(
            keys=["image", "label"],
            pixdim=prep["spacing"],
            mode=("bilinear", "nearest"),
        ),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ConvertBraTSLabels(keys="label"),
    ])
