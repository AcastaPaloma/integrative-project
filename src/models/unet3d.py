"""
3D U-Net model via MONAI with optional gradient checkpointing.
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm


def get_model(cfg: dict) -> nn.Module:
    """
    Build a segmentation model from configuration.

    Dispatches to MONAI UNet or plain CNN3D based on cfg["model"]["architecture"].

    Args:
        cfg: Full config dict

    Returns:
        Model ready for training
    """
    model_cfg = cfg["model"]
    architecture = model_cfg.get("architecture", "UNet")

    if architecture == "CNN":
        from src.models.cnn3d import get_cnn_model
        return get_cnn_model(cfg)

    # Default: MONAI U-Net
    model = UNet(
        spatial_dims=3,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=model_cfg["channels"],
        strides=model_cfg["strides"],
        num_res_units=model_cfg["num_res_units"],
        norm=Norm.INSTANCE,
        dropout=model_cfg.get("dropout", 0.0),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 3D U-Net created")
    print(f"[Model]   Total parameters:     {total_params:,}")
    print(f"[Model]   Trainable parameters: {trainable_params:,}")
    print(f"[Model]   Channels: {model_cfg['channels']}")
    print(f"[Model]   In/Out: {model_cfg['in_channels']}/{model_cfg['out_channels']}")

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    """
    Load model weights from checkpoint.

    Returns:
        Checkpoint dict (contains epoch, metrics, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Model] Loaded checkpoint from {checkpoint_path}")
    print(f"[Model]   Epoch: {checkpoint.get('epoch', '?')}")
    print(f"[Model]   Best Dice: {checkpoint.get('best_dice', '?'):.4f}")
    return checkpoint
