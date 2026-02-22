"""
Loss functions for brain tumor segmentation.
"""

from monai.losses import DiceCELoss as MonaiDiceCELoss


def get_loss_function(cfg: dict):
    """
    Build loss function from config.

    Uses MONAI's DiceCELoss with sigmoid activation for multi-label segmentation.
    """
    loss_cfg = cfg["training"]["loss"]

    loss_fn = MonaiDiceCELoss(
        to_onehot_y=False,     # Labels already converted to multi-channel
        sigmoid=True,           # Multi-label (not mutually exclusive)
        lambda_dice=loss_cfg.get("lambda_dice", 1.0),
        lambda_ce=loss_cfg.get("lambda_ce", 1.0),
    )

    print(f"[Loss] DiceCE loss (lambda_dice={loss_cfg.get('lambda_dice', 1.0)}, "
          f"lambda_ce={loss_cfg.get('lambda_ce', 1.0)})")

    return loss_fn
