"""
Logging utilities — WandB integration + console logging.
"""

import os
from pathlib import Path
from typing import Optional

try:
    import wandb
except ImportError:
    wandb = None


_run = None


def init_wandb(config: dict, run_name: Optional[str] = None) -> None:
    """Initialize WandB logging."""
    global _run

    log_cfg = config.get("logging", {})

    if not log_cfg.get("use_wandb", False) or wandb is None:
        print("[Logging] WandB disabled")
        return

    mode = log_cfg.get("wandb_mode", "online")

    _run = wandb.init(
        project=config.get("project_name", "brain-tumor-seg"),
        name=run_name,
        config=config,
        mode=mode,
        reinit=True,
    )

    print(f"[Logging] WandB initialized (mode={mode})")


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log scalar metrics to WandB and console."""
    if _run is not None:
        _run.log(metrics, step=step)


def log_image(key: str, image, caption: str = "", step: Optional[int] = None) -> None:
    """Log an image to WandB."""
    if _run is not None and wandb is not None:
        _run.log({key: wandb.Image(image, caption=caption)}, step=step)


def log_table(key: str, columns: list, data: list) -> None:
    """Log a table to WandB."""
    if _run is not None and wandb is not None:
        table = wandb.Table(columns=columns, data=data)
        _run.log({key: table})


def finish_wandb() -> None:
    """Finalize WandB run."""
    global _run
    if _run is not None:
        _run.finish()
        _run = None
        print("[Logging] WandB run finished")


def print_log(msg: str, level: str = "INFO") -> None:
    """Simple console logging with level prefix."""
    print(f"[{level}] {msg}")
