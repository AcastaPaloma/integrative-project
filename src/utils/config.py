"""
YAML configuration loader with hierarchical merge support.

Usage:
    cfg = load_config("dev")   # merges default.yaml + dev.yaml
    cfg = load_config("full")  # merges default.yaml + full.yaml
"""

import os
import copy
import yaml
import argparse
from pathlib import Path


# Project root is two levels up from this file (src/utils/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_name: str = "default") -> dict:
    """
    Load configuration by merging default.yaml with the named config.

    Args:
        config_name: One of "default", "dev", "full"

    Returns:
        Merged configuration dictionary.
    """
    # Always load default as base
    default_path = CONFIGS_DIR / "default.yaml"
    with open(default_path, "r") as f:
        config = yaml.safe_load(f)

    # If a specific config is requested, merge on top
    if config_name != "default":
        override_path = CONFIGS_DIR / f"{config_name}.yaml"
        if not override_path.exists():
            raise FileNotFoundError(f"Config file not found: {override_path}")
        with open(override_path, "r") as f:
            overrides = yaml.safe_load(f)
        if overrides:
            config = deep_merge(config, overrides)

    # Resolve paths relative to project root
    for key, val in config.get("paths", {}).items():
        config["paths"][key] = str(PROJECT_ROOT / val)

    return config


def get_config_from_args() -> dict:
    """Parse --config CLI argument and return loaded config."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="dev",
                        choices=["default", "dev", "full"],
                        help="Config profile to use")
    args, _ = parser.parse_known_args()
    return load_config(args.config)


class ConfigAccessor:
    """Dot-access wrapper for config dictionary."""

    def __init__(self, d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigAccessor(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigAccessor):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
