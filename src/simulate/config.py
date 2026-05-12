import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries.

    Nested dictionaries are merged cumulatively.
    Lists and other types are replaced entirely by the override.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(filepath: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a raw dictionary."""
    with Path(filepath).open() as f:
        config = yaml.safe_load(f)
        if not isinstance(config, dict):
            msg = f"Config file {filepath} must be a dictionary"
            raise TypeError(msg)
        return config
