from pathlib import Path
from typing import Any

import yaml


def load_config(filepath: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a raw dictionary."""
    with Path(filepath).open() as f:
        config = yaml.safe_load(f)
        if not isinstance(config, dict):
            msg = f"Config file {filepath} must be a dictionary"
            raise TypeError(msg)
        return config
