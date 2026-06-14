from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from .sensor import MeasurementModel


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries.

    Nested dictionaries are merged cumulatively.
    Lists and other types are replaced entirely by the override.

    Returns
    -------
    dict
        A new dictionary with ``override`` merged into ``base``.
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


def build_component(comp_config: dict[str, Any]) -> Any:  # noqa: ANN401
    """Instantiate a component from a ``{class_path, ...}`` dict via its ``from_config``."""
    cfg = comp_config.copy()
    class_path: str = cfg.pop("class_path")
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name).from_config(cfg)


def build_measurement(spec: dict[str, Any]) -> MeasurementModel:
    """Resolve a measurement model from a ``{class_path, ...}`` dict.

    ``class_path`` may point to a class (instantiated with the remaining keys as keyword
    arguments) or to a bare function (used as-is; no extra keys are then allowed).

    Returns
    -------
    MeasurementModel
        The instantiated measurement class or the resolved measurement function.
    """
    cfg = spec.copy()
    class_path: str = cfg.pop("class_path")
    module_name, attr_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)
    if isinstance(target, type):
        return target(**cfg)
    if cfg:
        msg = f"Measurement function {class_path} takes no parameters, got {sorted(cfg)}"
        raise ValueError(msg)
    return target
