from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator


class UniversalLog(BaseModel):
    """Standardized signal vectors logged universally across all simulations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t: float
    x: float | npt.NDArray[np.float64]
    y: float | npt.NDArray[np.float64]
    y_mea: float | npt.NDArray[np.float64]
    x_hat: float | npt.NDArray[np.float64]
    u: float | npt.NDArray[np.float64]
    ref: float | npt.NDArray[np.float64]

    @field_validator("x", "y", "y_mea", "x_hat", "u", "ref", mode="after")
    @classmethod
    def validate_1d(cls, v: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Validate that the signal is a float or a 1D array."""
        if isinstance(v, int | float | np.floating | np.integer):
            return float(v)
        if v.ndim != 1:
            msg = f"Array must be 1D, but has shape {v.shape}"
            raise ValueError(msg)
        return v


class Logger:
    """Centralized logger handling both universal signals and component-specific logs."""

    def __init__(self) -> None:
        """Initialize the logger."""
        self.universal_logs: list[dict[str, Any]] = []
        self.component_logs: dict[str, list[dict[str, Any]]] = {}
        self._chunk_idx: int = 0

    def log(self, universal: UniversalLog, components: Mapping[str, BaseModel]) -> None:
        """
        Record a snapshot of the simulation state.

        Args:
            universal: The universal log signals for this step.
            components: A dictionary mapping component names to their Pydantic log models.
        """
        self.universal_logs.append(universal.model_dump())

        for name, log_model in components.items():
            if name not in self.component_logs:
                self.component_logs[name] = []

            # To correlate component logs with time, we explicitly add time
            log_dict = log_model.model_dump()
            log_dict["t"] = universal.t
            self.component_logs[name].append(log_dict)

    def flush_chunk(self, directory: str | Path, prefix: str = "sim") -> None:
        """
        Write current in-memory buffers to a numbered chunk file, then clear them.

        No file or directory is created when both buffers are empty.
        Output file: {prefix}_chunk_{_chunk_idx:04d}.npz
        """
        if not self.universal_logs and not self.component_logs:
            return

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        arrays_to_save: dict[str, np.ndarray] = {}

        if self.universal_logs:
            for key in self.universal_logs[0]:
                arrays_to_save[f"universal_{key}"] = np.array([entry[key] for entry in self.universal_logs])

        for name, logs in self.component_logs.items():
            if logs:
                for key in logs[0]:
                    arrays_to_save[f"{name}_{key}"] = np.array([entry[key] for entry in logs])

        if arrays_to_save:
            np.savez_compressed(
                dir_path / f"{prefix}_chunk_{self._chunk_idx:04d}.npz",
                **arrays_to_save,  # type: ignore[arg-type]
            )

        self._chunk_idx += 1
        self.universal_logs.clear()
        self.component_logs.clear()

    @staticmethod
    def merge_chunks(directory: str | Path, prefix: str = "sim") -> None:
        """Concatenate all chunk files for *prefix* into a single {prefix}.npz file.

        No-op when no chunk files are found. Individual chunk files are deleted
        after a successful merge when *delete_chunks* is True (default).
        """
        dir_path = Path(directory)
        chunk_files = sorted(dir_path.glob(f"{prefix}_chunk_*.npz"))
        if not chunk_files:
            return

        combined: dict[str, list[np.ndarray]] = {}
        for chunk_file in chunk_files:
            with np.load(chunk_file) as data:
                for key in data.files:
                    combined.setdefault(key, []).append(data[key].copy())

        merged = {key: np.concatenate(arrays, axis=0) for key, arrays in combined.items()}
        np.savez_compressed(dir_path / f"{prefix}.npz", **merged)  # type: ignore[arg-type]

        for chunk_file in chunk_files:
            chunk_file.unlink()
