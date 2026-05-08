from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


class UniversalLog(BaseModel):
    """Standardized signal vectors logged universally across all simulations."""

    t: float
    # We use list[float] or float for flexibility, depending on system dimensionality
    # For initial iteration, let's keep it simple with floats or basic types
    y: Any  # Plant output
    u: Any  # Control effort
    # Note: ref, x_hat, etc. could be added here as needed


class Logger:
    """Centralized logger handling both universal signals and component-specific logs."""

    def __init__(self) -> None:
        """Initialize the logger."""
        self.universal_logs: list[dict[str, Any]] = []
        self.component_logs: dict[str, list[dict[str, Any]]] = {}

    def log(self, universal: UniversalLog, components: dict[str, BaseModel]) -> None:
        """
        Record a snapshot of the simulation state.

        Args:
            universal: The universal log signals for this step.
            components: A dictionary mapping component names to their Pydantic log models.
        """
        # Store universal log as dictionary
        self.universal_logs.append(universal.model_dump())

        # Store component logs
        for name, log_model in components.items():
            if name not in self.component_logs:
                self.component_logs[name] = []

            # To correlate component logs with time, we explicitly add time
            log_dict = log_model.model_dump()
            log_dict["t"] = universal.t
            self.component_logs[name].append(log_dict)

    def export_csv(self, directory: str | Path, prefix: str = "sim") -> None:
        """Export accumulated logs to CSV files."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Export universal logs
        if self.universal_logs:
            df_universal = pd.DataFrame(self.universal_logs)
            df_universal.to_csv(dir_path / f"{prefix}_universal.csv", index=False)

        # Export component logs
        for name, logs in self.component_logs.items():
            if logs:
                df_comp = pd.DataFrame(logs)
                df_comp.to_csv(dir_path / f"{prefix}_comp_{name}.csv", index=False)

    def export_npz(self, directory: str | Path, prefix: str = "sim") -> None:
        """Export accumulated logs to a NumPy Archive (.npz) file."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        arrays_to_save: dict[str, np.ndarray[Any, Any]] = {}

        # Universal
        if self.universal_logs:
            # Simple conversion: keys to lists, then to arrays
            df_universal = pd.DataFrame(self.universal_logs)
            for col in df_universal.columns:
                arrays_to_save[f"universal_{col}"] = df_universal[col].to_numpy()

        # Components
        for name, logs in self.component_logs.items():
            if logs:
                df_comp = pd.DataFrame(logs)
                for col in df_comp.columns:
                    arrays_to_save[f"{name}_{col}"] = df_comp[col].to_numpy()

        if arrays_to_save:
            np.savez_compressed(dir_path / f"{prefix}_data.npz", **arrays_to_save)  # type: ignore[arg-type]
