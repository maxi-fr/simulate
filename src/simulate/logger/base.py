import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from simulate.component import NoLog


@dataclasses.dataclass(frozen=True)
class CoreLog:
    """Standardized signal vectors logged universally across all simulations."""

    t: float
    x: npt.NDArray[Any]
    x_hat: npt.NDArray[Any]
    u: npt.NDArray[Any]
    ref: npt.NDArray[Any]
    y_mea: npt.NDArray[Any] | None = None


class BaseLogger(ABC):
    """Shared logging behaviour for the RAM- and memmap-backed loggers.

    Each signal is accumulated into a pre-allocated array indexed by step. The run
    length is fixed at construction (``total_steps``): buffers are sized to exactly
    that many rows on the first :meth:`log`, and logging beyond it raises. Subclasses
    supply the storage backend by implementing :meth:`_make_buffer` and
    :meth:`_finalize`; everything else (schema inference, per-step writes, the
    ``*_logs`` views, and the :meth:`finalize` entry point) lives here.
    """

    def __init__(self, total_steps: int) -> None:
        """Initialize the logger for a run of known length.

        Parameters
        ----------
        total_steps : int
            Number of log rows the run will produce; the pre-allocated arrays are
            sized to exactly this and logging beyond it raises ``RuntimeError``.
        """
        self._total_steps = total_steps
        self._write_idx: int = 0
        self._buffers_initialized: bool = False

        self._core_buffers: dict[str, np.ndarray] = {}
        self._component_buffers: dict[str, dict[str, np.ndarray]] = {}
        self._component_fields: dict[str, list[str]] = {}

    @property
    def core_logs(self) -> list[dict[str, Any]]:
        """Return the core logs as a list of dictionaries (constructed on demand)."""
        logs = []
        if self._buffers_initialized and self._write_idx > 0:
            for i in range(self._write_idx):
                entry = {}
                for key, arr in self._core_buffers.items():
                    val = arr[i]
                    entry[key] = val
                logs.append(entry)
        return logs

    @property
    def component_logs(self) -> dict[str, list[dict[str, Any]]]:
        """Return the component logs as a dictionary of lists of dictionaries (constructed on demand)."""
        result = {}
        if self._buffers_initialized and self._write_idx > 0:
            for name, fields in self._component_buffers.items():
                if name not in result:
                    result[name] = []
                for i in range(self._write_idx):
                    entry = {}
                    for key, arr in fields.items():
                        val = arr[i]
                        entry[key] = val
                    result[name].append(entry)
        return result

    def _create_buffer_array(self, val: ArrayLike, arcname: str) -> np.ndarray:
        """Create a pre-allocated array of the correct shape and dtype for *val*."""
        arr = np.asarray(val)
        shape = (self._total_steps, *arr.shape)
        return self._make_buffer(arcname, shape, arr.dtype)

    def _init_buffers(self, core: CoreLog, components: Mapping[str, Any]) -> None:
        """Initialize the pre-allocated buffers based on incoming data shapes and types."""
        self._core_buffers = {}
        self._component_buffers = {}
        self._component_fields = {}
        self._prepare_storage()

        for field in dataclasses.fields(core):
            if field.name == "t":
                self._core_buffers["t"] = self._make_buffer("t", (self._total_steps,), np.dtype(np.float64))
                continue
            val = getattr(core, field.name)
            if val is not None:
                self._core_buffers[field.name] = self._create_buffer_array(val, field.name)

        for name, log_model in components.items():
            self._component_buffers[name] = {}
            self._component_buffers[name]["t"] = self._make_buffer(
                f"{name}_t", (self._total_steps,), np.dtype(np.float64)
            )

            if isinstance(log_model, NoLog):
                self._component_fields[name] = []
                continue

            fields = [
                f.name for f in dataclasses.fields(log_model) if f.name not in {"x", "y_mea", "x_hat", "u", "ref"}
            ]
            self._component_fields[name] = fields

            for key in fields:
                val = getattr(log_model, key)
                self._component_buffers[name][key] = self._create_buffer_array(val, f"{name}_{key}")

        self._buffers_initialized = True

    def log(self, core: CoreLog, components: Mapping[str, Any]) -> None:  # noqa: C901
        """
        Record a snapshot of the simulation state.

        Parameters
        ----------
        core : CoreLog
            The core log signals for this step.
        components : Mapping
            A dictionary mapping component names to their log models.
        """
        if not self._buffers_initialized:
            self._init_buffers(core, components)

        if self._write_idx >= self._total_steps:
            msg = (
                f"Logger capacity ({self._total_steps}) exceeded; "
                "construct the logger with the correct total_steps before logging."
            )
            raise RuntimeError(msg)

        # Write core signals
        for key in self._core_buffers:
            val = getattr(core, key, None)
            if val is not None:
                self._core_buffers[key][self._write_idx] = val

        # Write component signals
        for name, log_model in components.items():
            if name not in self._component_buffers:
                continue

            # Fast path for NoLog
            if isinstance(log_model, NoLog):
                if "t" in self._component_buffers[name]:
                    self._component_buffers[name]["t"][self._write_idx] = core.t
                continue

            # Write fields cached at initialization
            fields = self._component_fields.get(name, [])
            for key in fields:
                self._component_buffers[name][key][self._write_idx] = getattr(log_model, key)

            if "t" in self._component_buffers[name]:
                self._component_buffers[name]["t"][self._write_idx] = core.t

        self._write_idx += 1

    def finalize(self, directory: str | Path, prefix: str = "log", *, compress: bool = False) -> None:
        """Write all accumulated logs to ``{directory}/{prefix}.npz``.

        Dispatches to the subclass' :meth:`_finalize`. Does nothing (beyond backend
        cleanup) when no data was logged. Raises ``RuntimeError`` if the run logged
        fewer rows than ``total_steps``: the buffers are sized for the full run, so a
        partial fill would otherwise emit zero-padded trailing rows.

        Parameters
        ----------
        directory : str or Path
            Directory to write ``{prefix}.npz`` into; created if missing.
        prefix : str, optional
            Base name of the archive file.
        compress : bool, optional
            If True, deflate the archive; otherwise store it uncompressed.
        """
        if not self._buffers_initialized or self._write_idx == 0:
            self._cleanup()
            return

        if self._write_idx != self._total_steps:
            msg = (
                f"Logger recorded {self._write_idx} of {self._total_steps} expected rows; "
                "the buffers are sized for the full run, so finalizing a partial fill would "
                "emit zero-padded trailing rows. Log exactly total_steps rows before finalizing."
            )
            raise RuntimeError(msg)

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        zip_path = dir_path / f"{prefix}.npz"

        self._finalize(zip_path, compress=compress)

    def _iter_buffers(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield ``(archive_key, buffer)`` for every core and component signal."""
        yield from self._core_buffers.items()
        for name, fields in self._component_buffers.items():
            for key, arr in fields.items():
                yield f"{name}_{key}", arr

    def _prepare_storage(self) -> None:  # noqa: B027 - optional template hook; RAM backend needs no setup
        """Prepare backend storage before buffers are allocated; no-op unless overridden."""

    def _cleanup(self) -> None:  # noqa: B027 - optional template hook; RAM backend holds no resources
        """Release backend resources when finalizing an empty run; no-op unless overridden."""

    @abstractmethod
    def _make_buffer(self, arcname: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Allocate a single pre-sized buffer for signal *arcname*."""

    @abstractmethod
    def _finalize(self, zip_path: Path, *, compress: bool) -> None:
        """Pack the accumulated buffers into ``zip_path`` (only called when data exists)."""
