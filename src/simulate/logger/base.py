import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike


class BaseLogger(ABC):
    """Shared logging behaviour for the RAM- and memmap-backed loggers.

    Each signal is accumulated into a pre-allocated array indexed by step. The run
    length is fixed at construction (``total_steps``): buffers are sized to exactly
    that many rows on the first :meth:`log`, and logging beyond it raises. Subclasses
    supply the storage backend by implementing :meth:`_make_buffer` and
    :meth:`_finalize`; everything else (schema inference, per-step writes, the
    signal accessors, and the :meth:`finalize` entry point) lives here.
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

        self._t_buffer: np.ndarray = np.empty(0, dtype=np.float64)
        self._component_buffers: dict[str, dict[str, np.ndarray]] = {}

    @property
    def t(self) -> np.ndarray:
        """Return the run's time axis sliced to the rows written."""
        return self._t_buffer[: self._write_idx]

    def signal(self, component: str, field: str) -> np.ndarray:
        """Return a zero-copy view of one logged signal, sliced to the rows written."""
        if component not in self._component_buffers:
            self._raise_unknown_signal(f"Unknown component '{component}'")
        comp_bufs = self._component_buffers[component]
        if field not in comp_bufs:
            self._raise_unknown_signal(f"Unknown field '{field}' for component '{component}'")
        return comp_bufs[field][: self._write_idx]

    def _raise_unknown_signal(self, what: str) -> NoReturn:
        """Raise a ``KeyError`` for *what*, listing the signals that are available instead."""
        available = self.signals()
        avail_str = ", ".join(f"'{c}.{f}'" for c, f in available) if available else "none"
        msg = f"{what}. Available signals: {avail_str}"
        raise KeyError(msg)

    def signals(self) -> list[tuple[str, str]]:
        """List the (component, field) pairs available."""
        result = []
        for name, fields in self._component_buffers.items():
            result.extend((name, field) for field in fields)
        return result

    def _create_buffer_array(self, val: ArrayLike, arcname: str) -> np.ndarray:
        """Create a pre-allocated array of the correct shape and dtype for *val*."""
        arr = np.asarray(val)
        shape = (self._total_steps, *arr.shape)
        return self._make_buffer(arcname, shape, arr.dtype)

    def _init_buffers(self, components: Mapping[str, Any]) -> None:
        """Initialize the pre-allocated buffers based on incoming data shapes and types."""
        self._component_buffers = {}
        self._prepare_storage()

        self._t_buffer = self._make_buffer("t", (self._total_steps,), np.dtype(np.float64))

        for name, log_model in components.items():
            fields = [f.name for f in dataclasses.fields(log_model)]
            if not fields:
                continue

            self._component_buffers[name] = {
                key: self._create_buffer_array(getattr(log_model, key), f"{name}.{key}") for key in fields
            }

        self._buffers_initialized = True

    def log(self, t: float, components: Mapping[str, Any]) -> None:
        """Record a snapshot of the simulation state.

        Parameters
        ----------
        t : float
            Simulation time for this step.
        components : Mapping
            A dictionary mapping component names to their log models.
        """
        if not self._buffers_initialized:
            self._init_buffers(components)

        if self._write_idx >= self._total_steps:
            msg = (
                f"Logger capacity ({self._total_steps}) exceeded; "
                "construct the logger with the correct total_steps before logging."
            )
            raise RuntimeError(msg)

        self._t_buffer[self._write_idx] = t

        # Write component signals
        for name, log_model in components.items():
            if name not in self._component_buffers:
                continue

            for key, buffer in self._component_buffers[name].items():
                buffer[self._write_idx] = getattr(log_model, key)

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
        """Yield ``(archive_key, buffer)`` for global time and every component signal."""
        yield "t", self._t_buffer
        for name, fields in self._component_buffers.items():
            for key, arr in fields.items():
                yield f"{name}.{key}", arr

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
