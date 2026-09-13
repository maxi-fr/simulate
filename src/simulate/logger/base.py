import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

# The dtype kinds each buffer kind can hold with no silent loss, mirroring
# ``np.can_cast(src, dst, casting="same_kind")`` without the microsecond that call costs.
# Widths are deliberately not compared, so a declared float32 buffer may hold float64 values.
# A buffer kind missing here accepts only its own kind.
_ALLOWED_SRC_KINDS: dict[str, frozenset[str]] = {
    "b": frozenset("b"),
    "u": frozenset("bu"),
    "i": frozenset("bui"),
    "f": frozenset("buif"),
    "c": frozenset("buifc"),
}

# Kinds of the Python scalars a log model can hold, so the common case skips ``np.asarray``.
# A Python ``int`` maps to the unsigned kind because numpy range-checks Python integers on
# assignment (raising OverflowError), so it cannot silently corrupt an integer buffer of either
# signedness -- only a boolean one, which "u" is refused by. A numpy-typed signed integer carries
# its own "i" kind and so is still kept out of unsigned buffers, where it would wrap silently.
_PYTHON_KINDS: dict[type, str] = {bool: "b", int: "u", float: "f"}


class BaseLogger(ABC):
    """Shared logging behaviour for the RAM- and memmap-backed loggers.

    Each signal is accumulated into a pre-allocated array indexed by step. The run
    length is fixed at construction (``total_steps``): buffers are sized to exactly
    that many rows on the first :meth:`log`, and logging beyond it raises. A buffer's
    dtype is fixed just as early -- inferred from the first logged value, or taken from
    the log model field's ``dataclasses.field(metadata={"dtype": ...})`` when the first
    value does not fix it (a status string, whose width must cover every status the run
    can emit). Later values that would not survive that dtype are rejected rather than
    silently truncated; see :meth:`_check_castable`. Subclasses
    supply the storage backend by implementing :meth:`_make_buffer` and
    :meth:`_finalize`; everything else (schema inference, per-step writes, the
    signal accessors, and the :meth:`finalize` entry point) lives here.
    """

    def __init__(self, total_steps: dict[str, int] | int) -> None:
        """Initialize the logger for a run of known length.

        Parameters
        ----------
        total_steps : dict[str, int] or int
            Number of log rows per component (or a single integer acting as the default
            for all components); pre-allocated arrays are sized to exactly this and
            logging beyond it raises ``RuntimeError``.
        """
        if isinstance(total_steps, int):
            self._default_total_steps: int | None = total_steps
            self._total_steps: dict[str, int] = {}
        else:
            self._default_total_steps = None
            self._total_steps = dict(total_steps)

        self._write_indices: dict[str, int] = {}
        self._buffers_initialized: bool = False

        self._t_buffers: dict[str, np.ndarray] = {}
        self._component_buffers: dict[str, dict[str, np.ndarray]] = {}

    def time(self, component: str) -> np.ndarray:
        """Return the time axis for *component*, sliced to the rows written."""
        if component not in self._t_buffers:
            self._raise_unknown_signal(f"Unknown component '{component}'")
        write_idx = self._write_indices[component]
        return self._t_buffers[component][:write_idx]

    def signal(self, component: str, field: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (t, values) for one logged signal, sliced to the rows written."""
        if component not in self._component_buffers:
            self._raise_unknown_signal(f"Unknown component '{component}'")
        comp_bufs = self._component_buffers[component]
        if field not in comp_bufs:
            self._raise_unknown_signal(f"Unknown field '{field}' for component '{component}'")
        write_idx = self._write_indices[component]
        t = self._t_buffers[component][:write_idx]
        vals = comp_bufs[field][:write_idx]
        return t, vals

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

    def _create_buffer_array(
        self, val: ArrayLike, arcname: str, total_steps: int, dtype: DTypeLike | None = None
    ) -> np.ndarray:
        """Create a pre-allocated array of the correct shape and dtype for *val*.

        The shape always comes from *val*. So does the dtype, unless the log model's field
        declares one via ``dataclasses.field(metadata={"dtype": ...})``: that is how a signal
        whose first value does not fix its width -- a solver status string, say -- pins the
        width it needs for the whole run.
        """
        arr = np.asarray(val)
        shape = (total_steps, *arr.shape)
        return self._make_buffer(arcname, shape, np.dtype(dtype) if dtype is not None else arr.dtype)

    def _init_buffers(self, components: Mapping[str, Any]) -> None:
        """Initialize the pre-allocated buffers from the incoming data's shapes and declared types."""
        self._component_buffers = {}
        self._t_buffers = {}
        self._write_indices = {}
        self._prepare_storage()

        for name, log_model in components.items():
            fields = dataclasses.fields(log_model)
            if not fields:
                continue

            if name not in self._total_steps:
                if self._default_total_steps is not None:
                    self._total_steps[name] = self._default_total_steps
                else:
                    msg = f"No total_steps configured for component '{name}'"
                    raise ValueError(msg)

            total = self._total_steps[name]
            self._write_indices[name] = 0
            self._t_buffers[name] = self._make_buffer(f"{name}.t", (total,), np.dtype(np.float64))

            self._component_buffers[name] = {
                f.name: self._create_buffer_array(
                    getattr(log_model, f.name), f"{name}.{f.name}", total, f.metadata.get("dtype")
                )
                for f in fields
            }

        self._buffers_initialized = True

    @staticmethod
    def _check_castable(buffer: np.ndarray, value: ArrayLike, name: str, key: str) -> None:
        """Raise ``ValueError`` if writing *value* into *buffer* would silently corrupt it.

        Each buffer's dtype is fixed once, from the first logged value or the field's declared
        ``dtype`` metadata, and numpy assignment into it is lossy but silent: a string longer
        than the buffer's width is clipped, and a float written into an integer or boolean
        buffer is truncated. Both are caught here so the run fails at the offending step
        instead of finalizing a quietly corrupted archive. Narrowing within a kind
        (``float64`` into a declared ``float32``) is deliberate and stays allowed.

        This runs on every signal of every step, so it compares dtype *kinds* via
        :data:`_ALLOWED_SRC_KINDS` rather than calling ``np.can_cast``, which costs about as
        much as the buffer write it guards.
        """
        dtype = buffer.dtype
        if dtype.kind == "U":
            width = dtype.itemsize // 4  # numpy stores unicode as UCS-4, 4 bytes per character
            longest = max(len(s) for s in np.atleast_1d(value).flat)
            if longest > width:
                msg = (
                    f"Signal '{name}.{key}' is {width} characters wide but got a "
                    f"{longest}-character value ({value!r}); numpy would truncate it silently. "
                    f'Declare the width on the field: dataclasses.field(metadata={{"dtype": "<U{longest}"}}).'
                )
                raise ValueError(msg)
            return

        src_kind = _PYTHON_KINDS.get(type(value))
        if src_kind is None:  # an array, a numpy scalar, or a sequence
            src_kind = np.asarray(value).dtype.kind
        # Identical kinds are the overwhelmingly common case and always fit, so they settle it
        # with a string compare. Anything else -- including a kind outside the numeric
        # hierarchy, such as a string into a float buffer -- goes to the allow-map.
        if src_kind != dtype.kind and src_kind not in _ALLOWED_SRC_KINDS.get(dtype.kind, ()):
            msg = (
                f"Signal '{name}.{key}' has dtype {dtype} but got a '{src_kind}'-kind value "
                f"({value!r}) that it cannot hold without loss. The buffer dtype is fixed by the "
                'first logged value, or by dataclasses.field(metadata={"dtype": ...}).'
            )
            raise ValueError(msg)

    def log(self, t: float, components: Mapping[str, Any]) -> None:
        """Record a snapshot of the simulation state.

        Parameters
        ----------
        t : float
            Simulation time for this step.
        components : Mapping
            A dictionary mapping component names to their log models.

        Raises
        ------
        RuntimeError
            If more than ``total_steps`` rows have been logged.
        ValueError
            If a value cannot be stored in its buffer without silent truncation; see
            :meth:`_check_castable`.
        """
        if not self._buffers_initialized:
            self._init_buffers(components)

        # Write component signals
        for name, log_model in components.items():
            if name not in self._component_buffers:
                continue

            idx = self._write_indices[name]
            total = self._total_steps[name]
            if idx >= total:
                msg = (
                    f"Logger capacity ({total}) exceeded for component '{name}'; "
                    "construct the logger with the correct total_steps before logging."
                )
                raise RuntimeError(msg)

            self._t_buffers[name][idx] = t
            for key, buffer in self._component_buffers[name].items():
                value = getattr(log_model, key)
                self._check_castable(buffer, value, name, key)
                buffer[idx] = value

            self._write_indices[name] = idx + 1

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
        if not self._buffers_initialized or all(idx == 0 for idx in self._write_indices.values()):
            self._cleanup()
            return

        for name in self._component_buffers:
            idx = self._write_indices[name]
            total = self._total_steps[name]
            if idx != total:
                msg = (
                    f"Logger recorded {idx} of {total} expected rows for component '{name}'; "
                    "the buffers are sized for the full run, so finalizing a partial fill would "
                    "emit zero-padded trailing rows. Log exactly total_steps rows before finalizing."
                )
                raise RuntimeError(msg)

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        zip_path = dir_path / f"{prefix}.npz"

        self._finalize(zip_path, compress=compress)

    def _iter_buffers(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield ``(archive_key, buffer)`` for each component's time and signal buffers."""
        for name, t_buf in self._t_buffers.items():
            yield f"{name}.t", t_buf
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
