import contextlib
import dataclasses
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.lib.format import open_memmap
from numpy.typing import ArrayLike

from .component import NoLog


@dataclasses.dataclass(frozen=True)
class CoreLog:
    """Standardized signal vectors logged universally across all simulations."""

    t: float
    x: npt.NDArray[Any]
    x_hat: npt.NDArray[Any]
    u: npt.NDArray[Any]
    ref: npt.NDArray[Any]
    y_mea: npt.NDArray[Any] | None = None


class Logger:
    """Centralized logger handling both core signals and component-specific logs."""

    def __init__(self, *, compress: bool = False) -> None:
        """Initialize the logger."""
        self.compress = compress
        self._chunk_idx: int = 0
        self._buffer_size: int = 10_000
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

    def set_buffer_size(self, size: int) -> None:
        """Set or update the size of pre-allocated buffers."""
        if not self._buffers_initialized:
            self._buffer_size = size
        elif size != self._buffer_size:
            self._resize_buffers(size)

    def _resize_buffers(self, new_size: int) -> None:
        """Resize all pre-allocated buffers to a new size."""
        for key, arr in self._core_buffers.items():
            new_shape = (new_size, *arr.shape[1:])
            new_arr = np.zeros(new_shape, dtype=arr.dtype)
            new_arr[: self._buffer_size] = arr
            self._core_buffers[key] = new_arr

        for fields in self._component_buffers.values():
            for key, arr in fields.items():
                new_shape = (new_size, *arr.shape[1:])
                new_arr = np.zeros(new_shape, dtype=arr.dtype)
                new_arr[: self._buffer_size] = arr
                fields[key] = new_arr

        self._buffer_size = new_size

    def _create_buffer_array(self, val: ArrayLike) -> np.ndarray:
        """Create a pre-allocated NumPy array of the correct shape and dtype."""
        arr = np.asarray(val)
        dtype = arr.dtype
        shape = (self._buffer_size, *arr.shape)
        return np.zeros(shape, dtype=dtype)

    def _init_buffers(self, core: CoreLog, components: Mapping[str, Any]) -> None:
        """Initialize the pre-allocated buffers based on incoming data shapes and types."""
        self._core_buffers = {}
        self._component_buffers = {}
        self._component_fields = {}

        # Universal signals
        for field in dataclasses.fields(core):
            if field.name == "t":
                self._core_buffers["t"] = np.zeros((self._buffer_size,), dtype=np.float64)
                continue
            val = getattr(core, field.name)
            if val is not None:
                self._core_buffers[field.name] = self._create_buffer_array(val)

        # Component signals
        for name, log_model in components.items():
            self._component_buffers[name] = {}
            self._component_buffers[name]["t"] = np.zeros((self._buffer_size,), dtype=np.float64)

            if isinstance(log_model, NoLog):
                self._component_fields[name] = []
                continue

            # Assuming log_model is a dataclass
            fields = [
                f.name for f in dataclasses.fields(log_model) if f.name not in {"x", "y_mea", "x_hat", "u", "ref"}
            ]
            self._component_fields[name] = fields

            for key in fields:
                val = getattr(log_model, key)
                self._component_buffers[name][key] = self._create_buffer_array(val)

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

        if self._write_idx >= self._buffer_size:
            self._resize_buffers(self._buffer_size * 2)

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

    def flush_chunk(self, directory: str | Path, prefix: str = "log", *, compress: bool | None = None) -> None:
        """
        Write current in-memory buffers to a numbered chunk file, then clear them.

        No file or directory is created when both buffers are empty.
        Output file: {prefix}_chunk_{_chunk_idx:04d}.npz
        """
        if not self._buffers_initialized or self._write_idx == 0:
            return

        dir_path = Path(directory)
        chunk_dir = dir_path / f".{prefix}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        arrays_to_save: dict[str, np.ndarray] = {}

        if self._core_buffers:
            for key, arr in self._core_buffers.items():
                arrays_to_save[key] = arr[: self._write_idx]

        for name, fields in self._component_buffers.items():
            for key, arr in fields.items():
                arrays_to_save[f"{name}_{key}"] = arr[: self._write_idx]

        should_compress = compress if compress is not None else self.compress
        save_fn = np.savez_compressed if should_compress else np.savez

        if arrays_to_save:
            save_fn(
                chunk_dir / f"{prefix}_chunk_{self._chunk_idx:04d}.npz",
                **arrays_to_save,  # ty:ignore[invalid-argument-type]
            )

        self._chunk_idx += 1
        self._write_idx = 0

    @staticmethod
    def merge_chunks(directory: str | Path, prefix: str = "log", *, compress: bool = False) -> None:  # noqa: C901
        """Concatenate all chunk files for *prefix* into a single {prefix}.npz file.

        No-op when no chunk files are found. Individual chunk files are deleted
        after a successful merge.
        """
        dir_path = Path(directory)
        chunk_dir = dir_path / f".{prefix}_chunks"
        chunk_files = sorted(chunk_dir.glob(f"{prefix}_chunk_*.npz"))
        if not chunk_files:
            return

        # 1. Inspect first chunk file to get keys, dtypes, and shapes
        with np.load(chunk_files[0]) as data:
            keys = list(data.files)
            if not keys:
                return
            dtypes = {key: data[key].dtype for key in keys}
            sub_shapes = {key: data[key].shape[1:] for key in keys}

        # 2. Compute total length along axis 0 for each key
        total_lengths = dict.fromkeys(keys, 0)
        for chunk_file in chunk_files:
            with np.load(chunk_file) as data:
                for key in keys:
                    total_lengths[key] += data[key].shape[0]

        # 3. Create temporary memory-mapped .npy files for each key and concatenate on disk
        temp_files: dict[str, Path] = {}
        try:
            for key in keys:
                total_shape = (total_lengths[key], *sub_shapes[key])
                temp_path = chunk_dir / f"{prefix}_merge_{key}.npy.tmp"
                temp_files[key] = temp_path

                # Create the memory-mapped file with the proper NPY header
                mmap_arr = open_memmap(
                    temp_path,
                    mode="w+",
                    dtype=dtypes[key],
                    shape=total_shape,
                )

                # Append chunk data incrementally
                offset = 0
                for chunk_file in chunk_files:
                    with np.load(chunk_file) as data:
                        arr = data[key]
                        chunk_len = arr.shape[0]
                        mmap_arr[offset : offset + chunk_len] = arr
                        offset += chunk_len

                mmap_arr.flush()
                del mmap_arr

            # 4. Package all the temporary .npy files into the final .npz zip file
            zip_path = dir_path / f"{prefix}.npz"
            zip_mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
            with zipfile.ZipFile(zip_path, mode="w", compression=zip_mode) as zip_file:
                for key, temp_path in temp_files.items():
                    zip_file.write(temp_path, arcname=f"{key}.npy")

        finally:
            # 5. Clean up temporary files
            for temp_path in temp_files.values():
                if temp_path.exists():
                    with contextlib.suppress(Exception):
                        temp_path.unlink()

        # 6. Delete the chunk files after a successful merge
        for chunk_file in chunk_files:
            chunk_file.unlink()

        # 7. Delete the chunks directory if empty
        with contextlib.suppress(OSError):
            chunk_dir.rmdir()
