import contextlib
import dataclasses
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.lib.format import open_memmap

from .component import NoLog


@dataclasses.dataclass(frozen=True)
class UniversalLog:
    """Standardized signal vectors logged universally across all simulations."""

    t: float
    x: float | npt.NDArray[np.float64]
    x_hat: float | npt.NDArray[np.float64]
    u: float | npt.NDArray[np.float64]
    ref: float | npt.NDArray[np.float64]
    y_mea: float | npt.NDArray[np.float64] | None = None


def _determine_dtype(val: object) -> np.dtype | type:
    """Determine the NumPy dtype or Python type for a given value."""
    if isinstance(val, np.ndarray):
        return val.dtype  # type: ignore[no-any-return]
    if isinstance(val, bool):
        return bool
    if isinstance(val, int):
        return np.int64
    return np.float64


class Logger:
    """Centralized logger handling both universal signals and component-specific logs."""

    def __init__(self, *, compress: bool = False) -> None:
        """Initialize the logger."""
        self.compress = compress
        self._chunk_idx: int = 0
        self._buffer_size: int = 10_000
        self._write_idx: int = 0
        self._buffers_initialized: bool = False

        self._universal_buffers: dict[str, np.ndarray] = {}
        self._component_buffers: dict[str, dict[str, np.ndarray]] = {}
        self._component_fields: dict[str, list[str]] = {}

        self._universal_list: list[dict[str, Any]] = []
        self._component_lists: dict[str, list[dict[str, Any]]] = {}

    @property
    def universal_logs(self) -> list[dict[str, Any]]:
        """Return the universal logs as a list of dictionaries (constructed on demand)."""
        logs = list(self._universal_list)
        if self._buffers_initialized and self._write_idx > 0:
            for i in range(self._write_idx):
                entry = {}
                for key, arr in self._universal_buffers.items():
                    val = arr[i]
                    if isinstance(val, np.ndarray) and val.ndim == 0:
                        entry[key] = val.item()
                    else:
                        entry[key] = val
                logs.append(entry)
        return logs

    @property
    def component_logs(self) -> dict[str, list[dict[str, Any]]]:
        """Return the component logs as a dictionary of lists of dictionaries (constructed on demand)."""
        result = {name: list(lst) for name, lst in self._component_lists.items()}
        if self._buffers_initialized and self._write_idx > 0:
            for name, fields in self._component_buffers.items():
                if name not in result:
                    result[name] = []
                for i in range(self._write_idx):
                    entry = {}
                    for key, arr in fields.items():
                        val = arr[i]
                        if isinstance(val, np.ndarray) and val.ndim == 0:
                            entry[key] = val.item()
                        else:
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
        for key, arr in self._universal_buffers.items():
            new_shape = (new_size, *arr.shape[1:])
            new_arr = np.zeros(new_shape, dtype=arr.dtype)
            new_arr[: self._buffer_size] = arr
            self._universal_buffers[key] = new_arr

        for fields in self._component_buffers.values():
            for key, arr in fields.items():
                new_shape = (new_size, *arr.shape[1:])
                new_arr = np.zeros(new_shape, dtype=arr.dtype)
                new_arr[: self._buffer_size] = arr
                fields[key] = new_arr

        self._buffer_size = new_size

    def _create_buffer_array(self, val: object) -> np.ndarray:
        """Create a pre-allocated NumPy array of the correct shape and dtype."""
        dtype = _determine_dtype(val)
        if isinstance(val, np.ndarray) and val.ndim > 0:
            shape = (self._buffer_size, *val.shape)
        else:
            shape = (self._buffer_size,)
        return np.zeros(shape, dtype=dtype)

    def _init_buffers(self, universal: UniversalLog, components: Mapping[str, Any]) -> None:
        """Initialize the pre-allocated buffers based on incoming data shapes and types."""
        self._universal_buffers = {}
        self._component_buffers = {}
        self._component_fields = {}

        # Universal signals
        for key in ("t", "x", "y_mea", "x_hat", "u", "ref"):
            val = getattr(universal, key, None)
            if val is not None:
                self._universal_buffers[key] = self._create_buffer_array(val)

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

    def log(self, universal: UniversalLog, components: Mapping[str, Any]) -> None:  # noqa: C901
        """
        Record a snapshot of the simulation state.

        Parameters
        ----------
        universal : UniversalLog
            The universal log signals for this step.
        components : Mapping
            A dictionary mapping component names to their log models.
        """
        if not self._buffers_initialized:
            self._init_buffers(universal, components)

        if self._write_idx >= self._buffer_size:
            self._resize_buffers(self._buffer_size * 2)

        # Write universal signals
        for key in self._universal_buffers:
            val = getattr(universal, key, None)
            if val is not None:
                self._universal_buffers[key][self._write_idx] = val

        # Write component signals
        for name, log_model in components.items():
            if name not in self._component_buffers:
                continue

            # Fast path for NoLog
            if isinstance(log_model, NoLog):
                if "t" in self._component_buffers[name]:
                    self._component_buffers[name]["t"][self._write_idx] = universal.t
                continue

            # Write fields cached at initialization
            fields = self._component_fields.get(name, [])
            for key in fields:
                self._component_buffers[name][key][self._write_idx] = getattr(log_model, key)

            if "t" in self._component_buffers[name]:
                self._component_buffers[name]["t"][self._write_idx] = universal.t

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
        dir_path.mkdir(parents=True, exist_ok=True)

        arrays_to_save: dict[str, np.ndarray] = {}

        if self._universal_buffers:
            for key, arr in self._universal_buffers.items():
                arrays_to_save[f"universal_{key}"] = arr[: self._write_idx]

        for name, fields in self._component_buffers.items():
            for key, arr in fields.items():
                arrays_to_save[f"{name}_{key}"] = arr[: self._write_idx]

        should_compress = compress if compress is not None else self.compress
        save_fn = np.savez_compressed if should_compress else np.savez

        if arrays_to_save:
            save_fn(
                dir_path / f"{prefix}_chunk_{self._chunk_idx:04d}.npz",
                **arrays_to_save,  # ty:ignore[invalid-argument-type]
            )

        self._chunk_idx += 1
        self._write_idx = 0
        self._universal_list.clear()
        self._component_lists.clear()

    @staticmethod
    def merge_chunks(directory: str | Path, prefix: str = "log", *, compress: bool = False) -> None:  # noqa: C901
        """Concatenate all chunk files for *prefix* into a single {prefix}.npz file.

        No-op when no chunk files are found. Individual chunk files are deleted
        after a successful merge.
        """
        dir_path = Path(directory)
        chunk_files = sorted(dir_path.glob(f"{prefix}_chunk_*.npz"))
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
                temp_path = dir_path / f"{prefix}_merge_{key}.npy.tmp"
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
