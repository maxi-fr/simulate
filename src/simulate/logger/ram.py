from pathlib import Path

import numpy as np

from .base import BaseLogger


class RamLogger(BaseLogger):
    """Logger that keeps every signal in ordinary in-RAM numpy arrays.

    Buffers live on the Python heap, so resident memory scales with the run length.
    :meth:`finalize` saves the accumulated arrays (sliced to the number of logged
    rows) with ``np.savez``. Suitable when the logs are small enough to hold in
    memory or need to be inspected via ``core_logs`` / ``component_logs``.
    """

    def _make_buffer(self, arcname: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:  # noqa: ARG002 - arcname is part of the backend interface, only the memmap backend needs it
        """Allocate a zero-filled in-RAM buffer of *shape* and *dtype*."""
        return np.zeros(shape, dtype=dtype)

    def _finalize(self, zip_path: Path, *, compress: bool) -> None:
        """Save the in-RAM buffers to ``zip_path`` (they are sized to the exact run length)."""
        arrays_to_save = dict(self._iter_buffers())
        save_fn = np.savez_compressed if compress else np.savez
        save_fn(zip_path, **arrays_to_save)  # ty:ignore[invalid-argument-type]
