import contextlib
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .base import BaseLogger


class MmapLogger(BaseLogger):
    """Logger that streams every signal to memory-mapped ``.npy`` files on disk.

    Each buffer is an ``open_memmap`` file under ``directory/.{prefix}_arrays`` sized
    to the known step count, so writes go to the OS page cache and resident memory
    stays bounded regardless of run length. :meth:`finalize` flushes, closes, and
    packs those files into a single ``{prefix}.npz`` (a byte copy, no re-serialization).
    """

    def __init__(self, total_steps: int, directory: str | Path, prefix: str = "log") -> None:
        """Initialize the logger, backing signal buffers with files under *directory*.

        Parameters
        ----------
        total_steps : int
            Number of log rows the run will produce; each memmap is sized to this.
        directory : str or Path
            Directory under which the temporary ``.{prefix}_arrays`` folder is created.
        prefix : str, optional
            Base name for the temporary array folder.
        """
        super().__init__(total_steps)
        self._directory = Path(directory)
        self._prefix = prefix
        self._array_dir = self._directory / f".{self._prefix}_arrays"
        self._memmap_paths: dict[str, Path] = {}

    def _prepare_storage(self) -> None:
        """Create the temporary array directory and reset the path map before allocation."""
        self._memmap_paths = {}
        self._array_dir.mkdir(parents=True, exist_ok=True)

    def _make_buffer(self, arcname: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Allocate a memory-mapped ``.npy`` buffer for signal *arcname*."""
        path = self._array_dir / f"{arcname}.npy"
        mmap_arr = open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        self._memmap_paths[arcname] = path
        return mmap_arr

    def _finalize(self, zip_path: Path, *, compress: bool) -> None:
        """Flush and close the memmaps, then pack them into ``zip_path``.

        The memmaps are sized to the exact run length (``finalize`` rejects a partial
        fill), so each ``.npy`` is packed as-is with no trimming or re-serialization.
        """
        buffers = list(self._iter_buffers())
        for _, mmap_arr in buffers:
            flush = getattr(mmap_arr, "flush", None)
            if callable(flush):
                with contextlib.suppress(Exception):
                    flush()
        for _, mmap_arr in buffers:
            self._release_memmap(mmap_arr)
        # Drop references to the (now closed) memmaps so the files are fully unlocked.
        self._core_buffers = {}
        self._component_buffers = {}
        del buffers

        zip_mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        with zipfile.ZipFile(zip_path, mode="w", compression=zip_mode) as zip_file:
            for arcname, path in self._memmap_paths.items():
                zip_file.write(path, arcname=f"{arcname}.npy")

        self._cleanup()

    def _cleanup(self) -> None:
        """Delete the temporary memmap directory and its ``.npy`` files, if any."""
        if self._array_dir.exists():
            for path in self._array_dir.iterdir():
                with contextlib.suppress(OSError):
                    path.unlink()
            with contextlib.suppress(OSError):
                self._array_dir.rmdir()
        self._memmap_paths = {}

    @staticmethod
    def _release_memmap(mmap_arr: np.ndarray) -> None:
        """Close the underlying mmap so the OS releases its file handle (needed on Windows)."""
        base = getattr(mmap_arr, "_mmap", None)
        if base is None:
            base = getattr(mmap_arr, "base", None)
        if base is not None and hasattr(base, "close"):
            with contextlib.suppress(Exception):
                base.close()
