from pathlib import Path

from .base import BaseLogger, CoreLog
from .memmap import MmapLogger
from .ram import RamLogger

__all__ = ["BaseLogger", "CoreLog", "MmapLogger", "RamLogger", "create_logger"]


def create_logger(
    total_steps: int,
    directory: str | Path | None = None,
    prefix: str = "log",
    *,
    compress: bool = False,
) -> BaseLogger:
    """Build the appropriate logger for a run of known length.

    Parameters
    ----------
    total_steps : int
        Number of log rows the run will produce.
    directory : str or Path or None, optional
        When given, a :class:`MmapLogger` streams signals to memory-mapped files
        under this directory. When ``None`` a :class:`RamLogger` keeps them in RAM.
    prefix : str, optional
        Base name for the memmap logger's temporary array folder.
    compress : bool, optional
        Whether the logger compresses its archive on :meth:`finalize` by default.

    Returns
    -------
    BaseLogger
        A :class:`MmapLogger` when *directory* is given, else a :class:`RamLogger`.
    """
    if directory is not None:
        return MmapLogger(total_steps, directory, prefix, compress=compress)
    return RamLogger(total_steps, compress=compress)
