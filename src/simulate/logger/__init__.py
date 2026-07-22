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
    use_mmap: bool = False,
) -> BaseLogger:
    """Build the appropriate logger for a run of known length.

    Compression is not decided here: it is chosen when the archive is written, via
    the ``compress`` argument of :meth:`BaseLogger.finalize`.

    Parameters
    ----------
    total_steps : int
        Number of log rows the run will produce.
    directory : str or Path or None, optional
        The directory for the logger. Required if use_mmap is True.
    prefix : str, optional
        Base name for the memmap logger's temporary array folder.
    use_mmap : bool, optional
        If True, a :class:`MmapLogger` streams signals to memory-mapped files.
        If False, a :class:`RamLogger` keeps them in RAM.

    Returns
    -------
    BaseLogger
        A :class:`MmapLogger` when use_mmap is True, else a :class:`RamLogger`.
    """
    if use_mmap:
        if directory is None:
            msg = "MmapLogger requires a directory."
            raise ValueError(msg)
        return MmapLogger(total_steps, directory, prefix)
    return RamLogger(total_steps)
