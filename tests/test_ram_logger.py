import dataclasses
import zipfile
from pathlib import Path

import numpy as np
import pytest

from simulate.logger import RamLogger


def _npz_compress_types(path: Path) -> set[int]:
    """Return the set of zip compression methods used by the members of an ``.npz``.

    ``np.savez`` stores members with ``ZIP_STORED`` (0); ``np.savez_compressed``
    deflates them with ``ZIP_DEFLATED`` (8). Inspecting the members lets a test
    assert whether the archive was actually compressed, independent of its size.
    """
    with zipfile.ZipFile(path) as zf:
        return {info.compress_type for info in zf.infolist()}


@dataclasses.dataclass(frozen=True)
class MockComponentLog:
    value: float


def test_logger_log_storage() -> None:
    """Test that RamLogger stores component logs correctly."""
    logger = RamLogger(total_steps=1)
    components = {"comp1": MockComponentLog(value=42.0)}

    logger.log(0.1, components)

    assert len(logger.t) == 1
    assert logger.t[0] == 0.1

    assert logger.signal("comp1", "value")[0] == 42.0
    assert ("comp1", "value") in logger.signals()


def test_logger_signal_key_error() -> None:
    """Test that signal() raises KeyError with available signals on unknown component or field."""
    logger = RamLogger(total_steps=1)
    logger.log(0.0, {"comp1": MockComponentLog(value=42.0)})

    with pytest.raises(KeyError, match="Unknown component 'foo'"):
        logger.signal("foo", "value")

    with pytest.raises(KeyError, match="Unknown field 'bar'"):
        logger.signal("comp1", "bar")


def test_logger_export_ram_mode(tmp_path: Path) -> None:
    """Test that finalize works for an in-RAM logger."""
    logger = RamLogger(total_steps=3)

    for i in range(3):
        logger.log(float(i), {"comp1": MockComponentLog(value=float(i * 10))})
    logger.finalize(tmp_path, prefix="test")

    data = np.load(tmp_path / "test.npz")
    assert len(data["t"]) == 3
    assert list(data["t"]) == [0.0, 1.0, 2.0]
    assert list(data["comp1.value"]) == [0.0, 10.0, 20.0]
    # RAM mode never creates the memmap array directory.
    assert not (tmp_path / ".test_arrays").exists()


def test_logger_raises_when_capacity_exceeded() -> None:
    """Test that logging past the configured total_steps raises instead of growing."""
    logger = RamLogger(total_steps=2)

    logger.log(0.0, {})
    logger.log(1.0, {})
    with pytest.raises(RuntimeError, match="capacity"):
        logger.log(2.0, {})


def test_logger_finalize_raises_on_partial_fill(tmp_path: Path) -> None:
    """Test that finalizing after logging fewer rows than total_steps raises loudly."""
    logger = RamLogger(total_steps=5)

    for i in range(3):
        logger.log(float(i), {})

    with pytest.raises(RuntimeError, match="3 of 5"):
        logger.finalize(tmp_path, prefix="test")


def test_ram_export_compressed_roundtrip(tmp_path: Path) -> None:
    """Test that a compressed RAM export deflates every member and round-trips its data."""
    logger = RamLogger(total_steps=4)
    for i in range(4):
        logger.log(float(i), {"comp1": MockComponentLog(value=float(i))})
    logger.finalize(tmp_path, prefix="c", compress=True)

    npz_file = tmp_path / "c.npz"
    assert _npz_compress_types(npz_file) == {zipfile.ZIP_DEFLATED}

    data = np.load(npz_file)
    assert list(data["t"]) == [0.0, 1.0, 2.0, 3.0]
    assert data["comp1.value"][2] == 2.0


def test_ram_export_uncompressed_is_stored(tmp_path: Path) -> None:
    """Test that an uncompressed RAM export stores its members without deflation."""
    logger = RamLogger(total_steps=2)
    for i in range(2):
        logger.log(float(i), {})
    logger.finalize(tmp_path, prefix="u", compress=False)

    assert _npz_compress_types(tmp_path / "u.npz") == {zipfile.ZIP_STORED}


def test_ram_finalize_defaults_to_uncompressed(tmp_path: Path) -> None:
    """Test that finalize() with no compress argument stores the archive uncompressed."""
    logger = RamLogger(total_steps=2)
    for i in range(2):
        logger.log(float(i), {})
    logger.finalize(tmp_path, prefix="d")  # compress defaults to False

    assert _npz_compress_types(tmp_path / "d.npz") == {zipfile.ZIP_STORED}
