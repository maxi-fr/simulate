import dataclasses
from pathlib import Path

import numpy as np
import pytest

from simulate.logger import CoreLog, MmapLogger, RamLogger


@dataclasses.dataclass(frozen=True)
class MockComponentLog:
    value: float


def _core(t: float, dim: int = 1) -> CoreLog:
    """Build a CoreLog whose vectors are filled with *t* (dimension *dim*)."""
    vec = np.full((dim,), t)
    return CoreLog(t=t, x=vec, y_mea=vec.copy(), x_hat=vec.copy(), u=vec.copy(), ref=vec.copy())


def test_logger_export_memmap(tmp_path: Path) -> None:
    """Test that a memory-mapped run packs core and component signals into {prefix}.npz."""
    logger = MmapLogger(total_steps=1, directory=tmp_path, prefix="test")

    core = CoreLog(
        t=0.1,
        x=np.array([1.0, 2.0]),
        y_mea=np.array([1.1, 2.1]),
        x_hat=np.array([1.2, 2.2]),
        u=np.array([0.5]),
        ref=np.array([1.0]),
    )
    logger.log(core, {"comp1": MockComponentLog(value=42.0)})
    logger.finalize(tmp_path, prefix="test")

    npz_file = tmp_path / "test.npz"
    assert npz_file.exists()

    data = np.load(npz_file)
    assert data["t"][0] == 0.1
    assert np.array_equal(data["x"][0], np.array([1.0, 2.0]))
    assert data["comp1_value"][0] == 42.0
    assert data["comp1_t"][0] == 0.1

    # The temporary per-signal .npy files are cleaned up after packing.
    assert not (tmp_path / ".test_arrays").exists()


def test_logger_empty_export(tmp_path: Path) -> None:
    """Test that finalizing with no logged data writes no files."""
    logger = MmapLogger(total_steps=10, directory=tmp_path, prefix="test")

    logger.finalize(tmp_path, prefix="test")

    assert not (tmp_path / "test.npz").exists()
    assert not (tmp_path / ".test_arrays").exists()


def test_logger_export_multiple_rows(tmp_path: Path) -> None:
    """Test that multiple logged steps are concatenated in order in the archive."""
    logger = MmapLogger(total_steps=2, directory=tmp_path, prefix="test")

    logger.log(_core(0.0), {})
    logger.log(_core(1.0), {})
    logger.finalize(tmp_path, prefix="test")

    data = np.load(tmp_path / "test.npz")
    assert len(data["t"]) == 2
    assert data["t"][0] == 0.0
    assert data["t"][1] == 1.0


def test_logger_finalize_raises_on_partial_fill(tmp_path: Path) -> None:
    """Test that finalizing after logging fewer rows than total_steps raises loudly."""
    logger = MmapLogger(total_steps=100, directory=tmp_path, prefix="test")

    for i in range(3):
        logger.log(_core(float(i)), {})

    with pytest.raises(RuntimeError, match="3 of 100"):
        logger.finalize(tmp_path, prefix="test")


def test_logger_export_large_multidim(tmp_path: Path) -> None:
    """Test that large multi-dimensional signals round-trip through a compressed archive."""
    x_shape = (6, 76)
    n = 6
    logger = MmapLogger(total_steps=n, directory=tmp_path, prefix="test")

    for i in range(n):
        core = CoreLog(
            t=float(i),
            x=np.full(x_shape, float(i)),
            y_mea=np.array([float(i)]),
            x_hat=np.array([float(i)]),
            u=np.array([float(i)]),
            ref=np.array([float(i)]),
        )
        logger.log(core, {})
    logger.finalize(tmp_path, prefix="test", compress=True)

    npz_file = tmp_path / "test.npz"
    assert npz_file.exists()

    # No temporary .npy files or directory are left behind.
    assert not list(tmp_path.glob("**/*.npy"))
    assert not (tmp_path / ".test_arrays").exists()

    data = np.load(npz_file)
    assert data["x"].shape == (n, 6, 76)
    for i in range(n):
        assert np.all(data["x"][i] == float(i))


def test_memmap_matches_ram(tmp_path: Path) -> None:
    """Test that memmap-backed and in-RAM logging produce byte-identical archives."""
    entries = [_core(float(i), dim=2) for i in range(5)]

    mm_logger = MmapLogger(total_steps=5, directory=tmp_path, prefix="mm")
    for entry in entries:
        mm_logger.log(entry, {"comp1": MockComponentLog(value=float(entry.t))})
    mm_logger.finalize(tmp_path, prefix="mm")
    mm_data = np.load(tmp_path / "mm.npz")

    ram_logger = RamLogger(total_steps=5)
    for entry in entries:
        ram_logger.log(entry, {"comp1": MockComponentLog(value=float(entry.t))})
    ram_logger.finalize(tmp_path, prefix="ram")
    ram_data = np.load(tmp_path / "ram.npz")

    assert set(mm_data.files) == set(ram_data.files)
    for key in mm_data.files:
        assert np.array_equal(mm_data[key], ram_data[key]), key
