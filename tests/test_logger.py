import dataclasses
from pathlib import Path

import numpy as np
import pytest

from simulate.logger import CoreLog, Logger


@dataclasses.dataclass(frozen=True)
class MockComponentLog:
    value: float


def test_logger_log_storage() -> None:
    """Test that Logger stores universal and component logs correctly."""
    logger = Logger()
    core = CoreLog(
        t=0.1,
        x=1.0,
        y_mea=1.1,
        x_hat=1.2,
        u=0.5,
        ref=1.0,
    )
    components = {"comp1": MockComponentLog(value=42.0)}

    logger.log(core, components)

    assert len(logger.core_logs) == 1
    assert logger.core_logs[0]["t"] == 0.1
    assert logger.core_logs[0]["y_mea"] == 1.1

    assert "comp1" in logger.component_logs
    assert len(logger.component_logs["comp1"]) == 1
    assert logger.component_logs["comp1"][0]["value"] == 42.0
    assert logger.component_logs["comp1"][0]["t"] == 0.1


def test_logger_flush_chunk(tmp_path: Path) -> None:
    """Test that Logger writes a chunk file and clears buffers."""
    logger = Logger()
    core = CoreLog(
        t=0.1,
        x=np.array([1.0, 2.0]),
        y_mea=np.array([1.1, 2.1]),
        x_hat=np.array([1.2, 2.2]),
        u=np.array([0.5]),
        ref=np.array([1.0]),
    )
    components = {"comp1": MockComponentLog(value=42.0)}
    logger.log(core, components)

    export_dir = tmp_path / "export_npz"
    logger.flush_chunk(export_dir, prefix="test")

    npz_file = export_dir / "test_chunk_0000.npz"
    assert npz_file.exists()

    data = np.load(npz_file)
    assert "core_t" in data
    assert data["core_t"][0] == 0.1
    assert "comp1_value" in data
    assert data["comp1_value"][0] == 42.0
    assert "comp1_t" in data
    assert data["comp1_t"][0] == 0.1

    assert len(logger.core_logs) == 0
    assert len(logger.component_logs) == 0


def test_logger_empty_export(tmp_path: Path) -> None:
    """Test that flushing empty logs doesn't crash and writes no files."""
    logger = Logger()
    export_dir = tmp_path / "empty_export"

    logger.flush_chunk(export_dir, prefix="test")

    if export_dir.exists():
        assert not any(export_dir.iterdir())


def test_logger_merge_chunks(tmp_path: Path) -> None:
    """Test that merge_chunks concatenates chunk files into one and deletes the originals."""
    logger = Logger()

    logger.log(CoreLog(t=0.0, x=1.0, y_mea=1.0, x_hat=1.0, u=0.0, ref=1.0), {})
    logger.flush_chunk(tmp_path, prefix="test")

    logger.log(CoreLog(t=1.0, x=2.0, y_mea=2.0, x_hat=2.0, u=1.0, ref=2.0), {})
    logger.flush_chunk(tmp_path, prefix="test")

    Logger.merge_chunks(tmp_path, prefix="test")

    merged_file = tmp_path / "test.npz"
    assert merged_file.exists()
    assert not (tmp_path / "test_chunk_0000.npz").exists()
    assert not (tmp_path / "test_chunk_0001.npz").exists()

    data = np.load(merged_file)
    assert len(data["core_t"]) == 2
    assert data["core_t"][0] == 0.0
    assert data["core_t"][1] == 1.0


def test_logger_multiple_chunks(tmp_path: Path) -> None:
    """Test that consecutive flush_chunk calls write distinct chunk files."""
    logger = Logger()

    core = CoreLog(t=0.0, x=1.0, y_mea=1.0, x_hat=1.0, u=0.0, ref=1.0)
    logger.log(core, {})
    logger.flush_chunk(tmp_path, prefix="test")

    core = CoreLog(t=1.0, x=2.0, y_mea=2.0, x_hat=2.0, u=1.0, ref=2.0)
    logger.log(core, {})
    logger.flush_chunk(tmp_path, prefix="test")

    chunk0 = np.load(tmp_path / "test_chunk_0000.npz")
    chunk1 = np.load(tmp_path / "test_chunk_0001.npz")

    assert chunk0["core_t"][0] == 0.0
    assert chunk1["core_t"][0] == 1.0
    assert len(chunk0["core_t"]) == 1
    assert len(chunk1["core_t"]) == 1


def test_logger_merge_chunks_memory_efficient(tmp_path: Path) -> None:
    """Test that merge_chunks can handle large multi-dimensional arrays memory-efficiently."""
    logger = Logger()

    # Create several chunks with a multi-dimensional array
    # We will simulate `core_x` having shape (6, 76)
    x_shape = (6, 76)

    for i in range(3):
        x_val = np.full(x_shape, float(i))
        core = CoreLog(t=float(i), x=x_val, y_mea=float(i), x_hat=float(i), u=float(i), ref=float(i))
        logger.log(core, {})
        # Log another point to make chunk length > 1
        x_val2 = np.full(x_shape, float(i) + 0.5)
        core2 = CoreLog(
            t=float(i) + 0.5, x=x_val2, y_mea=float(i) + 0.5, x_hat=float(i) + 0.5, u=float(i) + 0.5, ref=float(i) + 0.5
        )
        logger.log(core2, {})

        logger.flush_chunk(tmp_path, prefix="test")

    # At this point, we should have 3 chunk files, each with 2 rows.
    # Total shape of merged core_x should be (6, 6, 76)

    Logger.merge_chunks(tmp_path, prefix="test", compress=True)

    merged_file = tmp_path / "test.npz"
    assert merged_file.exists()

    # Ensure temporary files were deleted
    temp_files = list(tmp_path.glob("*.npy.tmp"))
    assert not temp_files

    # Ensure chunk files were deleted
    chunk_files = list(tmp_path.glob("test_chunk_*.npz"))
    assert not chunk_files

    data = np.load(merged_file)
    assert "core_x" in data

    x_merged = data["core_x"]
    assert x_merged.shape == (6, 6, 76)

    # Verify the contents
    assert np.all(x_merged[0] == 0.0)
    assert np.all(x_merged[1] == 0.5)
    assert np.all(x_merged[2] == 1.0)
    assert np.all(x_merged[3] == 1.5)
    assert np.all(x_merged[4] == 2.0)
    assert np.all(x_merged[5] == 2.5)
