from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from simulate.logger import Logger, UniversalLog


class MockComponentLog(BaseModel):
    value: float


def test_universal_log_validation() -> None:
    """Test that UniversalLog validates signals."""
    UniversalLog(
        t=0.0,
        x=1.0,
        y=1.0,
        y_mea=1.1,
        x_hat=1.2,
        u=0.5,
        ref=1.0,
    )
    UniversalLog(
        t=0.0,
        x=np.array([1.0]),
        y=np.array([1.0]),
        y_mea=np.array([1.1]),
        x_hat=np.array([1.2]),
        u=np.array([0.5]),
        ref=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="Array must be 1D"):
        UniversalLog(
            t=0.0,
            x=1.0,
            y=np.array([[1.0]]),
            y_mea=1.1,
            x_hat=1.2,
            u=0.5,
            ref=1.0,
        )


def test_logger_log_storage() -> None:
    """Test that Logger stores universal and component logs correctly."""
    logger = Logger()
    universal = UniversalLog(
        t=0.1,
        x=1.0,
        y=1.0,
        y_mea=1.1,
        x_hat=1.2,
        u=0.5,
        ref=1.0,
    )
    components = {"comp1": MockComponentLog(value=42.0)}

    logger.log(universal, components)

    assert len(logger.universal_logs) == 1
    assert logger.universal_logs[0]["t"] == 0.1
    assert logger.universal_logs[0]["y"] == 1.0

    assert "comp1" in logger.component_logs
    assert len(logger.component_logs["comp1"]) == 1
    assert logger.component_logs["comp1"][0]["value"] == 42.0
    assert logger.component_logs["comp1"][0]["t"] == 0.1


def test_logger_flush_chunk(tmp_path: Path) -> None:
    """Test that Logger writes a chunk file and clears buffers."""
    logger = Logger()
    universal = UniversalLog(
        t=0.1,
        x=np.array([1.0, 2.0]),
        y=np.array([1.0, 2.0]),
        y_mea=np.array([1.1, 2.1]),
        x_hat=np.array([1.2, 2.2]),
        u=np.array([0.5]),
        ref=np.array([1.0]),
    )
    components = {"comp1": MockComponentLog(value=42.0)}
    logger.log(universal, components)

    export_dir = tmp_path / "export_npz"
    logger.flush_chunk(export_dir, prefix="test")

    npz_file = export_dir / "test_chunk_0000.npz"
    assert npz_file.exists()

    data = np.load(npz_file)
    assert "universal_t" in data
    assert data["universal_t"][0] == 0.1
    assert "comp1_value" in data
    assert data["comp1_value"][0] == 42.0
    assert "comp1_t" in data
    assert data["comp1_t"][0] == 0.1

    assert len(logger.universal_logs) == 0
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

    logger.log(UniversalLog(t=0.0, x=1.0, y=1.0, y_mea=1.0, x_hat=1.0, u=0.0, ref=1.0), {})
    logger.flush_chunk(tmp_path, prefix="test")

    logger.log(UniversalLog(t=1.0, x=2.0, y=2.0, y_mea=2.0, x_hat=2.0, u=1.0, ref=2.0), {})
    logger.flush_chunk(tmp_path, prefix="test")

    Logger.merge_chunks(tmp_path, prefix="test")

    merged_file = tmp_path / "test.npz"
    assert merged_file.exists()
    assert not (tmp_path / "test_chunk_0000.npz").exists()
    assert not (tmp_path / "test_chunk_0001.npz").exists()

    data = np.load(merged_file)
    assert len(data["universal_t"]) == 2
    assert data["universal_t"][0] == 0.0
    assert data["universal_t"][1] == 1.0


def test_logger_multiple_chunks(tmp_path: Path) -> None:
    """Test that consecutive flush_chunk calls write distinct chunk files."""
    logger = Logger()

    universal = UniversalLog(t=0.0, x=1.0, y=1.0, y_mea=1.0, x_hat=1.0, u=0.0, ref=1.0)
    logger.log(universal, {})
    logger.flush_chunk(tmp_path, prefix="test")

    universal = UniversalLog(t=1.0, x=2.0, y=2.0, y_mea=2.0, x_hat=2.0, u=1.0, ref=2.0)
    logger.log(universal, {})
    logger.flush_chunk(tmp_path, prefix="test")

    chunk0 = np.load(tmp_path / "test_chunk_0000.npz")
    chunk1 = np.load(tmp_path / "test_chunk_0001.npz")

    assert chunk0["universal_t"][0] == 0.0
    assert chunk1["universal_t"][0] == 1.0
    assert len(chunk0["universal_t"]) == 1
    assert len(chunk1["universal_t"]) == 1
