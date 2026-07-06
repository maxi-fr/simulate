import dataclasses
from pathlib import Path

import numpy as np
import pytest

from simulate.logger import CoreLog, RamLogger


@dataclasses.dataclass(frozen=True)
class MockComponentLog:
    value: float


def _core(t: float, dim: int = 1) -> CoreLog:
    """Build a CoreLog whose vectors are filled with *t* (dimension *dim*)."""
    vec = np.full((dim,), t)
    return CoreLog(t=t, x=vec, y_mea=vec.copy(), x_hat=vec.copy(), u=vec.copy(), ref=vec.copy())


def test_logger_log_storage() -> None:
    """Test that RamLogger stores universal and component logs correctly."""
    logger = RamLogger(total_steps=1)
    core = CoreLog(
        t=0.1,
        x=np.array([1.0]),
        y_mea=np.array([1.1]),
        x_hat=np.array([1.2]),
        u=np.array([0.5]),
        ref=np.array([1.0]),
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


def test_logger_export_ram_mode(tmp_path: Path) -> None:
    """Test that finalize works for an in-RAM logger."""
    logger = RamLogger(total_steps=3)

    for i in range(3):
        logger.log(_core(float(i)), {})
    logger.finalize(tmp_path, prefix="test")

    data = np.load(tmp_path / "test.npz")
    assert len(data["t"]) == 3
    assert list(data["t"]) == [0.0, 1.0, 2.0]
    # RAM mode never creates the memmap array directory.
    assert not (tmp_path / ".test_arrays").exists()


def test_logger_raises_when_capacity_exceeded() -> None:
    """Test that logging past the configured total_steps raises instead of growing."""
    logger = RamLogger(total_steps=2)

    logger.log(_core(0.0), {})
    logger.log(_core(1.0), {})
    with pytest.raises(RuntimeError, match="capacity"):
        logger.log(_core(2.0), {})


def test_logger_finalize_raises_on_partial_fill(tmp_path: Path) -> None:
    """Test that finalizing after logging fewer rows than total_steps raises loudly."""
    logger = RamLogger(total_steps=5)

    for i in range(3):
        logger.log(_core(float(i)), {})

    with pytest.raises(RuntimeError, match="3 of 5"):
        logger.finalize(tmp_path, prefix="test")
