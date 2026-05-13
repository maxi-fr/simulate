from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from simulate.logger import Logger, UniversalLog


class MockComponentLog(BaseModel):
    value: float


def test_universal_log_validation() -> None:
    """Test that UniversalLog validates signals."""
    UniversalLog(
        t=0.0,
        y=1.0,
        y_mea=1.1,
        x_hat=1.2,
        u=0.5,
        ref=1.0,
    )
    UniversalLog(
        t=0.0,
        y=np.array([1.0]),
        y_mea=np.array([1.1]),
        x_hat=np.array([1.2]),
        u=np.array([0.5]),
        ref=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="Array must be 1D"):
        UniversalLog(
            t=0.0,
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


def test_logger_export_csv(tmp_path: Path) -> None:
    """Test that Logger exports to CSV correctly."""
    logger = Logger()
    universal = UniversalLog(
        t=0.1,
        y=1.0,
        y_mea=1.1,
        x_hat=1.2,
        u=0.5,
        ref=1.0,
    )
    components = {"comp1": MockComponentLog(value=42.0)}
    logger.log(universal, components)

    export_dir = tmp_path / "export"
    logger.export_csv(export_dir, prefix="test")

    universal_csv = export_dir / "test_universal.csv"
    comp_csv = export_dir / "test_comp_comp1.csv"

    assert universal_csv.exists()
    assert comp_csv.exists()

    df_universal = pd.read_csv(universal_csv)
    assert df_universal.iloc[0]["t"] == 0.1

    df_comp = pd.read_csv(comp_csv)
    assert df_comp.iloc[0]["value"] == 42.0
    assert df_comp.iloc[0]["t"] == 0.1


def test_logger_export_npz(tmp_path: Path) -> None:
    """Test that Logger exports to NPZ correctly."""
    logger = Logger()
    universal = UniversalLog(
        t=0.1,
        y=np.array([1.0, 2.0]),
        y_mea=np.array([1.1, 2.1]),
        x_hat=np.array([1.2, 2.2]),
        u=np.array([0.5]),
        ref=np.array([1.0]),
    )
    components = {"comp1": MockComponentLog(value=42.0)}
    logger.log(universal, components)

    export_dir = tmp_path / "export_npz"
    logger.export_npz(export_dir, prefix="test")

    npz_file = export_dir / "test_data.npz"
    assert npz_file.exists()

    data = np.load(npz_file)
    assert "universal_t" in data
    assert data["universal_t"][0] == 0.1
    assert "comp1_value" in data
    assert data["comp1_value"][0] == 42.0
    assert "comp1_t" in data
    assert data["comp1_t"][0] == 0.1


def test_logger_empty_export(tmp_path: Path) -> None:
    """Test that exporting empty logs doesn't crash and behaves as expected."""
    logger = Logger()
    export_dir = tmp_path / "empty_export"

    logger.export_csv(export_dir)
    logger.export_npz(export_dir)

    assert not any(export_dir.iterdir())
