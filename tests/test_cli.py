from pathlib import Path

import pytest
import yaml

from simulate.main import main


def test_cli_missing_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.yaml"
    monkeypatch.setattr("sys.argv", ["simulate", str(non_existent)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_cli_run_single(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "t_end": 0.05,
        "dynamics": {
            "class_path": "simulate.dynamics.LinearDynamics",
            "dt": 0.01,
            "A": [[0.0]],
            "B": [[1.0]],
            "C": [[1.0]],
            "D": [[0.0]],
        },
        "reference": {
            "class_path": "simulate.reference.StepReference",
            "dt": 0.01,
            "step_value": 1.0,
            "start_time": 0.0,
        },
        "sensors": [
            {
                "class_path": "simulate.sensor.GaussianSensor",
                "dt": 0.01,
                "std_dev": 0.0,
                "measurement": {
                    "class_path": "simulate.sensor.LinearMeasurement",
                    "C": [[1.0]],
                    "D": [[0.0]],
                },
            },
        ],
        "estimator": {
            "class_path": "simulate.estimator.IdentityEstimator",
            "dt": 0.01,
        },
        "controller": {
            "class_path": "simulate.controller.PIController",
            "dt": 0.01,
            "kp": [[1.0]],
            "ki": [[0.0]],
        },
    }
    config_file = tmp_path / "sim.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    output_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["simulate", str(config_file), "--output-dir", str(output_dir)])
    main()

    assert (output_dir / "log.npz").exists()
