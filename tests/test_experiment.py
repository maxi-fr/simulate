from pathlib import Path
from typing import Any

import pytest

from simulate.experiment import ExperimentManager
from simulate.simulation import Simulation


def get_base_config() -> dict[str, Any]:
    """Return a valid base configuration for a simple simulation."""
    return {
        "t_end": 0.1,
        "plant": {
            "class_path": "simulate.plant.LinearPlant",
            "dt": 0.01,
            "a": [[0.0]],
            "b": [[1.0]],
            "c": [[1.0]],
            "d": [[0.0]],
        },
        "sensor": {
            "class_path": "simulate.sensor.GaussianSensor",
            "dt": 0.01,
            "std_dev": 0.0,
        },
        "estimator": {
            "class_path": "simulate.estimator.IdentityEstimator",
            "dt": 0.01,
        },
        "controller": {
            "class_path": "simulate.controller.PIDController",
            "dt": 0.01,
            "kp": [[1.0]],
            "ki": [[0.0]],
            "kd": [[0.0]],
        },
    }


def test_single_simulation_from_config() -> None:
    """Verify that a single simulation can be instantiated and run from a config dict."""
    config = get_base_config()
    sim = Simulation.from_config(config)
    sim.run()
    assert sim.plant.dt == 0.01


def test_experiment_manager_run_batch(tmp_path: Path) -> None:
    """Test that ExperimentManager can run a batch of simulations in parallel."""
    output_dir = tmp_path / "results"
    manager = ExperimentManager(output_dir=output_dir)

    # Create 3 slightly different configs
    configs = []
    for i in range(3):
        config = get_base_config()
        config["controller"]["kp"] = [[float(i + 1)]]
        configs.append(config)

    prefixes = ["sim_1", "sim_2", "sim_3"]
    results = manager.run_batch(configs, prefixes=prefixes)

    # Verify results
    assert len(results) == 3
    assert all(results)

    # Verify files were created
    for prefix in prefixes:
        assert (output_dir / f"{prefix}_data.npz").exists()
        assert (output_dir / f"{prefix}_universal.csv").exists()


def test_experiment_manager_failure_handling(tmp_path: Path) -> None:
    """Test that ExperimentManager handles failed simulations gracefully."""
    output_dir = tmp_path / "results"
    manager = ExperimentManager(output_dir=output_dir)

    configs = []
    # One valid config
    configs.append(get_base_config())
    # One invalid config (missing class_path)
    invalid_config = get_base_config()
    del invalid_config["plant"]["class_path"]
    configs.append(invalid_config)

    results = manager.run_batch(configs)

    assert len(results) == 2
    assert results[0] is True
    assert results[1] is False
