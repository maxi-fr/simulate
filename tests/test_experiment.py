from pathlib import Path
from typing import Any

import pytest

from simulate.experiment import ExperimentManager
from simulate.simulation import Simulation


def get_base_config() -> dict[str, Any]:
    """Return a valid base configuration for a simple simulation."""
    return {
        "t_end": 0.1,
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
    assert sim.dynamics.dt == 0.01


def test_experiment_manager_run_batch(tmp_path: Path) -> None:
    """Test that ExperimentManager can run a batch of simulations in parallel."""
    output_dir = tmp_path / "results"
    manager = ExperimentManager(output_dir=output_dir)

    configs = []
    for i in range(3):
        config = get_base_config()
        config["controller"]["kp"] = [[float(i + 1)]]
        configs.append(config)

    prefixes = ["sim_1", "sim_2", "sim_3"]
    results = manager.run_batch(configs, prefixes=prefixes)

    assert len(results) == 3
    assert all(results)

    for prefix in prefixes:
        assert (output_dir / f"{prefix}.npz").exists()


def test_experiment_manager_failure_handling(tmp_path: Path) -> None:
    """Test that ExperimentManager handles failed simulations gracefully."""
    output_dir = tmp_path / "results"
    manager = ExperimentManager(output_dir=output_dir)

    configs = []
    configs.append(get_base_config())
    invalid_config = get_base_config()
    del invalid_config["dynamics"]["class_path"]
    configs.append(invalid_config)

    results = manager.run_batch(configs)

    assert len(results) == 2
    assert results[0] is True
    assert results[1] is False


def test_single_simulation_from_config_with_single_elements() -> None:
    """Verify that a simulation config with a single dict for sensors can be loaded."""
    config = get_base_config()
    config["sensors"] = {
        "class_path": "simulate.sensor.GaussianSensor",
        "dt": 0.01,
        "std_dev": 0.0,
        "measurement": {
            "class_path": "simulate.sensor.LinearMeasurement",
            "C": [[1.0]],
            "D": [[0.0]],
        },
    }
    sim = Simulation.from_config(config)
    sim.run()
    assert sim.dynamics.dt == 0.01
    assert len(sim.sensors) == 1
