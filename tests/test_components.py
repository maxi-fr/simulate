import math

import numpy as np
import pytest

from simulate.controller import PIDController
from simulate.dynamics import LinearDynamics
from simulate.estimator import IdentityEstimator
from simulate.output import LinearOutput
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor
from simulate.simulation import Simulation


def test_component_conversion_utilities() -> None:
    """Test to_col_vec and from_col_vec utilities in Component."""
    dynamics = LinearDynamics(dt=0.1, a=[[1]], b=[[1]])
    LinearOutput(dt=0.1, c=[[1]], d=[[0]])

    res = dynamics.to_col_vec(1.0)
    assert res.shape == (1, 1)
    assert res[0, 0] == 1.0

    res = dynamics.to_col_vec(np.array([1.0, 2.0]))
    assert res.shape == (2, 1)
    assert res[0, 0] == 1.0
    assert res[1, 0] == 2.0

    res = dynamics.to_col_vec(np.array([[1.0], [2.0]]))
    assert res.shape == (2, 1)

    res_back = dynamics.from_col_vec(np.array([[1.0]]))
    assert isinstance(res_back, float)
    assert res_back == 1.0

    res_back = dynamics.from_col_vec(np.array([[1.0], [2.0]]))
    assert isinstance(res_back, np.ndarray)
    assert res_back.shape == (2,)
    assert res_back[0] == 1.0
    assert res_back[1] == 2.0


def test_plant_step_logic() -> None:
    """Test standard plant update dynamics."""
    dynamics = LinearDynamics(dt=0.1, a=[[0.9]], b=[[1.0]])
    output = LinearOutput(dt=0.1, c=[[1.0]], d=[[0.0]])

    assert dynamics.x[0, 0] == 0.0

    u1 = 1.0
    x, dynamics_log = dynamics.step(0.0, 1.0)
    y, _ = output.step(0.0, x, u1)
    assert y == 1.0
    assert dynamics_log.x[0, 0] == 1.0

    u2 = 0.5
    x, dynamics_log = dynamics.step(0.1, u2)
    y, _output_log = output.step(0.1, x, u2)
    assert y == 1.4
    assert dynamics_log.x[0, 0] == 1.4


def test_sensor_step_logic() -> None:
    """Test sensor behavior with Gaussian noise."""
    sensor = GaussianSensor(dt=0.1, std_dev=0.0)
    y = 1.0
    y_mea, log = sensor.step(0.0, y)
    assert y_mea == 1.0
    assert log.noise == 0.0

    sensor_noise = GaussianSensor(dt=0.1, std_dev=0.1)
    y_mea2, log2 = sensor_noise.step(0.0, y)
    assert y_mea2 != 1.0
    assert log2.noise != 0.0


def test_estimator_step_logic() -> None:
    """Test identity estimator behavior."""
    estimator = IdentityEstimator(dt=0.1)
    y_mea = 1.2
    u = 0.5
    x_hat, log = estimator.step(0.0, y_mea, u)
    assert x_hat == 1.2
    assert log.y_mea == 1.2


def test_controller_step_logic() -> None:
    """Test PI controller behavior and integration accumulation."""
    controller = PIDController(dt=0.1, kp=[[0.5]], ki=[[0.1]], kd=[[0.0]])

    ref = 1.0
    x_hat = 0.0
    u, log = controller.step(0.0, ref, x_hat)
    assert math.isclose(u, 0.51)
    assert log.error == 1.0
    assert log.integral == 0.1


def test_component_zoh_behavior() -> None:
    """Test that a component retains its last output between scheduled updates."""
    controller = PIDController(dt=0.2, kp=[[0.5]], ki=[[0.1]], kd=[[0.0]])

    ref1 = 1.0
    x_hat1 = 0.0
    u1, log1 = controller.step(0.0, ref1, x_hat1)
    assert math.isclose(u1, 0.52)

    ref2 = 5.0
    u2, log2 = controller.step(0.1, ref2, x_hat1)
    assert u2 == u1
    assert log2.error == log1.error

    u3, log3 = controller.step(0.2, ref1, x_hat1)
    assert u3 != u1
    assert log3.integral > log1.integral


def test_invalid_simulation_config_non_integer_multiple() -> None:
    """Test that a ValueError is raised when sample times are not integer multiples."""
    dynamics = LinearDynamics(dt=0.1, a=[[1]], b=[[1]])
    output = LinearOutput(dt=0.1, c=[[1]], d=[[0]])
    reference = StepReference(dt=0.1)
    sensor = GaussianSensor(dt=0.1)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIDController(dt=0.15, kp=[[1]], ki=[[0]], kd=[[0]])

    with pytest.raises(ValueError, match="must be an integer multiple"):
        Simulation(
            t_end=1.0,
            dynamics=dynamics,
            output=output,
            reference=reference,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
        )


def test_floating_point_precision_handling() -> None:
    """Test that precision issues (e.g. 0.3 / 0.1) are handled properly."""
    dynamics = LinearDynamics(dt=0.1, a=[[1]], b=[[1]])
    output = LinearOutput(dt=0.1, c=[[1]], d=[[0]])
    reference = StepReference(dt=0.1)
    sensor = GaussianSensor(dt=0.1)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIDController(dt=0.3, kp=[[1]], ki=[[0]], kd=[[0]])

    sim = Simulation(
        t_end=1.0,
        dynamics=dynamics,
        output=output,
        reference=reference,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
    )
    assert sim.controller.dt == 0.3


def test_step_reference_trajectory() -> None:
    """Test StepReference trajectory generation for horizon > 1."""
    dt = 0.1
    horizon = 5
    start_time = 0.5
    step_value = 2.0
    ref_gen = StepReference(dt=dt, step_value=step_value, start_time=start_time, horizon=horizon)

    res, log = ref_gen.step(0.0)
    assert isinstance(res, np.ndarray)
    assert res.shape == (5,)
    assert np.all(res == 0.0)
    assert log.horizon == 5

    res, log = ref_gen.step(0.3)
    expected = np.array([0.0, 0.0, 2.0, 2.0, 2.0])
    assert np.allclose(res, expected)

    res, log = ref_gen.step(0.6)
    assert np.all(res == 2.0)


def test_simulation_execution_and_logging() -> None:
    """Test full simulation execution, ensuring correct loop length and log aggregation."""
    dynamics = LinearDynamics(dt=0.1, a=[[0.9]], b=[[1.0]])
    output = LinearOutput(dt=0.1, c=[[1.0]], d=[[0.0]])
    reference = StepReference(dt=0.1, start_time=0.5)
    sensor = GaussianSensor(dt=0.1, std_dev=0.0)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIDController(dt=0.2, kp=[[0.5]], ki=[[0.1]], kd=[[0.0]])

    sim = Simulation(
        t_end=1.0,
        dynamics=dynamics,
        output=output,
        reference=reference,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
    )
    sim.run()

    assert len(sim.logger.universal_logs) == 11

    assert len(sim.logger.component_logs["dynamics"]) == 11
    assert len(sim.logger.component_logs["reference"]) == 11
    assert len(sim.logger.component_logs["sensor"]) == 11
    assert len(sim.logger.component_logs["estimator"]) == 11
    assert len(sim.logger.component_logs["controller"]) == 11

    assert sim.logger.universal_logs[0]["t"] == 0.0
    assert sim.logger.universal_logs[0]["u"] == 0.0

    assert math.isclose(sim.logger.universal_logs[-1]["t"], 1.0, rel_tol=1e-9)
    assert sim.logger.universal_logs[-1]["u"] != 0.0
