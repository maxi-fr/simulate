import math

import numpy as np
import pytest
from scipy.signal import place_poles

from simulate.component import NoLog
from simulate.controller import PIController
from simulate.dynamics import LinearDynamics
from simulate.estimator import IdentityEstimator, LuenbergerObserver
from simulate.integrator import rk4
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor, LinearMeasurement, RandomWalkBiasSensor, full_state_measurement
from simulate.simulation import Simulation


def test_plant_step_logic() -> None:
    """Test standard plant update dynamics."""
    dynamics = LinearDynamics(dt=0.1, A=[[0.9]], B=[[1.0]])
    measurement = LinearMeasurement(C=[[1.0]], D=[[0.0]])

    assert dynamics.x[0] == 0.0

    u1 = 1.0
    x, _dynamics_log = dynamics.evaluate(0.0, 1.0)
    y = measurement(0.0, x, u1)
    assert np.allclose(y, 1.0)
    assert np.allclose(x, 1.0)

    u2 = 0.5
    x, _dynamics_log = dynamics.evaluate(0.1, u2)
    y = measurement(0.1, x, u2)
    assert np.allclose(y, 1.4)
    assert np.allclose(x, 1.4)


def test_sensor_step_logic() -> None:
    """Test sensor behavior with Gaussian noise."""
    sensor = GaussianSensor(dt=0.1, measurement=full_state_measurement, std_dev=0.0)
    y = 1.0
    y_mea, log = sensor.evaluate(0.0, y, 0.0)
    assert np.allclose(y_mea, 1.0)
    assert np.allclose(log.truth, 1.0)
    assert np.allclose(log.noise, 0.0)

    sensor_noise = GaussianSensor(dt=0.1, measurement=full_state_measurement, std_dev=0.1)
    y_mea2, log2 = sensor_noise.evaluate(0.0, y, 0.0)
    assert not np.allclose(y_mea2, 1.0)
    assert not np.allclose(log2.noise, 0.0)


def test_estimator_step_logic() -> None:
    """Test identity estimator behavior."""
    estimator = IdentityEstimator(dt=0.1)
    y_mea = 1.2
    u = 0.5
    x_hat, _log = estimator.evaluate(0.0, y_mea, u)
    assert np.allclose(x_hat, 1.2)


def test_luenberger_observer_reconstructs_unmeasured_state() -> None:
    """The observer reconstructs the unmeasured second state from a first-state-only measurement."""
    dt = 0.001
    a = np.array([[-1.0, 50.0], [-5.0, -100.0]])
    b = np.array([[0.0], [100.0]])
    c = np.array([[1.0, 0.0]])
    gain_l = place_poles(a.T, c.T, [-200.0, -300.0]).gain_matrix.T

    plant = LinearDynamics(dt=dt, A=a, B=b, integrator=rk4)
    plant.x = np.array([1.0, -0.5])  # nonzero initial state, unknown to the observer (starts at zero)
    sensor = GaussianSensor(dt=dt, measurement=LinearMeasurement(C=c, D=[[0.0]]), std_dev=0.0)
    observer = LuenbergerObserver(dt=dt, A=a, B=b, C=c, L=gain_l, integrator=rk4)

    u = np.array([0.2])
    x_hat: float | np.ndarray = np.zeros(2)
    for k in range(400):
        t = k * dt
        y_mea, _ = sensor.evaluate(t, plant.x, u)
        x_hat, _ = observer.evaluate(t, y_mea, u)
        plant.evaluate(t, u)

    # Both states, including the unmeasured second component, converge to the truth.
    assert np.linalg.norm(np.asarray(x_hat) - plant.x) < 1e-2


def test_controller_step_logic() -> None:
    """Test PI controller behavior and integration accumulation."""
    controller = PIController(dt=0.1, kp=[[0.5]], ki=[[0.1]])

    ref = 1.0
    x_hat = 0.0
    u, log = controller.evaluate(0.0, ref, x_hat)
    assert np.isclose(float(np.asarray(u).item()), 0.51)
    assert np.allclose(log.error, 1.0)
    assert np.allclose(log.integral, 0.1)


def test_pi_controller_matrix_gain_feeds_back_derivative_state() -> None:
    """A (1, 2) kp column lets the PI controller feed back an estimated derivative state."""
    controller = PIController(dt=0.1, kp=[[1.0, 0.5]], ki=[[2.0, 0.0]])

    ref = np.array([1.0, 0.0])
    x_hat = np.array([0.0, 0.4])  # [measured output, estimated derivative state]
    u, log = controller.evaluate(0.0, ref, x_hat)

    # error = ref - x_hat = [1.0, -0.4]; integral = error * dt = [0.1, -0.04]
    # u = kp @ error + ki @ integral = (1*1 + 0.5*-0.4) + (2*0.1 + 0*-0.04) = 0.8 + 0.2 = 1.0
    assert np.isclose(float(np.asarray(u).item()), 1.0)
    assert np.allclose(log.error, [1.0, -0.4])
    assert np.allclose(log.integral, [0.1, -0.04])


def test_component_zoh_behavior() -> None:
    """Test that a component retains its last output between scheduled updates."""
    controller = PIController(dt=0.2, kp=[[0.5]], ki=[[0.1]])

    ref1 = 1.0
    x_hat1 = 0.0
    u1, log1 = controller.evaluate(0.0, ref1, x_hat1)
    assert np.isclose(float(np.asarray(u1).item()), 0.52)

    ref2 = 5.0
    u2, log2 = controller.evaluate(0.1, ref2, x_hat1)
    assert np.allclose(u2, u1)
    assert np.allclose(log2.error, log1.error)

    u3, log3 = controller.evaluate(0.2, ref1, x_hat1)
    assert not np.allclose(u3, u1)
    assert np.all(log3.integral > log1.integral)


def test_invalid_simulation_config_non_integer_multiple() -> None:
    """Test that a ValueError is raised when sample times are not integer multiples."""
    dynamics = LinearDynamics(dt=0.1, A=[[1]], B=[[1]])
    reference = StepReference(dt=0.1)
    sensor = GaussianSensor(dt=0.1, measurement=full_state_measurement)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIController(dt=0.15, kp=[[1]], ki=[[0]])

    with pytest.raises(ValueError, match="must be an integer multiple"):
        Simulation(
            t_end=1.0,
            dynamics=dynamics,
            reference=reference,
            sensors=[sensor],
            estimator=estimator,
            controller=controller,
        )


def test_floating_point_precision_handling() -> None:
    """Test that precision issues (e.g. 0.3 / 0.1) are handled properly."""
    dynamics = LinearDynamics(dt=0.1, A=[[1]], B=[[1]])
    reference = StepReference(dt=0.1)
    sensor = GaussianSensor(dt=0.1, measurement=full_state_measurement)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIController(dt=0.3, kp=[[1]], ki=[[0]])

    sim = Simulation(
        t_end=1.0,
        dynamics=dynamics,
        reference=reference,
        sensors=[sensor],
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

    res, _log = ref_gen.evaluate(0.0)
    assert isinstance(res, np.ndarray)
    assert res.shape == (5,)
    assert np.all(res == 0.0)

    res, _log = ref_gen.evaluate(0.3)
    expected = np.array([0.0, 0.0, 2.0, 2.0, 2.0])
    assert np.allclose(res, expected)

    res, _log = ref_gen.evaluate(0.6)
    assert np.all(res == 2.0)


def test_simulation_execution_and_logging() -> None:
    """Test full simulation execution, ensuring correct loop length and log aggregation."""
    dynamics = LinearDynamics(dt=0.1, A=[[0.9]], B=[[1.0]])
    reference = StepReference(dt=0.1, start_time=0.5)
    sensor = GaussianSensor(dt=0.1, measurement=full_state_measurement, std_dev=0.0)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIController(dt=0.2, kp=[[0.5]], ki=[[0.1]])

    sim = Simulation(
        t_end=1.0,
        dynamics=dynamics,
        reference=reference,
        sensors=[sensor],
        estimator=estimator,
        controller=controller,
    )
    sim.run()

    assert len(sim.logger.core_logs) == 11

    assert len(sim.logger.component_logs["dynamics"]) == 11
    assert len(sim.logger.component_logs["reference"]) == 11
    assert len(sim.logger.component_logs["sensor_0"]) == 11
    assert len(sim.logger.component_logs["estimator"]) == 11
    assert len(sim.logger.component_logs["controller"]) == 11

    assert sim.logger.core_logs[0]["t"] == 0.0
    assert np.allclose(sim.logger.core_logs[0]["u"], 0.0)

    assert math.isclose(sim.logger.core_logs[-1]["t"], 1.0, rel_tol=1e-9)
    assert not np.allclose(sim.logger.core_logs[-1]["u"], 0.0)


def test_simulation_single_sensor() -> None:
    """Test that Simulation accepts a single sensor directly instead of a list."""
    dynamics = LinearDynamics(dt=0.1, A=[[0.9]], B=[[1.0]])
    reference = StepReference(dt=0.1, start_time=0.5)
    sensor = GaussianSensor(dt=0.1, measurement=full_state_measurement, std_dev=0.0)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIController(dt=0.2, kp=[[0.5]], ki=[[0.1]])

    sim = Simulation(
        t_end=1.0,
        dynamics=dynamics,
        reference=reference,
        sensors=sensor,
        estimator=estimator,
        controller=controller,
    )
    sim.run()

    assert len(sim.logger.core_logs) == 11
    assert len(sim.sensors) == 1
    assert sim.sensors[0] is sensor


def test_random_walk_bias_sensor() -> None:
    """Test RandomWalkBiasSensor behavior with zero noise/bias, noise only, and random walk bias."""
    # 1. Zero noise, zero bias
    sensor = RandomWalkBiasSensor(dt=0.1, measurement=full_state_measurement, std_dev_noise=0.0, std_dev_bias=0.0)
    y = np.array([1.0, 2.0])

    # First step (t=0)
    y_mea, log = sensor.evaluate(0.0, y, 0.0)
    assert np.allclose(y_mea, y)
    assert np.allclose(log.truth, y)
    assert np.allclose(log.noise, 0.0)
    assert np.allclose(log.bias, 0.0)

    # Second step (t=0.1)
    y_mea2, log2 = sensor.evaluate(0.1, y, 0.0)
    assert np.allclose(y_mea2, y)
    assert np.allclose(log2.noise, 0.0)
    assert np.allclose(log2.bias, 0.0)

    # 2. Noise only
    sensor_noise = RandomWalkBiasSensor(
        dt=0.1, measurement=full_state_measurement, std_dev_noise=0.1, std_dev_bias=0.0, seed=123
    )
    y_mea_n1, log_n1 = sensor_noise.evaluate(0.0, y, 0.0)
    assert not np.allclose(y_mea_n1, y)
    assert np.allclose(log_n1.bias, 0.0)
    assert not np.allclose(log_n1.noise, 0.0)

    _y_mea_n2, log_n2 = sensor_noise.evaluate(0.1, y, 0.0)
    assert np.allclose(log_n2.bias, 0.0)

    # 3. Bias only
    sensor_bias = RandomWalkBiasSensor(
        dt=0.1, measurement=full_state_measurement, std_dev_noise=0.0, std_dev_bias=0.1, seed=456
    )

    # At t=0, bias is initialized to zero
    y_mea_b1, log_b1 = sensor_bias.evaluate(0.0, y, 0.0)
    assert np.allclose(y_mea_b1, y)
    assert np.allclose(log_b1.bias, 0.0)
    assert np.allclose(log_b1.noise, 0.0)

    # At t=0.1, bias step is added
    y_mea_b2, log_b2 = sensor_bias.evaluate(0.1, y, 0.0)
    assert not np.allclose(y_mea_b2, y)
    assert not np.allclose(log_b2.bias, 0.0)
    assert np.allclose(log_b2.noise, 0.0)
    assert np.allclose(y_mea_b2, y + log_b2.bias)

    # At t=0.2, bias step is added again, changing the bias
    y_mea_b3, log_b3 = sensor_bias.evaluate(0.2, y, 0.0)
    assert not np.allclose(log_b3.bias, log_b2.bias)
    assert np.allclose(y_mea_b3, y + log_b3.bias)

    # 4. from_config
    config = {
        "dt": 0.2,
        "std_dev_noise": 0.5,
        "std_dev_bias": 0.2,
        "seed": 99,
        "measurement": {"class_path": "simulate.sensor.LinearMeasurement", "C": [[1.0]], "D": [[0.0]]},
    }
    sensor_cfg = RandomWalkBiasSensor.from_config(config)
    assert sensor_cfg.dt == 0.2
    assert sensor_cfg.std_dev_noise == 0.5
    assert sensor_cfg.std_dev_bias == 0.2
