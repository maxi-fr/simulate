import math

import numpy as np
import pytest

from simulate.controller import PIDController
from simulate.estimator import IdentityEstimator
from simulate.plant import LinearPlant
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor
from simulate.simulation import Simulation


def test_component_conversion_utilities() -> None:
    """Test to_col_vec and from_col_vec utilities in Component."""
    # We can use LinearPlant as a concrete component to test these static methods
    plant = LinearPlant(dt=0.1, a=[[1]], b=[[1]], c=[[1]], d=[[0]])

    # to_col_vec conversion for float
    res = plant.to_col_vec(1.0)
    assert res.shape == (1, 1)
    assert res[0, 0] == 1.0

    # to_col_vec: 1D array
    res = plant.to_col_vec(np.array([1.0, 2.0]))
    assert res.shape == (2, 1)
    assert res[0, 0] == 1.0
    assert res[1, 0] == 2.0

    # to_col_vec: 2D array (already col vec)
    res = plant.to_col_vec(np.array([[1.0], [2.0]]))
    assert res.shape == (2, 1)

    # from_col_vec: size 1
    res_back = plant.from_col_vec(np.array([[1.0]]))
    assert isinstance(res_back, float)
    assert res_back == 1.0

    # from_col_vec conversion for size > 1
    res_back = plant.from_col_vec(np.array([[1.0], [2.0]]))
    assert isinstance(res_back, np.ndarray)
    assert res_back.shape == (2,)
    assert res_back[0] == 1.0
    assert res_back[1] == 2.0


def test_plant_step_logic() -> None:
    """Test standard plant update dynamics."""
    plant = LinearPlant(dt=0.1, a=[[0.9]], b=[[1.0]], c=[[1.0]], d=[[0.0]])

    # Initial state should be 0.0
    assert plant.x[0, 0] == 0.0

    # Step 1: u=1.0 -> x_1 = 0.9*0 + 1.0*1.0 = 1.0
    u1 = 1.0
    y, log = plant.step(0.0, u1)
    assert y == 1.0
    assert log.x[0, 0] == 1.0

    # Step 2: u=0.5 -> x_2 = 0.9*1.0 + 1.0*0.5 = 1.4
    u2 = 0.5
    y, log = plant.step(0.1, u2)
    assert y == 1.4
    assert log.x[0, 0] == 1.4


def test_sensor_step_logic() -> None:
    """Test sensor behavior with Gaussian noise."""
    # Zero noise
    sensor = GaussianSensor(dt=0.1, std_dev=0.0)
    y = 1.0
    y_mea, log = sensor.step(0.0, y)
    assert y_mea == 1.0
    assert log.noise == 0.0

    # With noise
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

    # Step 1: ref=1.0, x_hat=0.0 -> error=1.0
    # integral = 0 + 1.0*0.1 = 0.1
    # u = kp*1.0 + ki*0.1 + kd*10 = 0.5*1.0 + 0.1*0.1 = 0.51
    ref = 1.0
    x_hat = 0.0
    u, log = controller.step(0.0, ref, x_hat)
    assert math.isclose(u, 0.51)
    assert log.error == 1.0
    assert log.integral == 0.1


def test_component_zoh_behavior() -> None:
    """Test that a component retains its last output between scheduled updates."""
    controller = PIDController(dt=0.2, kp=[[0.5]], ki=[[0.1]], kd=[[0.0]])

    # Time 0.0: Update should happen
    ref1 = 1.0
    x_hat1 = 0.0
    u1, log1 = controller.step(0.0, ref1, x_hat1)
    assert math.isclose(u1, 0.52)  # dt=0.2 -> int=0.2 -> u = 0.5 + 0.02 = 0.52

    # Time 0.1: No update should happen (ZOH)
    # Even if inputs change, output should remain identical
    ref2 = 5.0
    u2, log2 = controller.step(0.1, ref2, x_hat1)
    assert u2 == u1
    assert log2.error == log1.error

    # Time 0.2: Update should happen again
    u3, log3 = controller.step(0.2, ref1, x_hat1)
    assert u3 != u1
    assert log3.integral > log1.integral


def test_invalid_simulation_config_non_integer_multiple() -> None:
    """Test that a ValueError is raised when sample times are not integer multiples."""
    plant = LinearPlant(dt=0.1, a=[[1]], b=[[1]], c=[[1]], d=[[0]])
    reference = StepReference(dt=0.1)
    sensor = GaussianSensor(dt=0.1)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIDController(dt=0.15, kp=[[1]], ki=[[0]], kd=[[0]])

    with pytest.raises(ValueError, match="must be an integer multiple"):
        Simulation(
            t_end=1.0,
            plant=plant,
            reference=reference,
            sensor=sensor,
            estimator=estimator,
            controller=controller,
        )


def test_floating_point_precision_handling() -> None:
    """Test that precision issues (e.g. 0.3 / 0.1) are handled properly."""
    plant = LinearPlant(dt=0.1, a=[[1]], b=[[1]], c=[[1]], d=[[0]])
    reference = StepReference(dt=0.1)
    sensor = GaussianSensor(dt=0.1)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIDController(dt=0.3, kp=[[1]], ki=[[0]], kd=[[0]])

    # Should not raise an error
    sim = Simulation(
        t_end=1.0,
        plant=plant,
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

    # At t=0.0: [0, 0.1, 0.2, 0.3, 0.4] < 0.5 -> all 0.0
    res, log = ref_gen.step(0.0)
    assert isinstance(res, np.ndarray)
    assert res.shape == (5,)
    assert np.all(res == 0.0)
    assert log.horizon == 5

    # At t=0.3: [0.3, 0.4, 0.5, 0.6, 0.7] -> [0, 0, 2, 2, 2]
    res, log = ref_gen.step(0.3)
    expected = np.array([0.0, 0.0, 2.0, 2.0, 2.0])
    assert np.allclose(res, expected)

    # At t=0.6: [0.6, 0.7, 0.8, 0.9, 1.0] -> all 2.0
    res, log = ref_gen.step(0.6)
    assert np.all(res == 2.0)


def test_simulation_execution_and_logging() -> None:
    """Test full simulation execution, ensuring correct loop length and log aggregation."""
    plant = LinearPlant(dt=0.1, a=[[0.9]], b=[[1.0]], c=[[1.0]], d=[[0.0]])
    reference = StepReference(dt=0.1, start_time=0.5)
    sensor = GaussianSensor(dt=0.1, std_dev=0.0)
    estimator = IdentityEstimator(dt=0.1)
    controller = PIDController(dt=0.2, kp=[[0.5]], ki=[[0.1]], kd=[[0.0]])

    sim = Simulation(
        t_end=1.0,
        plant=plant,
        reference=reference,
        sensor=sensor,
        estimator=estimator,
        controller=controller,
    )
    sim.run()

    # Check that universal logs have 11 entries
    assert len(sim.logger.universal_logs) == 11

    # Check that component logs have 11 entries
    assert len(sim.logger.component_logs["plant"]) == 11
    assert len(sim.logger.component_logs["reference"]) == 11
    assert len(sim.logger.component_logs["sensor"]) == 11
    assert len(sim.logger.component_logs["estimator"]) == 11
    assert len(sim.logger.component_logs["controller"]) == 11

    # First step validation (t=0.0)
    assert sim.logger.universal_logs[0]["t"] == 0.0
    assert sim.logger.universal_logs[0]["u"] == 0.0

    # Final step validation (t=1.0)
    assert math.isclose(sim.logger.universal_logs[-1]["t"], 1.0, rel_tol=1e-9)
    assert sim.logger.universal_logs[-1]["u"] != 0.0
