import pytest
from pydantic import ValidationError
import numpy as np

from simulate.config import PIDControllerConfig, LinearPlantConfig, SimulationConfig
from simulate.controller import PIDController
from simulate.plant import LinearPlant
from simulate.simulation import Simulation

def get_test_plant_config(dt=0.1):
    return LinearPlantConfig(
        dt=dt,
        a=[[0.9]],
        b=[[1.0]],
        c=[[1.0]],
        d=[[0.0]]
    )

def get_test_controller_config(dt=0.2):
    return PIDControllerConfig(
        dt=dt,
        kp=[[0.5]],
        ki=[[0.1]],
        kd=[[0.0]]
    )

def test_plant_step_logic():
    """Test standard plant update dynamics."""
    config = get_test_plant_config(dt=0.1)
    plant = LinearPlant(config)

    # Initial state should be 0.0
    assert plant.x[0, 0] == 0.0

    # Step 1: u=1.0 -> x_1 = 0.9*0 + 1.0*1.0 = 1.0
    u1 = np.array([[1.0]])
    y, log = plant.step(0.0, u1)
    assert y[0, 0] == 1.0
    assert log.x[0, 0] == 1.0

    # Step 2: u=0.5 -> x_2 = 0.9*1.0 + 1.0*0.5 = 1.4
    u2 = np.array([[0.5]])
    y, log = plant.step(0.1, u2)
    assert y[0, 0] == 1.4
    assert log.x[0, 0] == 1.4

def test_controller_step_logic():
    """Test PI controller behavior and integration accumulation."""
    config = get_test_controller_config(dt=0.1)
    controller = PIDController(config)

    # Step 1: ref=1.0, y=0.0 -> error=1.0
    # integral = 0 + 1.0*0.1 = 0.1
    # u = kp*1.0 + ki*0.1 + kd*10 = 0.5*1.0 + 0.1*0.1 = 0.51
    ref = np.array([[1.0]])
    y = np.array([[0.0]])
    u, log = controller.step(0.0, ref, y)
    assert np.isclose(u[0, 0], 0.51)
    assert log.error[0, 0] == 1.0
    assert log.integral[0, 0] == 0.1

def test_component_zoh_behavior():
    """Test that a component retains its last output between scheduled updates."""
    config = get_test_controller_config(dt=0.2)
    controller = PIDController(config)

    # Time 0.0: Update should happen
    ref1 = np.array([[1.0]])
    y1 = np.array([[0.0]])
    u1, log1 = controller.step(0.0, ref1, y1)
    assert np.isclose(u1[0, 0], 0.52)  # dt=0.2 -> int=0.2 -> u = 0.5 + 0.02 = 0.52

    # Time 0.1: No update should happen (ZOH)
    # Even if inputs change, output should remain identical
    ref2 = np.array([[5.0]])
    u2, log2 = controller.step(0.1, ref2, y1)
    assert u2[0, 0] == u1[0, 0]
    assert log2.error[0, 0] == log1.error[0, 0]

    # Time 0.2: Update should happen again
    u3, log3 = controller.step(0.2, ref1, y1)
    assert u3[0, 0] != u1[0, 0]
    assert log3.integral[0, 0] > log1.integral[0, 0]

def test_valid_simulation_config():
    """Test that a valid configuration with integer multiple sample times is accepted."""
    plant_cfg = get_test_plant_config(dt=0.1)
    controller_cfg = get_test_controller_config(dt=0.2)

    sim_cfg = SimulationConfig(plant=plant_cfg, controller=controller_cfg, t_end=1.0)

    assert sim_cfg.plant.dt == 0.1
    assert sim_cfg.controller.dt == 0.2
    assert sim_cfg.t_end == 1.0

def test_invalid_simulation_config_non_integer_multiple():
    """Test that a ValueError is raised when sample times are not integer multiples."""
    plant_cfg = get_test_plant_config(dt=0.1)
    controller_cfg = get_test_controller_config(dt=0.15)

    with pytest.raises(ValidationError) as exc_info:
        SimulationConfig(plant=plant_cfg, controller=controller_cfg, t_end=1.0)

    assert "must be an integer multiple" in str(exc_info.value)

def test_floating_point_precision_handling():
    """Test that precision issues (e.g. 0.3 / 0.1) are handled properly."""
    plant_cfg = get_test_plant_config(dt=0.1)
    controller_cfg = get_test_controller_config(dt=0.3)

    # Should not raise an error
    sim_cfg = SimulationConfig(plant=plant_cfg, controller=controller_cfg, t_end=1.0)
    assert sim_cfg.controller.dt == 0.3

def test_simulation_execution_and_logging():
    """Test full simulation execution, ensuring correct loop length and log aggregation."""
    sim_cfg = SimulationConfig(
        plant=get_test_plant_config(dt=0.1),
        controller=get_test_controller_config(dt=0.2), # Multi-rate controller
        t_end=1.0
    )

    sim = Simulation(sim_cfg)
    sim.run()

    # Check that universal logs have 11 entries
    assert len(sim.logger.universal_logs) == 11

    # Check that component logs have 11 entries
    assert len(sim.logger.component_logs["plant"]) == 11
    assert len(sim.logger.component_logs["controller"]) == 11

    # First step validation (t=0.0)
    assert sim.logger.universal_logs[0]["t"] == 0.0
    assert np.all(sim.logger.universal_logs[0]["u"] == 0.0)

    # Final step validation (t=1.0)
    import math
    assert math.isclose(sim.logger.universal_logs[-1]["t"], 1.0, rel_tol=1e-9)
    assert np.any(sim.logger.universal_logs[-1]["u"] != 0.0)
