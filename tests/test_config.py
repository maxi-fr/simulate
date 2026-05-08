import pytest
from pydantic import ValidationError

from simulate.config import ControllerConfig, PlantConfig, SimulationConfig


def test_valid_simulation_config():
    """Test that a valid configuration with integer multiple sample times is accepted."""
    plant_cfg = PlantConfig(dt=0.1)
    controller_cfg = ControllerConfig(dt=0.2)

    sim_cfg = SimulationConfig(plant=plant_cfg, controller=controller_cfg, t_end=1.0)

    assert sim_cfg.plant.dt == 0.1
    assert sim_cfg.controller.dt == 0.2
    assert sim_cfg.t_end == 1.0


def test_invalid_simulation_config_non_integer_multiple():
    """Test that a ValueError is raised when sample times are not integer multiples."""
    plant_cfg = PlantConfig(dt=0.1)
    # 0.15 / 0.1 = 1.5 (not an integer)
    controller_cfg = ControllerConfig(dt=0.15)

    with pytest.raises(ValidationError) as exc_info:
        SimulationConfig(plant=plant_cfg, controller=controller_cfg, t_end=1.0)

    assert "must be an integer multiple" in str(exc_info.value)


def test_floating_point_precision_handling():
    """Test that precision issues (e.g. 0.3 / 0.1) are handled properly."""
    plant_cfg = PlantConfig(dt=0.1)
    # 0.3 / 0.1 might evaluate to 2.9999999999999996 in python
    controller_cfg = ControllerConfig(dt=0.3)

    # Should not raise an error
    sim_cfg = SimulationConfig(plant=plant_cfg, controller=controller_cfg, t_end=1.0)
    assert sim_cfg.controller.dt == 0.3
