from simulate.config import ControllerConfig, PlantConfig
from simulate.controller import Controller
from simulate.plant import DiscretePlant


def test_plant_step_logic():
    """Test standard plant update dynamics."""
    config = PlantConfig(dt=0.1)
    plant = DiscretePlant(config)

    # Initial state should be 0.0
    assert plant.x == 0.0

    # Step 1: u=1.0 -> x_1 = 0.9*0 + 1.0*1.0 = 1.0
    y, log = plant.step(0.0, 1.0)
    assert y == 1.0
    assert log.x == 1.0

    # Step 2: u=0.5 -> x_2 = 0.9*1.0 + 1.0*0.5 = 1.4
    y, log = plant.step(0.1, 0.5)
    assert y == 1.4
    assert log.x == 1.4


def test_controller_step_logic():
    """Test PI controller behavior and integration accumulation."""
    config = ControllerConfig(dt=0.1)
    controller = Controller(config)

    # Step 1: ref=1.0, y=0.0 -> error=1.0
    # integral = 0 + 1.0*0.1 = 0.1
    # u = kp*1.0 + ki*0.1 = 0.5*1.0 + 0.1*0.1 = 0.51
    u, log = controller.step(0.0, 1.0, 0.0)
    assert u == 0.51
    assert log.error == 1.0
    assert log.integral == 0.1


def test_component_zoh_behavior():
    """Test that a component retains its last output between scheduled updates."""
    config = ControllerConfig(dt=0.2)
    controller = Controller(config)

    # Time 0.0: Update should happen
    u1, log1 = controller.step(0.0, 1.0, 0.0)
    assert u1 == 0.52  # dt=0.2 -> int=0.2 -> u = 0.5 + 0.02 = 0.52

    # Time 0.1: No update should happen (ZOH)
    # Even if inputs change, output should remain identical
    u2, log2 = controller.step(0.1, 5.0, 0.0)
    assert u2 == u1
    assert log2.error == log1.error

    # Time 0.2: Update should happen again
    u3, log3 = controller.step(0.2, 1.0, 0.0)
    assert u3 != u1
    assert log3.integral > log1.integral
