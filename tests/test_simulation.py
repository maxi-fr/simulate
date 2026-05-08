from simulate.config import ControllerConfig, PlantConfig, SimulationConfig
from simulate.simulation import Simulation


def test_simulation_execution_and_logging():
    """Test full simulation execution, ensuring correct loop length and log aggregation."""
    # dt=0.1, t_end=1.0 -> 11 steps (0.0 to 1.0 inclusive)
    sim_cfg = SimulationConfig(
        plant=PlantConfig(dt=0.1),
        controller=ControllerConfig(dt=0.2), # Multi-rate controller
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
    # Ref=0.0 (step is at t>=1.0), y=0, u=0
    assert sim.logger.universal_logs[0]["t"] == 0.0
    assert sim.logger.universal_logs[0]["u"] == 0.0

    # Final step validation (t=1.0)
    import math
    assert math.isclose(sim.logger.universal_logs[-1]["t"], 1.0, rel_tol=1e-9)
    assert sim.logger.universal_logs[-1]["u"] != 0.0  # Ref becomes 1.0 at t=1.0, u should react
