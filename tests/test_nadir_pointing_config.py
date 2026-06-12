"""End-to-end smoke test for the Phase-6 nadir-pointing YAML configuration."""

from pathlib import Path

import numpy as np

from rigid_body.frames import orc_from_orbit
from rigid_body.quaternion import Quaternion
from simulate.simulation import Simulation

_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "03_nadir_pointing.yaml"


def _nadir_angle(x: np.ndarray) -> float:
    """Body-vs-nadir geodesic angle [rad] from a full rigid-body state."""
    q_err = Quaternion.from_array(x[6:10]).error_to(orc_from_orbit(x[0:3], x[3:6]))
    return float(2.0 * np.arctan2(np.linalg.norm(q_err.vec), abs(q_err.scalar)))


def test_nadir_config_builds_and_runs() -> None:
    """`Simulation.from_yaml` builds the full satellite stack and runs without error."""
    sim = Simulation.from_yaml(_CONFIG)
    sim.t_end = 4.0  # a few base steps -- the example runs a full orbit

    sim.run()

    logs = sim.logger.universal_logs
    assert len(logs) > 0
    x_hat = np.asarray(logs[-1]["x_hat"])
    assert x_hat.shape == (19,)  # [r, v, q, omega, b_body, h_wheel]
    assert np.all(np.isfinite(x_hat))
    assert np.all(np.isfinite(np.asarray(logs[-1]["x"])))


def test_nadir_config_drives_toward_nadir() -> None:
    """The full stack (estimator + quaternion feedback) acquires and holds nadir under disturbances."""
    sim = Simulation.from_yaml(_CONFIG)
    sim.t_end = 60.0

    sim.run()

    logs = sim.logger.universal_logs
    angle0 = _nadir_angle(np.asarray(logs[0]["x"]))
    angle_end = _nadir_angle(np.asarray(logs[-1]["x"]))
    assert angle0 > np.deg2rad(10.0)  # starts well off nadir
    assert angle_end < np.deg2rad(3.0)  # acquires and holds nadir
