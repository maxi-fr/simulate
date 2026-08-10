"""End-to-end smoke test for the Phase-6 nadir-pointing YAML configuration."""

from pathlib import Path

import numpy as np

from simulate.simulation import Simulation
from spacecraft.frames import lvlh_from_orbit
from spacecraft.quaternion import Quaternion

_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "03_satellite" / "quat_feedback.yaml"


def _nadir_angle(x: np.ndarray) -> float:
    """Body-vs-nadir geodesic angle [rad] from a full rigid-body state."""
    q_err = Quaternion.from_array(x[6:10]).error_to(lvlh_from_orbit(x[0:3], x[3:6]))
    return float(2.0 * np.arctan2(np.linalg.norm(q_err.vec), abs(q_err.scalar)))


def test_nadir_config_builds_and_runs() -> None:
    """`Simulation.from_yaml` builds the full satellite stack and runs without error."""
    sim = Simulation.from_yaml(_CONFIG)
    sim.t_end = 4.0  # a few base steps -- the example runs a full orbit

    sim.run()

    assert sim.logger is not None
    r = sim.logger.signal("estimator", "r")[-1]
    v = sim.logger.signal("estimator", "v")[-1]
    q = sim.logger.signal("estimator", "q")[-1]
    omega = sim.logger.signal("estimator", "omega")[-1]
    b_body = sim.logger.signal("estimator", "b_field_body")[-1]
    h_wheel = sim.logger.signal("estimator", "wheel_momentum")[-1]
    x_hat = np.concatenate([r, v, q, omega, b_body, h_wheel])
    assert x_hat.shape == (19,)  # [r, v, q, omega, b_body, h_wheel]
    assert np.all(np.isfinite(x_hat))
    x_last = sim.logger.signal("dynamics", "x")[-1]
    assert np.all(np.isfinite(x_last))


def test_nadir_config_drives_toward_nadir() -> None:
    """The full stack (estimator + quaternion feedback) acquires and holds nadir under disturbances."""
    sim = Simulation.from_yaml(_CONFIG)
    sim.t_end = 60.0

    sim.run()

    assert sim.logger is not None
    x_all = sim.logger.signal("dynamics", "x")
    angle0 = _nadir_angle(x_all[0])
    angle_end = _nadir_angle(x_all[-1])
    assert angle0 > np.deg2rad(10.0)  # starts well off nadir
    assert angle_end < np.deg2rad(3.0)  # acquires and holds nadir
