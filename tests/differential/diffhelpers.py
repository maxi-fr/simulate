"""Pure input generators shared by the differential port tests.

These helpers depend only on numpy/scipy so they can be imported from any test module without
pulling in either codebase. Quaternion arrays are scalar-last ``[x, y, z, w]`` to match both the
old and new ``Quaternion`` layout.
"""

import numpy as np
from scipy.spatial.transform import Rotation

_LEO_RADIUS = 7.0e6  # representative LEO orbit radius [m]
_LEO_SPEED = 7.5e3  # representative LEO orbital speed [m/s]


def rand_unit_vec(rng: np.random.Generator) -> np.ndarray:
    """Random unit 3-vector."""
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def rand_quat_array(rng: np.random.Generator) -> np.ndarray:
    """Random unit quaternion as a scalar-last ``[x, y, z, w]`` array."""
    return Rotation.random(rng=rng).as_quat(scalar_first=False)


def rand_inertia(rng: np.random.Generator) -> np.ndarray:
    """Random symmetric positive-definite 3x3 inertia tensor [kg*m^2]."""
    principal = rng.uniform(2.0, 20.0, size=3)
    rot = Rotation.random(rng=rng).as_matrix()
    return rot @ np.diag(principal) @ rot.T


def leo_rv(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Representative LEO state ``(r_eci, v_eci)`` with a well-conditioned ``r x v``."""
    r_dir = rand_unit_vec(rng)
    # Velocity direction with a strong component perpendicular to r (so r x v is non-degenerate).
    v_dir = rand_unit_vec(rng)
    v_dir = v_dir - 0.5 * np.dot(v_dir, r_dir) * r_dir
    v_dir /= np.linalg.norm(v_dir)
    return r_dir * _LEO_RADIUS, v_dir * _LEO_SPEED
