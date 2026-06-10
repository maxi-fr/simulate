"""Quaternion and rotation helpers for rigid body attitude dynamics.

Conventions
-----------
Quaternions are scalar-first Hamilton quaternions ``q = [w, x, y, z]`` with unit norm,
representing the rotation that maps **body-frame** vectors into the **inertial frame**.
All vectors are numpy ``(N, 1)`` column vectors, matching the framework convention.
"""

from typing import cast

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric cross-product matrix of a 3-vector ``v`` (shape ``(3, 1)``)."""
    vx, vy, vz = v[0, 0], v[1, 0], v[2, 0]
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ]
    )


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a scalar-first unit quaternion ``(4, 1)`` to its body->inertial rotation matrix ``(3, 3)``."""
    w, x, y, z = q[0, 0], q[1, 0], q[2, 0], q[3, 0]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_kinematics_matrix(omega: np.ndarray) -> np.ndarray:
    """Return the 4x4 matrix ``Omega(omega)`` such that ``q_dot = 0.5 * Omega(omega) @ q``.

    ``omega`` is the body-frame angular velocity ``(3, 1)``.
    """
    wx, wy, wz = omega[0, 0], omega[1, 0], omega[2, 0]
    return np.array(
        [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ]
    )


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Return the unit-norm quaternion for ``q`` (shape ``(4, 1)``)."""
    return cast("np.ndarray", q / np.linalg.norm(q))
