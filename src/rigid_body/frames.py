"""Reference-frame kinematics: the orbital (ORC/LVLH) frame, Euler angles, and orbital rate.

The orbital reference frame (ORC, a local-vertical/local-horizontal frame) used here has the
following axes, expressed in the inertial (ECI) frame::

    z_orc = -r / |r|                  (nadir, points toward the central body)
    y_orc = -(r x v) / |r x v|        (negative orbit normal, the pitch axis)
    x_orc =  y_orc x z_orc            (along-track, the roll axis)

A nadir-pointing body aligns its body frame with this frame, so the desired attitude is the
inertial->ORC rotation (consistent with the ``q_bi`` convention used elsewhere, where
``q_bi.apply`` maps an inertial vector into the body frame) and the desired body rate is the
ORC frame's angular velocity expressed in ORC coordinates.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .quaternion import FloatArray, Quaternion, Vec3

# Intrinsic Euler sequence used throughout (matches the legacy attitude convention).
_EULER_SEQUENCE = "YXZ"


def _eci_to_orc_matrix(r_eci: FloatArray, v_eci: FloatArray) -> FloatArray:
    """Build the inertial->ORC rotation matrix whose rows are the ORC axes in ECI."""
    z = -r_eci / np.linalg.norm(r_eci)
    h = np.cross(r_eci, v_eci)
    y = -h / np.linalg.norm(h)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    return np.vstack((x, y, z))


def orc_from_orbit(r_eci: ArrayLike, v_eci: ArrayLike) -> Quaternion:
    """Desired nadir-pointing attitude (inertial->ORC) for a given orbit state.

    Parameters
    ----------
    r_eci : ArrayLike
        Inertial-frame position vector [m], shape ``(3,)``.
    v_eci : ArrayLike
        Inertial-frame velocity vector [m/s], shape ``(3,)``.

    Returns
    -------
    Quaternion
        The rotation ``q`` such that ``q.apply(v_eci) = v_orc``, i.e. a body frame aligned
        with this quaternion points its ``z`` axis at nadir.
    """
    matrix = _eci_to_orc_matrix(np.asarray(r_eci, dtype=float), np.asarray(v_eci, dtype=float))
    return Quaternion.from_scipy(Rotation.from_matrix(matrix))


def orbital_rate(r_eci: ArrayLike, v_eci: ArrayLike) -> Vec3:
    """Angular velocity of the ORC frame, expressed in ORC coordinates.

    For a nadir-pointing body this is the feedforward desired body rate. The inertial-frame
    orbital angular velocity is ``omega = (r x v) / |r|**2`` (magnitude equals the mean
    motion for a circular orbit); it is rotated into the ORC frame before being returned.

    Parameters
    ----------
    r_eci : ArrayLike
        Inertial-frame position vector [m], shape ``(3,)``.
    v_eci : ArrayLike
        Inertial-frame velocity vector [m/s], shape ``(3,)``.

    Returns
    -------
    numpy.ndarray
        The desired body angular velocity [rad/s], shape ``(3,)``.
    """
    r = np.asarray(r_eci, dtype=float)
    v = np.asarray(v_eci, dtype=float)
    omega_eci = np.cross(r, v) / np.dot(r, r)
    return _eci_to_orc_matrix(r, v) @ omega_eci


def euler_from_quaternion(q: Quaternion, *, degrees: bool = False) -> Vec3:
    """Intrinsic ``Y-X-Z`` Euler angles of the rotation represented by ``q``.

    Parameters
    ----------
    q : Quaternion
        The rotation to decompose.
    degrees : bool, optional
        Return angles in degrees instead of radians, by default ``False``.

    Returns
    -------
    numpy.ndarray
        The ``[Y, X, Z]`` Euler angles, shape ``(3,)``.
    """
    return q.to_scipy().as_euler(_EULER_SEQUENCE, degrees=degrees)


def quaternion_from_euler(angles: ArrayLike, *, degrees: bool = False) -> Quaternion:
    """Quaternion from intrinsic ``Y-X-Z`` Euler angles (inverse of :func:`euler_from_quaternion`).

    Parameters
    ----------
    angles : ArrayLike
        The ``[Y, X, Z]`` Euler angles, shape ``(3,)``.
    degrees : bool, optional
        Interpret ``angles`` as degrees instead of radians, by default ``False``.

    Returns
    -------
    Quaternion
        The corresponding rotation.
    """
    rot = Rotation.from_euler(_EULER_SEQUENCE, np.asarray(angles, dtype=float), degrees=degrees)
    return Quaternion.from_scipy(rot)
