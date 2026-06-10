import numpy as np

from simulate.attitude import (
    normalize_quat,
    quat_kinematics_matrix,
    quat_to_rotation_matrix,
    skew,
)


def test_identity_quaternion_gives_identity_rotation() -> None:
    """The identity quaternion maps to the identity rotation matrix."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_to_rotation_matrix(q), np.eye(3))


def test_rotation_matrix_is_orthonormal() -> None:
    """A unit quaternion yields a proper rotation (R R^T = I, det = +1)."""
    q = normalize_quat(np.array([0.3, -0.5, 0.2, 0.78]))
    rot = quat_to_rotation_matrix(q)
    assert np.allclose(rot @ rot.T, np.eye(3))
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_known_90_degree_rotation_about_z() -> None:
    """A +90 deg rotation about z maps body x-axis to inertial y-axis."""
    half = np.pi / 4
    q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    body_x = np.array([1.0, 0.0, 0.0])
    assert np.allclose(quat_to_rotation_matrix(q) @ body_x, np.array([0.0, 1.0, 0.0]))


def test_skew_matches_cross_product() -> None:
    """skew(a) @ b equals a x b."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-4.0, 5.0, 6.0])
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_kinematics_matrix_matches_analytic_derivative() -> None:
    """0.5 * Omega(omega) @ q matches the closed-form derivative of a z-axis rotation."""
    wz = 0.7
    omega = np.array([0.0, 0.0, wz])
    theta = 0.4  # half-angle 0.2
    half = theta / 2
    q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    q_dot = 0.5 * quat_kinematics_matrix(omega) @ q
    expected = np.array([-(wz / 2) * np.sin(half), 0.0, 0.0, (wz / 2) * np.cos(half)])
    assert np.allclose(q_dot, expected)
