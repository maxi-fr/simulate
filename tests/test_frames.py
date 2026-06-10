import numpy as np

from rigid_body.frames import (
    euler_from_quaternion,
    orbital_rate,
    orc_from_orbit,
    quaternion_from_euler,
)
from rigid_body.quaternion import Quaternion

# A simple circular equatorial orbit: position on +x, velocity on +y.
_RADIUS = 7.0e6  # m
_SPEED = 7.5e3  # m/s
_R_ECI = np.array([_RADIUS, 0.0, 0.0])
_V_ECI = np.array([0.0, _SPEED, 0.0])


def test_orc_nadir_points_at_earth() -> None:
    q = orc_from_orbit(_R_ECI, _V_ECI)

    # The ORC z axis (3rd row of the inertial->ORC matrix) is nadir == -r_hat.
    nadir_inertial = -_R_ECI / np.linalg.norm(_R_ECI)
    z_orc_in_body = q.apply(nadir_inertial)
    np.testing.assert_allclose(z_orc_in_body, np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_orc_matrix_is_orthonormal_right_handed() -> None:
    matrix = orc_from_orbit(_R_ECI, _V_ECI).to_rot_mat()

    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-9)


def test_orbital_rate_matches_mean_motion() -> None:
    omega_orc = orbital_rate(_R_ECI, _V_ECI)

    expected_magnitude = _SPEED / _RADIUS  # circular orbit mean motion
    np.testing.assert_allclose(np.linalg.norm(omega_orc), expected_magnitude, rtol=1e-12)
    # Orbital angular velocity (+orbit normal) maps onto the ORC -y (pitch) axis.
    np.testing.assert_allclose(omega_orc, np.array([0.0, -expected_magnitude, 0.0]), atol=1e-9)


def test_euler_round_trip() -> None:
    angles = np.array([0.3, -0.4, 1.2])  # radians, intrinsic Y-X-Z
    q = quaternion_from_euler(angles)

    np.testing.assert_allclose(euler_from_quaternion(q), angles, atol=1e-9)


def test_euler_round_trip_degrees() -> None:
    q = quaternion_from_euler(np.array([15.0, -40.0, 110.0]), degrees=True)

    angles = euler_from_quaternion(q, degrees=True)
    q_back = quaternion_from_euler(angles, degrees=True)

    # Compare at the rotation-matrix level to avoid quaternion sign ambiguity.
    np.testing.assert_allclose(q_back.to_rot_mat(), q.to_rot_mat(), atol=1e-9)


def test_error_to_identity_for_equal() -> None:
    q = quaternion_from_euler(np.array([0.2, 0.5, -0.3]))
    q_err = q.error_to(q)

    np.testing.assert_allclose(q_err.vec, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(abs(q_err.scalar), 1.0, atol=1e-12)


def test_error_to_small_rotation_axis() -> None:
    reference = Quaternion(np.zeros(3), 1.0)  # identity
    angle = 1e-3
    current = Quaternion(np.array([np.sin(angle / 2), 0.0, 0.0]), np.cos(angle / 2))

    q_err = current.error_to(reference)

    # Vector part is ~ (angle / 2) along the rotation axis (+x).
    np.testing.assert_allclose(q_err.vec, np.array([angle / 2, 0.0, 0.0]), atol=1e-6)
    assert q_err.scalar > 0
