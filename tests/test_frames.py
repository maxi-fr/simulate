import numpy as np

from spacecraft.frames import (
    eci_attitude_from_lvlh,
    euler_from_quaternion,
    lvlh_from_orbit,
    orbital_rate,
    quaternion_from_euler,
)
from spacecraft.quaternion import Quaternion

# A simple circular equatorial orbit: position on +x, velocity on +y.
_RADIUS = 7.0e6  # m
_SPEED = 7.5e3  # m/s
_R_ECI = np.array([_RADIUS, 0.0, 0.0])
_V_ECI = np.array([0.0, _SPEED, 0.0])


def test_lvlh_nadir_points_at_earth() -> None:
    q = lvlh_from_orbit(_R_ECI, _V_ECI)

    # The LVLH z axis (3rd row of the inertial->LVLH matrix) is nadir == -r_hat.
    nadir_inertial = -_R_ECI / np.linalg.norm(_R_ECI)
    z_lvlh_in_body = q.apply(nadir_inertial)
    np.testing.assert_allclose(z_lvlh_in_body, np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_lvlh_matrix_is_orthonormal_right_handed() -> None:
    matrix = lvlh_from_orbit(_R_ECI, _V_ECI).to_rot_mat()

    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-9)


def test_orbital_rate_matches_mean_motion() -> None:
    omega_lvlh = orbital_rate(_R_ECI, _V_ECI)

    expected_magnitude = _SPEED / _RADIUS  # circular orbit mean motion
    np.testing.assert_allclose(np.linalg.norm(omega_lvlh), expected_magnitude, rtol=1e-12)
    # Orbital angular velocity (+orbit normal) maps onto the LVLH -y (pitch) axis.
    np.testing.assert_allclose(omega_lvlh, np.array([0.0, -expected_magnitude, 0.0]), atol=1e-9)


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


def test_eci_attitude_from_lvlh_nadir_at_rest() -> None:
    # Zero LVLH-relative attitude and rate => body aligned with LVLH, rate == orbital feedforward.
    q_bi, omega = eci_attitude_from_lvlh(_R_ECI, _V_ECI, roll=0.0, pitch=0.0, yaw=0.0, omega_bo=np.zeros(3))

    q_bo = q_bi * lvlh_from_orbit(_R_ECI, _V_ECI).conjugate()
    np.testing.assert_allclose(q_bo.to_rot_mat(), np.eye(3), atol=1e-9)
    np.testing.assert_allclose(omega, orbital_rate(_R_ECI, _V_ECI), atol=1e-9)


def test_eci_attitude_from_lvlh_round_trips_lvlh_attitude() -> None:
    # The LVLH-relative attitude/rate fed in are recovered from the resulting inertial state.
    roll, pitch, yaw = 5.0, -12.0, 30.0
    omega_bo = np.array([0.01, -0.02, 0.03])  # deg/s, body wrt LVLH
    q_bi, omega = eci_attitude_from_lvlh(
        _R_ECI, _V_ECI, roll=roll, pitch=pitch, yaw=yaw, omega_bo=omega_bo, degrees=True
    )

    q_bo = q_bi * lvlh_from_orbit(_R_ECI, _V_ECI).conjugate()
    pitch_b, roll_b, yaw_b = euler_from_quaternion(q_bo, degrees=True)
    np.testing.assert_allclose([roll_b, pitch_b, yaw_b], [roll, pitch, yaw], atol=1e-9)

    omega_bo_recovered = np.rad2deg(omega - q_bo.apply(orbital_rate(_R_ECI, _V_ECI)))
    np.testing.assert_allclose(omega_bo_recovered, omega_bo, atol=1e-9)


def test_error_to_small_rotation_axis() -> None:
    reference = Quaternion(np.zeros(3), 1.0)  # identity
    angle = 1e-3
    current = Quaternion(np.array([np.sin(angle / 2), 0.0, 0.0]), np.cos(angle / 2))

    q_err = current.error_to(reference)

    # Vector part is ~ (angle / 2) along the rotation axis (+x).
    np.testing.assert_allclose(q_err.vec, np.array([angle / 2, 0.0, 0.0]), atol=1e-6)
    assert q_err.scalar > 0


def test_error_to_left_perturbation_nonidentity_reference() -> None:
    # With a non-identity reference the multiplication order matters. The body-frame error
    # is the LEFT perturbation `dq` in `current = dq (x) reference`, so
    # `current.error_to(reference) == dq == current (x) reference^-1`.
    reference = quaternion_from_euler(np.array([0.2, 0.5, -0.3]))
    angle = 0.1
    dq = Quaternion(np.array([0.0, 0.0, np.sin(angle / 2)]), np.cos(angle / 2))
    current = dq * reference

    q_err = current.error_to(reference)

    # Recovers the left perturbation: vector part ~ (angle / 2) about +z.
    np.testing.assert_allclose(q_err.vec, dq.vec, atol=1e-12)
    np.testing.assert_allclose(q_err.vec, np.array([0.0, 0.0, np.sin(angle / 2)]), atol=1e-12)

    # The reversed ordering (q_ref^-1 (x) q) gives a genuinely different error here.
    reversed_err = reference.conjugate() * current
    assert not np.allclose(reversed_err.vec, q_err.vec, atol=1e-6)
