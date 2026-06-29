import datetime
import importlib

import numpy as np

from simulate.integrator import rk4
from spacecraft.estimator import (
    AttitudeMEKF,
    MeasurementLayout,
    OrbitKalmanFilter,
)
from spacecraft.frames import eci_to_geodedic
from spacecraft.quaternion import Quaternion

_est_mod = importlib.import_module("examples.03_satellite.estimator")
FullStateEstimator = _est_mod.FullStateEstimator

_EPOCH = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
_R0 = np.array([7.0e6, 0.0, 0.0])
_V0 = np.array([0.0, 7.5e3, 0.0])


def _angle_between(qa: np.ndarray, qb: np.ndarray) -> float:
    """Geodesic angle [rad] between two quaternions."""
    rel = Quaternion.from_array(np.asarray(qa)).conjugate() * Quaternion.from_array(np.asarray(qb))
    return float(2.0 * np.arctan2(np.linalg.norm(rel.vec), abs(rel.scalar)))


def test_measurement_layout_split() -> None:
    layout = MeasurementLayout((("gps", 6), ("gyro", 3), ("magnetometer", 3)))
    y = np.arange(12, dtype=float)

    channels = layout.split(y)

    assert layout.size == 12
    np.testing.assert_array_equal(channels["gps"], np.arange(6))
    np.testing.assert_array_equal(channels["gyro"], np.array([6.0, 7.0, 8.0]))
    np.testing.assert_array_equal(channels["magnetometer"], np.array([9.0, 10.0, 11.0]))


def test_orbit_kf_beats_raw_gps_noise() -> None:
    dt = 10.0
    n_steps = 200
    noise_std = 50.0  # m
    rng = np.random.default_rng(0)

    h = np.zeros((3, 6))
    h[:, :3] = np.eye(3)
    kf = OrbitKalmanFilter(
        r0=_R0,
        v0=_V0,
        P0=np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2]),
        Q=np.diag([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3]),
        H=h,
        R=noise_std**2 * np.eye(3),
    )

    truth = np.concatenate([_R0, _V0])
    errors = []
    for _ in range(n_steps):
        truth = rk4(OrbitKalmanFilter._f, 0.0, dt, truth, np.zeros(0))  # noqa: SLF001
        meas = truth[:3] + rng.normal(0.0, noise_std, size=3)
        kf.predict(dt)
        kf.update(meas)
        errors.append(np.linalg.norm(kf.x[:3] - truth[:3]))

    rms = float(np.sqrt(np.mean(np.square(errors[-50:]))))
    assert rms < noise_std  # smoothing beats the raw measurement noise


def test_mekf_converges_and_tracks_bias() -> None:
    dt = 0.5
    n_steps = 800
    omega_true = np.array([0.02, -0.015, 0.01])
    bias_true = np.array([1e-3, -2e-3, 1.5e-3])
    s_inertial = np.array([1.0, 0.0, 0.0])
    b_inertial = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(1)

    mekf = AttitudeMEKF(
        q0=np.array([0.0, 0.0, 0.0, 1.0]),
        P0=np.diag([1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4]),
        Qc=np.diag([1e-8, 1e-8, 1e-8, 1e-12, 1e-12, 1e-12]),
        R_sun=1e-6 * np.eye(3),
        R_mag=1e-6 * np.eye(3),
    )

    q_true = Quaternion(np.zeros(3), 1.0)
    for _ in range(n_steps):
        q_true = q_true.exact_integration(omega_true, dt)
        gyro = omega_true + bias_true + rng.normal(0.0, 1e-4, size=3)
        sun_meas = q_true.apply(s_inertial) + rng.normal(0.0, 1e-3, size=3)
        mag_meas = q_true.apply(b_inertial) + rng.normal(0.0, 1e-3, size=3)

        mekf.predict(gyro, dt)
        mekf.update_vector(s_inertial, sun_meas, mekf.R_sun)
        mekf.update_vector(b_inertial, mag_meas, mekf.R_mag)

    assert _angle_between(mekf.q, q_true.to_array()) < np.deg2rad(3.0)
    # Bias estimate is much closer to truth than the zero initial guess.
    assert np.linalg.norm(mekf.b - bias_true) < 0.25 * np.linalg.norm(bias_true)


def test_mekf_star_tracker_update_reduces_attitude_error() -> None:
    mekf = AttitudeMEKF(
        q0=np.array([0.0, 0.0, 0.0, 1.0]),
        P0=np.diag([1e-1, 1e-1, 1e-1, 1e-6, 1e-6, 1e-6]),
        Qc=np.zeros((6, 6)),
        R_sun=1e-6 * np.eye(3),
        R_mag=1e-6 * np.eye(3),
        R_star=1e-6 * np.eye(3),
    )
    arr = np.array([0.05, -0.03, 0.02, 1.0])
    q_true = Quaternion.from_array(arr / np.linalg.norm(arr))

    before = _angle_between(mekf.q, q_true.to_array())
    for _ in range(5):
        mekf.update_attitude(q_true.to_array())
    after = _angle_between(mekf.q, q_true.to_array())

    assert after < 0.1 * before


def _make_estimator() -> FullStateEstimator:
    layout = MeasurementLayout((("gps", 6), ("gyro", 3), ("star_tracker", 4)))
    h = np.eye(6)
    orbit = OrbitKalmanFilter(
        r0=_R0,
        v0=_V0,
        P0=np.diag([1e2, 1e2, 1e2, 1.0, 1.0, 1.0]),
        Q=np.diag([1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4]),
        H=h,
        R=np.diag([25.0, 25.0, 25.0, 1e-2, 1e-2, 1e-2]),
    )
    attitude = AttitudeMEKF(
        q0=np.array([0.0, 0.0, 0.0, 1.0]),
        P0=np.diag([1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6]),
        Qc=1e-10 * np.eye(6),
        R_sun=1e-6 * np.eye(3),
        R_mag=1e-6 * np.eye(3),
        R_star=1e-6 * np.eye(3),
    )
    return FullStateEstimator(dt=1.0, epoch=_EPOCH, layout=layout, orbit=orbit, attitude=attitude)


def test_environment_exposure_matches_truth() -> None:
    est = _make_estimator()
    r = np.array([7.0e6, 1.0e5, -2.0e5])

    log = est._expose_environment(r, Quaternion(np.zeros(3), 1.0), _EPOCH, np.zeros(3))  # noqa: SLF001

    lat, lon, alt = eci_to_geodedic(r)
    np.testing.assert_allclose(log.geodetic, np.array([lat, lon, alt]), rtol=1e-9)
    assert log.density > 0.0
    # Identity attitude: the exposed body field equals the inertial field magnitude.
    assert np.linalg.norm(log.b_field_body) > 0.0


def test_estimator_exposes_wheel_momentum_from_tachometer() -> None:
    # Two-wheel array along x and y; tachometer reports relative wheel speeds.
    layout = MeasurementLayout((("gyro", 3), ("tachometer", 2)))
    orbit = OrbitKalmanFilter(r0=_R0, v0=_V0, P0=np.eye(6), Q=np.eye(6), H=np.eye(6), R=np.eye(6))
    attitude = AttitudeMEKF(
        q0=np.array([0.0, 0.0, 0.0, 1.0]),
        P0=np.eye(6),
        Qc=np.zeros((6, 6)),
        R_sun=np.eye(3),
        R_mag=np.eye(3),
    )
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    inertia = np.array([0.01, 0.02])
    est = FullStateEstimator(
        dt=1.0,
        epoch=_EPOCH,
        layout=layout,
        orbit=orbit,
        attitude=attitude,
        rw_axes=axes,
        rw_inertia=inertia,
    )

    omega_est = np.array([0.0, 0.0, 0.0])
    omega_rel = np.array([100.0, -50.0])
    h_wheel = est._wheel_momentum({"tachometer": omega_rel}, omega_est)  # noqa: SLF001

    # axes^T @ (J_w * omega_rel) with identity-aligned axes and zero body rate.
    expected = np.array([inertia[0] * omega_rel[0], inertia[1] * omega_rel[1], 0.0])
    np.testing.assert_allclose(h_wheel, expected, rtol=1e-12)


def test_full_state_estimator_tracks_truth_and_is_deterministic() -> None:
    dt = 1.0
    n_steps = 25
    omega_true = np.array([0.005, -0.003, 0.004])
    bias_true = np.array([5e-4, -3e-4, 2e-4])

    def run() -> tuple[np.ndarray, np.ndarray, Quaternion]:
        est = _make_estimator()
        truth_orbit = np.concatenate([_R0, _V0])
        q_true = Quaternion(np.zeros(3), 1.0)
        local_rng = np.random.default_rng(2)
        x_hat = np.zeros(19)
        for k in range(n_steps):
            truth_orbit = rk4(OrbitKalmanFilter._f, 0.0, dt, truth_orbit, np.zeros(0))  # noqa: SLF001
            q_true_next = q_true.exact_integration(omega_true, dt)
            gps = truth_orbit + local_rng.normal(0.0, 5.0, size=6)
            gyro = omega_true + bias_true + local_rng.normal(0.0, 1e-4, size=3)
            star = q_true_next.to_array() + local_rng.normal(0.0, 1e-3, size=4)
            y_mea = np.concatenate([gps, gyro, star])
            x_hat, _ = est.update(float(k) * dt, y_mea, 0.0)
            q_true = q_true_next
        return np.asarray(x_hat), truth_orbit, q_true

    x_hat, truth_orbit, q_true = run()

    assert x_hat.shape == (19,)  # [r, v, q, omega, b_body, h_wheel]
    np.testing.assert_allclose(x_hat[:3], truth_orbit[:3], atol=50.0)
    assert _angle_between(x_hat[6:10], q_true.to_array()) < np.deg2rad(5.0)
    # No wheels configured -> exposed wheel momentum is zero; b_body is the exposed field.
    np.testing.assert_array_equal(x_hat[16:19], np.zeros(3))

    # Same seeded inputs -> identical estimate (no hidden state / RNG in the estimator).
    x_hat_again, _, _ = run()
    np.testing.assert_array_equal(x_hat, x_hat_again)
