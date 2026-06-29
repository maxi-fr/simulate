import datetime

import numpy as np

from simulate.sensor import GaussianSensor, RandomWalkBiasSensor
from spacecraft.environment import magnetic_field_vector, sun_position
from spacecraft.frames import eci_to_geodedic, quaternion_from_euler
from spacecraft.measurement import (
    GpsMeasurement,
    MagneticFieldMeasurement,
    ReactionWheelTelemetry,
    SunDirectionMeasurement,
    rigid_body_rate,
)
from spacecraft.quaternion import Quaternion
from spacecraft.rigid_body import BASE_STATES, STATE

_EPOCH = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
_RADIUS = 7.0e6  # m, low Earth orbit

# Identity attitude placed at the head of a 13-element base state vector.
_IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


def _state(r_eci: np.ndarray, q: np.ndarray = _IDENTITY_Q, v_eci: np.ndarray | None = None) -> np.ndarray:
    x = np.zeros(BASE_STATES)
    x[0:3] = r_eci
    x[3:6] = np.zeros(3) if v_eci is None else v_eci
    x[6:10] = q
    return x


def test_magnetic_field_identity_attitude_matches_eci_truth() -> None:
    r_eci = np.array([_RADIUS, 0.0, 0.0])
    measure = MagneticFieldMeasurement(epoch=_EPOCH)

    y = measure(0.0, _state(r_eci), np.array([0.0]))

    lat, lon, alt = eci_to_geodedic(r_eci)
    b_eci = magnetic_field_vector(_EPOCH.replace(tzinfo=None), float(lat), float(lon), float(alt))

    # Identity attitude: body-frame field equals the inertial field.
    np.testing.assert_allclose(y, b_eci, rtol=1e-9, atol=1e-12)


def test_magnetic_field_rotates_with_attitude() -> None:
    r_eci = np.array([0.0, _RADIUS, 0.0])
    measure = MagneticFieldMeasurement(epoch=_EPOCH)
    q = quaternion_from_euler(np.array([0.3, -0.7, 1.1]))

    y_identity = measure(0.0, _state(r_eci), np.array([0.0]))
    y_rotated = measure(0.0, _state(r_eci, q.to_array()), np.array([0.0]))

    # Rotating the body re-expresses the same inertial field; norm is preserved.
    np.testing.assert_allclose(np.linalg.norm(y_rotated), np.linalg.norm(y_identity), rtol=1e-9)
    np.testing.assert_allclose(y_rotated, q.apply(np.asarray(y_identity)), rtol=1e-9, atol=1e-12)


def test_sun_direction_unit_in_sunlight() -> None:
    sun_unit = sun_position(_EPOCH)
    sun_unit = sun_unit / np.linalg.norm(sun_unit)
    r_eci = _RADIUS * sun_unit  # sub-solar point: fully illuminated

    measure = SunDirectionMeasurement(epoch=_EPOCH)
    y = measure(0.0, _state(r_eci), np.array([0.0]))

    np.testing.assert_allclose(np.linalg.norm(y), 1.0, rtol=1e-9)


def test_sun_direction_zero_in_eclipse() -> None:
    sun_unit = sun_position(_EPOCH)
    sun_unit = sun_unit / np.linalg.norm(sun_unit)
    r_eci = -_RADIUS * sun_unit  # anti-solar point: behind the Earth

    measure = SunDirectionMeasurement(epoch=_EPOCH)
    y = measure(0.0, _state(r_eci), np.array([0.0]))

    np.testing.assert_array_equal(np.asarray(y), np.zeros(3))


def test_gps_position_and_velocity() -> None:
    r_eci = np.array([_RADIUS, 1.0e5, -2.0e5])
    v_eci = np.array([10.0, 7.5e3, 3.0])
    state = _state(r_eci, v_eci=v_eci)

    y_full = GpsMeasurement()(0.0, state, np.array([0.0]))
    np.testing.assert_array_equal(np.asarray(y_full), np.concatenate([r_eci, v_eci]))

    y_pos = GpsMeasurement(include_velocity=False)(0.0, state, np.array([0.0]))
    np.testing.assert_array_equal(np.asarray(y_pos), r_eci)


def test_gyro_pairing_adds_bias_and_noise() -> None:
    # Rate gyro = rigid_body_rate truth + RandomWalkBiasSensor noise/bias.
    omega = np.array([0.01, -0.02, 0.03])
    x = np.zeros(BASE_STATES)
    x[STATE.omega] = omega

    truth = rigid_body_rate(0.0, x, np.array([0.0]))
    np.testing.assert_array_equal(np.asarray(truth), omega)

    noiseless = RandomWalkBiasSensor(dt=1.0, measurement=rigid_body_rate)
    y_clean, _ = noiseless.evaluate(0.0, x, np.array([0.0]))
    np.testing.assert_array_equal(np.asarray(y_clean), omega)

    noisy = RandomWalkBiasSensor(dt=1.0, measurement=rigid_body_rate, std_dev_noise=1e-3, std_dev_bias=1e-4)
    y_noisy, log = noisy.evaluate(0.0, x, np.array([0.0]))
    assert not np.allclose(np.asarray(y_noisy), omega)
    assert np.asarray(log.noise).shape == omega.shape


def test_tachometer_pairing_adds_noise() -> None:
    # RW tachometer = ReactionWheelTelemetry truth + GaussianSensor noise.
    wheel_speeds = np.array([250.0, -100.0, 50.0])
    x = np.zeros(BASE_STATES + 3)
    x[BASE_STATES : BASE_STATES + 3] = wheel_speeds

    measure = ReactionWheelTelemetry(base_index=BASE_STATES, n_wheels=3)
    truth = measure(0.0, x, np.array([0.0]))
    np.testing.assert_array_equal(np.asarray(truth), wheel_speeds)

    y_clean, _ = GaussianSensor(dt=1.0, measurement=measure, std_dev=0.0).evaluate(0.0, x, np.array([0.0]))
    np.testing.assert_array_equal(np.asarray(y_clean), wheel_speeds)

    y_noisy, _ = GaussianSensor(dt=1.0, measurement=measure, std_dev=1.0).evaluate(0.0, x, np.array([0.0]))
    assert not np.allclose(np.asarray(y_noisy), wheel_speeds)
