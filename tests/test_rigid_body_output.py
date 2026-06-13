import numpy as np

from spacecraft.rigid_body import (
    ANGULAR_VELOCITY,
    BASE_STATES,
    QUATERNION,
    ReactionWheelTelemetry,
    rigid_body_attitude,
    rigid_body_rate,
)


def _state() -> np.ndarray:
    """A full rigid body state (13 base + 6 reaction-wheel array states) with known parts."""
    x = np.zeros(BASE_STATES + 6)
    x[QUATERNION] = np.array([0.5, 0.5, 0.5, 0.5])
    x[ANGULAR_VELOCITY] = np.array([0.1, 0.2, 0.3])
    x[BASE_STATES] = 7.5  # first state (e.g. current_0)
    return x


def test_attitude_measurement_selects_quaternion() -> None:
    """rigid_body_attitude projects the quaternion slice as the truth measurement."""
    x = _state()
    y = rigid_body_attitude(0.0, x, 0.0)
    assert np.allclose(y, x[QUATERNION])


def test_rate_measurement_selects_angular_velocity() -> None:
    """rigid_body_rate projects the body-frame angular velocity slice."""
    x = _state()
    y = rigid_body_rate(0.0, x, 0.0)
    assert np.allclose(y, x[ANGULAR_VELOCITY])


def test_wheel_telemetry_reads_effector_slice() -> None:
    """ReactionWheelTelemetry reads the effector internal state at BASE_STATES."""
    measure = ReactionWheelTelemetry()  # default index == BASE_STATES
    x = _state()
    y = measure(0.0, x, 0.0)
    assert np.allclose(y, np.array([7.5]))


def test_wheel_telemetry_honours_index() -> None:
    """A non-default index selects a later effector state slot."""
    measure = ReactionWheelTelemetry(index=BASE_STATES + 2)
    x = np.zeros(BASE_STATES + 3)
    x[BASE_STATES + 2] = -2.0
    y = measure(0.0, x, 0.0)
    assert np.allclose(y, np.array([-2.0]))
