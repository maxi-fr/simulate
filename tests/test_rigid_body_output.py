import numpy as np

from spacecraft.measurement import (
    ReactionWheelTelemetry,
    rigid_body_attitude,
    rigid_body_rate,
)
from spacecraft.rigid_body import BASE_STATES, STATE


def _state() -> np.ndarray:
    """A full rigid body state (13 base + 6 reaction-wheel array states) with known parts."""
    x = np.zeros(BASE_STATES + 6)
    x[STATE.q] = np.array([0.5, 0.5, 0.5, 0.5])
    x[STATE.omega] = np.array([0.1, 0.2, 0.3])
    x[BASE_STATES] = 7.5  # first state (e.g. current_0)
    return x


def test_attitude_measurement_selects_quaternion() -> None:
    """rigid_body_attitude projects the quaternion slice as the truth measurement."""
    x = _state()
    y = rigid_body_attitude(0.0, x, np.array([0.0]))
    assert np.allclose(y, x[STATE.q])


def test_rate_measurement_selects_angular_velocity() -> None:
    """rigid_body_rate projects the body-frame angular velocity slice."""
    x = _state()
    y = rigid_body_rate(0.0, x, np.array([0.0]))
    assert np.allclose(y, x[STATE.omega])


def test_wheel_telemetry_reads_effector_slice() -> None:
    """ReactionWheelTelemetry reads the effector internal state at BASE_STATES."""
    measure = ReactionWheelTelemetry()  # default base_index == BASE_STATES, n_wheels == 1
    x = _state()
    y = measure(0.0, x, np.array([0.0]))
    assert np.allclose(y, np.array([7.5]))


def test_wheel_telemetry_honours_base_index_and_n_wheels() -> None:
    """A non-default base_index and n_wheels selects an array slice."""
    measure = ReactionWheelTelemetry(base_index=BASE_STATES + 1, n_wheels=2)
    x = np.zeros(BASE_STATES + 4)
    x[BASE_STATES + 1] = -2.0
    x[BASE_STATES + 2] = 4.5
    y = measure(0.0, x, np.array([0.0]))
    assert np.allclose(y, np.array([-2.0, 4.5]))
