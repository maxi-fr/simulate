import numpy as np

from simulate.rigid_body import (
    ANGULAR_VELOCITY,
    BASE_STATES,
    QUATERNION,
    ReactionWheelTelemetryOutput,
    RigidBodyAttitudeOutput,
    RigidBodyRateOutput,
)


def _state() -> np.ndarray:
    """A full rigid body state (13 base + 1 reaction-wheel momentum) with known parts."""
    x = np.zeros((BASE_STATES + 1, 1))
    x[QUATERNION] = np.array([[0.5], [0.5], [0.5], [0.5]])
    x[ANGULAR_VELOCITY] = np.array([[0.1], [0.2], [0.3]])
    x[BASE_STATES, 0] = 7.5  # wheel momentum h_w
    return x


def test_attitude_output_selects_quaternion() -> None:
    """RigidBodyAttitudeOutput projects the quaternion slice as the truth measurement."""
    out = RigidBodyAttitudeOutput(dt=0.1)
    x = _state()
    y, _log = out.evaluate(0.0, x, 0.0)
    assert np.allclose(np.asarray(y).reshape(4, 1), x[QUATERNION])


def test_rate_output_selects_angular_velocity() -> None:
    """RigidBodyRateOutput projects the body-frame angular velocity slice."""
    out = RigidBodyRateOutput(dt=0.1)
    x = _state()
    y, _log = out.evaluate(0.0, x, 0.0)
    assert np.allclose(np.asarray(y).reshape(3, 1), x[ANGULAR_VELOCITY])


def test_wheel_telemetry_reads_effector_slice() -> None:
    """ReactionWheelTelemetryOutput reads the effector internal state at BASE_STATES."""
    out = ReactionWheelTelemetryOutput(dt=0.1)  # default index == BASE_STATES
    x = _state()
    y, _log = out.evaluate(0.0, x, 0.0)
    assert np.isclose(float(y), 7.5)


def test_wheel_telemetry_honours_index() -> None:
    """A non-default index selects a later effector state slot."""
    out = ReactionWheelTelemetryOutput(dt=0.1, index=BASE_STATES + 2)
    x = np.zeros((BASE_STATES + 3, 1))
    x[BASE_STATES + 2, 0] = -2.0
    y, _log = out.evaluate(0.0, x, 0.0)
    assert np.isclose(float(y), -2.0)


def test_outputs_from_config_round_trip() -> None:
    """from_config rebuilds each bespoke output with the configured dt/index."""
    att = RigidBodyAttitudeOutput.from_config({"dt": 0.5})
    rate = RigidBodyRateOutput.from_config({"dt": 0.25})
    tach = ReactionWheelTelemetryOutput.from_config({"dt": 0.1, "index": 14})
    assert att.dt == 0.5
    assert rate.dt == 0.25
    assert tach.dt == 0.1
    assert tach.index == 14
