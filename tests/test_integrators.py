# ruff: noqa: N806
import math

import numpy as np

from simulate.dynamics import LinearDynamics
from simulate.integrator import euler, midpoint, rk4
from simulate.sensor import LinearMeasurement
from spacecraft.quaternion import QuaternionRK4


def test_euler_accuracy() -> None:
    """Test Euler integration accuracy."""

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1
    x0 = np.array([1.0])
    x1 = euler(f, 0.0, dt, x0, np.array([0.0]))
    assert math.isclose(x1[0], 1.1)


def test_midpoint_accuracy() -> None:
    """Test Midpoint integration accuracy."""

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1
    x0 = np.array([1.0])
    x1 = midpoint(f, 0.0, dt, x0, np.array([0.0]))
    assert math.isclose(x1[0], 1.105)


def test_rk4_accuracy() -> None:
    """Test RK4 integration accuracy."""

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1
    x0 = np.array([1.0])
    x1 = rk4(f, 0.0, dt, x0, np.array([0.0]))
    assert math.isclose(x1[0], math.exp(0.1), rel_tol=1e-5)


def test_linear_dynamics_continuous() -> None:
    """Test LinearPlant with continuous-time dynamics using RK4."""
    A = [[-1.0]]
    B = [[1.0]]
    C = [[1.0]]
    D = [[0.0]]
    dt = 0.1

    dynamics = LinearDynamics(dt, A, B, integrator=rk4)
    measurement = LinearMeasurement(C, D)

    x, _log = dynamics.evaluate(0.0, 1.0)
    y = measurement(0.0, x, 1.0)
    assert math.isclose(float(np.asarray(y).item()), 1 - math.exp(-0.1), rel_tol=1e-5)
    assert math.isclose(float(np.asarray(x).item()), 1 - math.exp(-0.1), rel_tol=1e-5)


def test_linear_dynamics_discrete_fallback() -> None:
    """Test LinearPlant fallback to discrete-time dynamics when no integrator is provided."""
    A = [[0.5]]
    B = [[1.0]]
    C = [[1.0]]
    D = [[0.0]]
    dt = 0.1

    dynamics = LinearDynamics(dt, A, B)
    measurement = LinearMeasurement(C, D)

    x, _log = dynamics.evaluate(0.0, 1.0)
    y = measurement(0.0, x, 1.0)
    assert float(np.asarray(y).item()) == 1.0
    assert float(np.asarray(x).item()) == 1.0


def test_linear_dynamics_from_config_dynamic_integrator() -> None:
    """Test dynamic loading of integrator via from_config."""
    config = {
        "dt": 0.1,
        "A": [[-1.0]],
        "B": [[1.0]],
        "integrator": "simulate.integrator.rk4",
    }
    measurement = LinearMeasurement(C=[[1.0]], D=[[0.0]])
    dynamics = LinearDynamics.from_config(config)
    assert dynamics.integrator == rk4

    x, _ = dynamics.evaluate(0.0, 1.0)
    y = measurement(0.0, x, 1.0)
    assert math.isclose(float(np.asarray(y).item()), 1 - math.exp(-0.1), rel_tol=1e-5)


def test_quaternion_rk4_normalizes_quat_slice() -> None:
    """QuaternionRK4 matches rk4 outside the quaternion slice and keeps it unit-norm."""

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.full_like(x, 0.1)

    x0 = np.zeros(13)
    x0[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
    u = np.zeros(1)

    plain = rk4(f, 0.0, 0.1, x0, u)
    integ = QuaternionRK4((6, 10))
    out = integ(f, 0.0, 0.1, x0, u)

    assert math.isclose(float(np.linalg.norm(out[6:10])), 1.0)
    assert np.allclose(out[0:6], plain[0:6])
    assert np.allclose(out[10:], plain[10:])
