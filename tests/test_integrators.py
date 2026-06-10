import math

import numpy as np

from simulate.dynamics import LinearDynamics
from simulate.integrator import QuaternionRK4, euler, midpoint, rk4
from simulate.output import LinearOutput


def test_euler_accuracy() -> None:
    """Test Euler integration accuracy."""

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1
    x0 = np.array([[1.0]])
    x1 = euler(f, 0.0, dt, x0, np.array([[0.0]]))
    assert math.isclose(x1[0, 0], 1.1)


def test_midpoint_accuracy() -> None:
    """Test Midpoint integration accuracy."""

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1
    x0 = np.array([[1.0]])
    x1 = midpoint(f, 0.0, dt, x0, np.array([[0.0]]))
    assert math.isclose(x1[0, 0], 1.105)


def test_rk4_accuracy() -> None:
    """Test RK4 integration accuracy."""

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1
    x0 = np.array([[1.0]])
    x1 = rk4(f, 0.0, dt, x0, np.array([[0.0]]))
    assert math.isclose(x1[0, 0], math.exp(0.1), rel_tol=1e-5)


def test_linear_dynamics_continuous() -> None:
    """Test LinearPlant with continuous-time dynamics using RK4."""
    a = [[-1.0]]
    b = [[1.0]]
    c = [[1.0]]
    d = [[0.0]]
    dt = 0.1

    dynamics = LinearDynamics(dt=dt, a=a, b=b, integrator=rk4)
    output = LinearOutput(dt=dt, c=c, d=d)

    x, _log = dynamics.evaluate(0.0, 1.0)
    y, _ = output.evaluate(0.0, x, 1.0)
    assert math.isclose(y, 1 - math.exp(-0.1), rel_tol=1e-5)
    assert math.isclose(x, 1 - math.exp(-0.1), rel_tol=1e-5)


def test_linear_dynamics_discrete_fallback() -> None:
    """Test LinearPlant fallback to discrete-time dynamics when no integrator is provided."""
    a = [[0.5]]
    b = [[1.0]]
    c = [[1.0]]
    d = [[0.0]]
    dt = 0.1

    dynamics = LinearDynamics(dt=dt, a=a, b=b)
    output = LinearOutput(dt=dt, c=c, d=d)

    x, _log = dynamics.evaluate(0.0, 1.0)
    y, _ = output.evaluate(0.0, x, 1.0)
    assert y == 1.0
    assert x == 1.0


def test_linear_dynamics_from_config_dynamic_integrator() -> None:
    """Test dynamic loading of integrator via from_config."""
    config = {
        "dt": 0.1,
        "a": [[-1.0]],
        "b": [[1.0]],
        "integrator": "simulate.integrator.rk4",
    }
    output_config = {
        "dt": 0.1,
        "c": [[1.0]],
        "d": [[0.0]],
    }
    output = LinearOutput.from_config(output_config)
    dynamics = LinearDynamics.from_config(config)
    assert dynamics.integrator == rk4

    x, _ = dynamics.evaluate(0.0, 1.0)
    y, _ = output.evaluate(0.0, x, 1.0)
    assert math.isclose(y, 1 - math.exp(-0.1), rel_tol=1e-5)


def test_quaternion_rk4_normalizes_quat_slice() -> None:
    """QuaternionRK4 matches rk4 outside the quaternion slice and keeps it unit-norm."""

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.full_like(x, 0.1)

    x0 = np.zeros((13, 1))
    x0[6:10] = np.array([[1.0], [0.0], [0.0], [0.0]])
    u = np.zeros((1, 1))

    plain = rk4(f, 0.0, 0.1, x0, u)
    integ = QuaternionRK4((6, 10))
    out = integ(f, 0.0, 0.1, x0, u)

    assert math.isclose(float(np.linalg.norm(out[6:10])), 1.0)
    assert np.allclose(out[0:6], plain[0:6])
    assert np.allclose(out[10:], plain[10:])
