import math

import numpy as np

from simulate.integrator import euler, midpoint, rk4
from simulate.plant import LinearPlant


def test_euler_accuracy() -> None:
    """Test Euler integration accuracy."""

    # dx/dt = x, x(0) = 1 => x(t) = exp(t)
    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1
    x0 = np.array([[1.0]])
    x1 = euler(f, 0.0, dt, x0, np.array([[0.0]]))
    # Euler: x1 = x0 + dt * x0 = 1 + 0.1 * 1 = 1.1
    assert math.isclose(x1[0, 0], 1.1)


def test_midpoint_accuracy() -> None:
    """Test Midpoint integration accuracy."""

    # dx/dt = x, x(0) = 1 => x(t) = exp(t)
    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1
    x0 = np.array([[1.0]])
    x1 = midpoint(f, 0.0, dt, x0, np.array([[0.0]]))
    assert math.isclose(x1[0, 0], 1.105)


def test_rk4_accuracy() -> None:
    """Test RK4 integration accuracy."""

    # dx/dt = x, x(0) = 1 => x(t) = exp(t)
    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x

    dt = 0.1
    x0 = np.array([[1.0]])
    x1 = rk4(f, 0.0, dt, x0, np.array([[0.0]]))
    # exp(0.1) approx 1.105170918
    assert math.isclose(x1[0, 0], math.exp(0.1), rel_tol=1e-5)


def test_linear_plant_continuous() -> None:
    """Test LinearPlant with continuous-time dynamics using RK4."""
    # Continuous system: dx/dt = -x + u, y = x
    # With u=1, x(0)=0 => x(t) = 1 - exp(-t)
    a = [[-1.0]]
    b = [[1.0]]
    c = [[1.0]]
    d = [[0.0]]
    dt = 0.1

    plant = LinearPlant(dt=dt, a=a, b=b, c=c, d=d, integrator=rk4)

    # Step 1 (t=0): u=1.0
    y, log = plant.step(0.0, 1.0)
    # x(0.1) = 1 - exp(-0.1) approx 0.09516258
    assert math.isclose(y, 1 - math.exp(-0.1), rel_tol=1e-5)
    assert math.isclose(log.x[0, 0], 1 - math.exp(-0.1), rel_tol=1e-5)


def test_linear_plant_discrete_fallback() -> None:
    """Test LinearPlant fallback to discrete-time dynamics when no integrator is provided."""
    # Discrete system: x[k+1] = 0.5*x[k] + u[k]
    a = [[0.5]]
    b = [[1.0]]
    c = [[1.0]]
    d = [[0.0]]
    dt = 0.1

    plant = LinearPlant(dt=dt, a=a, b=b, c=c, d=d)  # No integrator

    y, log = plant.step(0.0, 1.0)
    # x[1] = 0.5*0 + 1.0 = 1.0
    assert y == 1.0
    assert log.x[0, 0] == 1.0


def test_linear_plant_from_config_dynamic_integrator() -> None:
    """Test dynamic loading of integrator via from_config."""
    config = {
        "dt": 0.1,
        "a": [[-1.0]],
        "b": [[1.0]],
        "c": [[1.0]],
        "d": [[0.0]],
        "integrator": "simulate.integrator.rk4",
    }
    plant = LinearPlant.from_config(config)
    assert plant.integrator == rk4

    y, _ = plant.step(0.0, 1.0)
    assert math.isclose(y, 1 - math.exp(-0.1), rel_tol=1e-5)
