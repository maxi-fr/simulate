from collections.abc import Callable
from typing import Protocol

import numpy as np


class Integrator(Protocol):
    """Protocol for numerical integrators."""

    def __call__(
        self,
        f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        t: float,
        dt: float,
        x: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Integrate the dynamics function f over one time step dt."""
        ...


def euler(
    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t: float,
    dt: float,
    x: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Euler integration method."""
    return x + dt * f(t, x, u)


def midpoint(
    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t: float,
    dt: float,
    x: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Midpoint integration method."""
    k1 = f(t, x, u)
    return x + dt * f(t + dt / 2, x + (dt / 2) * k1, u)


def rk4(
    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t: float,
    dt: float,
    x: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Runge-Kutta 4th order integration method."""
    k1 = f(t, x, u)
    k2 = f(t + dt / 2, x + (dt / 2) * k1, u)
    k3 = f(t + dt / 2, x + (dt / 2) * k2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
