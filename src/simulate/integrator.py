from collections.abc import Callable
from typing import Protocol, cast

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
    return cast("np.ndarray", x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))


class QuaternionRK4:
    """RK4 integrator that renormalizes a unit quaternion slice of the state after each step.

    Euclidean RK4 lets a quaternion drift off the unit sphere; this wrapper integrates with
    :func:`rk4` and then rescales the quaternion sub-vector ``x[quat_slice]`` to unit norm.
    It is reusable for any state layout via ``quat_slice`` and satisfies the
    :class:`Integrator` protocol.
    """

    def __init__(self, quat_slice: tuple[int, int] = (6, 10)) -> None:
        """Store the half-open ``[start, stop)`` index range of the quaternion within the state."""
        self._sl = slice(*quat_slice)

    def __call__(
        self,
        f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        t: float,
        dt: float,
        x: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Integrate one step with RK4, then renormalize the quaternion slice."""
        x_next = rk4(f, t, dt, x, u).copy()
        q = x_next[self._sl]
        x_next[self._sl] = q / np.linalg.norm(q)
        return x_next
