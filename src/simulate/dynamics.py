import abc
import importlib
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike

from simulate.component import Component, NoLog
from simulate.integrator import Integrator


class Dynamics[L](Component[L], abc.ABC):
    """Abstract base class for system dynamics (state transition)."""

    x: np.ndarray

    def __init__(self, dt: float, integrator: Integrator | None = None) -> None:
        """Initialize the dynamics component."""
        super().__init__(dt)
        self.integrator = integrator

    def evaluate(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Evaluate the dynamics at time t. Returns the new state x."""
        return self._execute_zoh(t, self.update, u)

    def update(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """
        Advance the dynamics by one time step.

        If an integrator is provided, `dynamics(t, x, u)` is treated as the
        continuous-time RHS `x_dot = f(t, x, u)` and is integrated over `dt`.
        Otherwise, `dynamics(t, x, u)` is treated as a discrete state transition
        returning `x_next` directly.
        """
        u_arr = np.atleast_1d(u)

        if self.integrator is not None:
            self.x = self.integrator(self.dynamics, t, self.dt, self.x, u_arr)
        else:
            self.x = self.dynamics(t, self.x, u_arr)

        return self.x, self._make_log()

    @abc.abstractmethod
    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        System dynamics kernel.

        With an integrator: returns the continuous-time derivative `x_dot = f(t, x, u)`.
        Without an integrator: returns the discrete state transition `x_next = f(t, x, u)`.
        """

    @abc.abstractmethod
    def _make_log(self) -> L:
        """Build the component-specific log snapshot for the current state."""


class LinearDynamics(Dynamics[NoLog]):
    """Generic linear dynamics implementation using state space matrices A and B."""

    def __init__(
        self,
        dt: float,
        a: ArrayLike,
        b: ArrayLike,
        integrator: Integrator | None = None,
    ) -> None:
        """Initialize the linear dynamics."""
        super().__init__(dt, integrator)

        self.a = np.atleast_2d(a)
        self.b = np.atleast_2d(b)

        self.x = np.zeros(self.a.shape[0], dtype=float)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        integrator = config.get("integrator")
        if isinstance(integrator, str):
            module_name, func_name = integrator.rsplit(".", 1)
            module = importlib.import_module(module_name)
            integrator = getattr(module, func_name)

        return cls(
            dt=float(config["dt"]),
            a=config["a"],
            b=config["b"],
            integrator=integrator,
        )

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Linear dynamics kernel: Ax + Bu (interpreted as x_dot or x_next based on integrator)."""
        return cast("np.ndarray", self.a @ x + self.b @ u)

    def _make_log(self) -> NoLog:
        """Build a snapshot log of the current state."""
        return NoLog()
