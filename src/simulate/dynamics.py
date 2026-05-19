import abc
import importlib
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict

from simulate.component import Component
from simulate.integrator import Integrator


class Dynamics[L: BaseModel](Component[L], abc.ABC):
    """Abstract base class for system dynamics (state transition)."""

    def __init__(self, dt: float, integrator: Integrator | None = None) -> None:
        """Initialize the dynamics component."""
        super().__init__(dt)
        self.integrator = integrator

    def evaluate_dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time dynamics x_dot = f(t, x, u).

        Must be implemented by subclasses if using an integrator.
        """
        msg = "Subclasses must implement evaluate_dynamics() if using an integrator."
        raise NotImplementedError(msg)

    @abc.abstractmethod
    def step(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Advance the dynamics state by one step. Returns the new state x."""

    @abc.abstractmethod
    def update(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update dynamics. Returns the new state x."""


class LinearDynamicsLog(BaseModel):
    """Pydantic model for internal LinearDynamics state logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray


class LinearDynamics(Dynamics[LinearDynamicsLog]):
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

        self.x = np.zeros((self.a.shape[0], 1), dtype=float)

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

    def evaluate_dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Continuous-time dynamics x_dot = Ax + Bu."""
        return cast("np.ndarray", self.a @ x + self.b @ u)

    def step(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, LinearDynamicsLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, u)

    def update(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, LinearDynamicsLog]:
        """
        Advance the dynamics by one time step.

        If an integrator is provided, it uses continuous-time dynamics.
        Otherwise, it assumes discrete-time dynamics.

        Args:
            t: Simulation time.
            u: Control input vector.
        """
        u_vec = self.to_col_vec(u)

        if self.integrator is not None:
            self.x = self.integrator(self.evaluate_dynamics, t, self.dt, self.x, u_vec)
        else:
            self.x = cast("np.ndarray", self.a @ self.x + self.b @ u_vec)

        return self.from_col_vec(self.x), LinearDynamicsLog(x=self.x.copy())
