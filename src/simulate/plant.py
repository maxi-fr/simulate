import abc
import importlib
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict

from simulate.component import Component
from simulate.integrator import Integrator


class Plant[L: BaseModel](Component[L], abc.ABC):
    """Abstract base class for all plants."""

    def __init__(self, dt: float, integrator: Integrator | None = None) -> None:
        """Initialize the plant."""
        super().__init__(dt)
        self.integrator = integrator

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time dynamics x_dot = f(t, x, u).

        Must be implemented by subclasses if using an integrator.
        """
        msg = "Subclasses must implement dynamics() if using an integrator."
        raise NotImplementedError(msg)

    @abc.abstractmethod
    def step(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Advance the plant state by one step. Must be implemented by subclasses."""

    @abc.abstractmethod
    def update(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


class LinearPlantLog(BaseModel):
    """Pydantic model for internal LinearPlant state logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray


class LinearPlant(Plant[LinearPlantLog]):
    """Generic linear plant implementation using state space matrices."""

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        a: ArrayLike,
        b: ArrayLike,
        c: ArrayLike,
        d: ArrayLike,
        integrator: Integrator | None = None,
    ) -> None:
        """Initialize the linear plant."""
        super().__init__(dt, integrator)

        self.a = np.atleast_2d(a)
        self.b = np.atleast_2d(b)
        self.c = np.atleast_2d(c)
        self.d = np.atleast_2d(d)

        # Initialize state vector to zeros based on A matrix dimension
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
            c=config["c"],
            d=config["d"],
            integrator=integrator,
        )

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Continuous-time dynamics x_dot = Ax + Bu."""
        return cast("np.ndarray", self.a @ x + self.b @ u)

    def step(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, LinearPlantLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, u)

    def update(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, LinearPlantLog]:
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
            self.x = self.integrator(self.dynamics, t, self.dt, self.x, u_vec)
        else:
            self.x = cast("np.ndarray", self.a @ self.x + self.b @ u_vec)

        y_vec = cast("np.ndarray", self.c @ self.x + self.d @ u_vec)

        return self.from_col_vec(y_vec), LinearPlantLog(x=self.x.copy())
