import abc
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict

from simulate.component import Component


class Plant[L: BaseModel](Component[L], abc.ABC):
    """Abstract base class for all plants."""

    def __init__(self, dt: float) -> None:
        """Initialize the plant."""
        super().__init__(dt)

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
    """Generic discrete-time linear plant implementation using state space matrices."""

    def __init__(
        self,
        dt: float,
        a: ArrayLike,
        b: ArrayLike,
        c: ArrayLike,
        d: ArrayLike,
    ) -> None:
        """Initialize the linear plant."""
        super().__init__(dt)

        self.a = np.atleast_2d(a)
        self.b = np.atleast_2d(b)
        self.c = np.atleast_2d(c)
        self.d = np.atleast_2d(d)

        # Initialize state vector to zeros based on A matrix dimension
        self.x = np.zeros((self.a.shape[0], 1), dtype=float)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            a=config["a"],
            b=config["b"],
            c=config["c"],
            d=config["d"],
        )

    def step(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, LinearPlantLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, u)

    def update(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, LinearPlantLog]:  # noqa: ARG002
        """
        Advance the discrete dynamics by one time step.

        Args:
            t: Simulation time.
            u: Control input vector.
        """
        u_vec = self.to_col_vec(u)

        self.x = self.a @ self.x + self.b @ u_vec

        y_vec = self.c @ self.x + self.d @ u_vec

        return self.from_col_vec(y_vec), LinearPlantLog(x=self.x.copy())
