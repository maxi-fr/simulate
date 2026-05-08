import abc

import numpy as np
from pydantic import BaseModel, ConfigDict

from simulate.component import Component
from simulate.config import LinearPlantConfig, PlantConfig


class Plant[T, L: BaseModel](Component[T, L], abc.ABC):
    """Abstract base class for all plants."""

    def __init__(self, config: PlantConfig) -> None:
        """Initialize the plant."""
        super().__init__(config)

    @abc.abstractmethod
    def step(self, t: float, u: np.ndarray) -> tuple[T, L]:
        """Advance the plant state by one step. Must be implemented by subclasses."""

    @abc.abstractmethod
    def update(self, t: float, u: np.ndarray) -> tuple[T, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


class LinearPlantLog(BaseModel):
    """Pydantic model for internal LinearPlant state logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray


class LinearPlant(Plant[np.ndarray, LinearPlantLog]):
    """Generic discrete-time linear plant implementation using state space matrices."""

    def __init__(self, config: LinearPlantConfig) -> None:
        """Initialize the linear plant."""
        super().__init__(config)

        # Load matrices as numpy arrays
        self.a = np.array(config.a, dtype=float)
        self.b = np.array(config.b, dtype=float)
        self.c = np.array(config.c, dtype=float)
        self.d = np.array(config.d, dtype=float)

        # Initialize state vector to zeros based on A matrix dimension
        self.x = np.zeros((self.a.shape[0], 1), dtype=float)

    def step(self, t: float, u: np.ndarray) -> tuple[np.ndarray, LinearPlantLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, u)

    def update(self, t: float, u: np.ndarray) -> tuple[np.ndarray, LinearPlantLog]:  # noqa: ARG002
        """
        Advance the discrete dynamics by one time step.

        Args:
            t: Simulation time.
            u: Control input vector.
        """
        # Ensure u is shaped correctly as a column vector
        u = np.atleast_2d(u)
        if u.shape[0] == 1 and u.shape[1] > 1:
            u = u.T

        # Advance state: x_k+1 = A*x_k + B*u_k
        self.x = self.a @ self.x + self.b @ u

        y = self.c @ self.x + self.d @ u

        return y, LinearPlantLog(x=self.x.copy())
