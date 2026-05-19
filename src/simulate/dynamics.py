import abc

import numpy as np
from pydantic import BaseModel

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
