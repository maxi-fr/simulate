import abc

import numpy as np
from pydantic import BaseModel

from simulate.component import Component


class Output[L: BaseModel](Component[L], abc.ABC):
    """Abstract base class for system output (measurement generation)."""

    def __init__(self, dt: float) -> None:
        """Initialize the output component."""
        super().__init__(dt)

    @abc.abstractmethod
    def step(self, t: float, x: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Compute the output from the current state and input."""

    @abc.abstractmethod
    def update(self, t: float, x: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update for output. Returns the output y."""
