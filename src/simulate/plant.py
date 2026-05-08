import abc
from typing import Any

from pydantic import BaseModel

from simulate.component import Component
from simulate.config import PlantConfig


class Plant[T, L: BaseModel](Component[T, L], abc.ABC):
    """Abstract base class for all plants."""

    def __init__(self, config: PlantConfig) -> None:
        """Initialize the plant."""
        super().__init__(config)


class LinearPlantLog(BaseModel):
    """Pydantic model for internal LinearPlant state logging."""

    x: float


class LinearPlant(Plant[float, LinearPlantLog]):
    """Generic discrete-time linear plant implementation."""

    def __init__(self, config: PlantConfig) -> None:
        """Initialize the linear plant."""
        super().__init__(config)
        self.x: float = 0.0  # Internal state
        # A simple linear discrete system: x_k+1 = a*x_k + b*u_k
        self.a: float = 0.9
        self.b: float = 1.0

    def update(self, t: float, *args: Any, **kwargs: Any) -> tuple[float, LinearPlantLog]:  # noqa: ARG002, ANN401
        """
        Advance the discrete dynamics by one time step.

        Args:
            t: Simulation time.
            args: Expects the control input `u` as the first argument.
        """
        u: float = args[0] if args else 0.0

        # Advance state
        self.x = self.a * self.x + self.b * u

        # Output y = x
        y = self.x

        return y, LinearPlantLog(x=self.x)
