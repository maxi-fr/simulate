from typing import Any

from pydantic import BaseModel

from simulate.component import Component
from simulate.config import PlantConfig


class PlantLog(BaseModel):
    """Pydantic model for internal Plant state logging."""

    # For a generic discrete plant, we'll log its state x
    x: float


class DiscretePlant(Component[float, PlantLog]):
    """Generic discrete-time plant implementation."""

    def __init__(self, config: PlantConfig) -> None:
        """Initialize the plant."""
        super().__init__(config)
        self.x: float = 0.0  # Internal state
        # A simple linear discrete system: x_k+1 = a*x_k + b*u_k
        # For simplicity, we'll assume a=0.9, b=1.0 for this iteration
        # In a full implementation, these would come from config.
        self.a: float = 0.9
        self.b: float = 1.0

    def update(self, t: float, *args: Any, **kwargs: Any) -> tuple[float, PlantLog]:  # noqa: ARG002, ANN401
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

        return y, PlantLog(x=self.x)
