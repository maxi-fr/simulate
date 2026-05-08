from typing import Any

from pydantic import BaseModel

from simulate.component import Component
from simulate.config import ControllerConfig


class ControllerLog(BaseModel):
    """Pydantic model for internal Controller logging."""

    error: float
    integral: float


class Controller(Component[float, ControllerLog]):
    """Generic discrete-time controller (e.g., PI controller)."""

    def __init__(self, config: ControllerConfig) -> None:
        """Initialize the controller."""
        super().__init__(config)
        self.integral: float = 0.0
        # PI Gains - hardcoded for simplicity in this iteration
        self.kp: float = 0.5
        self.ki: float = 0.1

    def update(self, t: float, *args: Any, **kwargs: Any) -> tuple[float, ControllerLog]:  # noqa: ARG002, ANN401
        """
        Compute control action based on reference and measurement.

        Args:
            t: Simulation time.
            args: Expects `ref` and `y_mea` (measured output).
        """
        ref: float = args[0] if len(args) > 0 else 0.0
        y_mea: float = args[1] if len(args) > 1 else 0.0

        error = ref - y_mea

        # Accumulate integral
        self.integral += error * self.config.dt

        # Compute control effort
        u = self.kp * error + self.ki * self.integral

        return u, ControllerLog(error=error, integral=self.integral)
