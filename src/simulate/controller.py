import abc
from typing import Any

from pydantic import BaseModel

from simulate.component import Component
from simulate.config import ControllerConfig


class Controller[T, L: BaseModel](Component[T, L], abc.ABC):
    """Abstract base class for all controllers."""

    def __init__(self, config: ControllerConfig) -> None:
        """Initialize the controller."""
        super().__init__(config)


class PIDControllerLog(BaseModel):
    """Pydantic model for internal PIDController logging."""

    error: float
    integral: float


class PIDController(Controller[float, PIDControllerLog]):
    """Generic discrete-time PID controller (currently functioning as PI)."""

    def __init__(self, config: ControllerConfig) -> None:
        """Initialize the PID controller."""
        super().__init__(config)
        self.integral: float = 0.0
        # PI Gains - hardcoded for simplicity in this iteration
        self.kp: float = 0.5
        self.ki: float = 0.1

    def update(self, t: float, *args: Any, **kwargs: Any) -> tuple[float, PIDControllerLog]:  # noqa: ARG002, ANN401
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

        return u, PIDControllerLog(error=error, integral=self.integral)
