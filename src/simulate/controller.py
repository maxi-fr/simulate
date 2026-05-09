import abc

import numpy as np
from pydantic import BaseModel, ConfigDict

from simulate.component import Component
from simulate.config import ControllerConfig, PIDControllerConfig


class Controller[L: BaseModel](Component[L], abc.ABC):
    """Abstract base class for all controllers."""

    def __init__(self, config: ControllerConfig) -> None:
        """Initialize the controller."""
        super().__init__(config)

    @abc.abstractmethod
    def step(self, t: float, ref: np.ndarray, y_mea: np.ndarray) -> tuple[np.ndarray, L]:
        """Compute control action based on reference and measurement. Must be implemented by subclasses."""

    @abc.abstractmethod
    def update(self, t: float, ref: np.ndarray, y_mea: np.ndarray) -> tuple[np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


class PIDControllerLog(BaseModel):
    """Pydantic model for internal PIDController logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    error: np.ndarray
    integral: np.ndarray


class PIDController(Controller[PIDControllerLog]):
    """Generic discrete-time PID controller using matrix gains."""

    def __init__(self, config: PIDControllerConfig) -> None:
        """Initialize the PID controller."""
        super().__init__(config)

        self.kp = np.array(config.kp, dtype=float)
        self.ki = np.array(config.ki, dtype=float)
        self.kd = np.array(config.kd, dtype=float)

        # Initialize integral and previous error dynamically during first step based on input shape
        self.integral: np.ndarray | None = None
        self.prev_error: np.ndarray | None = None

    def step(self, t: float, ref: np.ndarray, y_mea: np.ndarray) -> tuple[np.ndarray, PIDControllerLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, ref, y_mea)

    def update(self, t: float, ref: np.ndarray, y_mea: np.ndarray) -> tuple[np.ndarray, PIDControllerLog]:  # noqa: ARG002
        """
        Compute control action based on reference and measurement.

        Args:
            t: Simulation time.
            ref: Reference trajectory vector.
            y_mea: Measured output vector.
        """
        ref = np.atleast_2d(ref)
        if ref.shape[0] == 1 and ref.shape[1] > 1:
            ref = ref.T

        y_mea = np.atleast_2d(y_mea)
        if y_mea.shape[0] == 1 and y_mea.shape[1] > 1:
            y_mea = y_mea.T

        error = ref - y_mea

        if self.integral is None:
            self.integral = np.zeros_like(error)
        if self.prev_error is None:
            self.prev_error = np.zeros_like(error)

        # Accumulate integral
        self.integral += error * self.config.dt

        # Compute derivative (discrete difference)
        derivative = (error - self.prev_error) / self.config.dt
        self.prev_error = error.copy()

        # Compute control effort: u = Kp*e + Ki*∫e + Kd*de/dt
        u = self.kp @ error + self.ki @ self.integral + self.kd @ derivative

        return u, PIDControllerLog(error=error.copy(), integral=self.integral.copy())
