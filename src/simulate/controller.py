import abc
import dataclasses
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike

from simulate.component import Component


class Controller[L](Component[L], abc.ABC):
    """Abstract base class for all controllers."""

    def __init__(self, dt: float) -> None:
        """Initialize the controller."""
        super().__init__(dt)

    def evaluate(self, t: float, ref: float | np.ndarray, x_hat: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Compute control action based on reference and estimated state (with ZOH)."""
        return self._execute_zoh(t, self.update, ref, x_hat)

    @abc.abstractmethod
    def update(self, t: float, ref: float | np.ndarray, x_hat: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update dynamics based on reference (or trajectory). Must be implemented by subclasses."""


@dataclasses.dataclass(frozen=True)
class PIDControllerLog:
    """Dataclass for internal PIDController logging."""

    error: float | np.ndarray
    integral: float | np.ndarray


class PIDController(Controller[PIDControllerLog]):
    """Generic discrete-time PID controller using matrix gains."""

    def __init__(
        self,
        dt: float,
        kp: ArrayLike,
        ki: ArrayLike,
        kd: ArrayLike,
    ) -> None:
        """Initialize the PID controller."""
        super().__init__(dt)

        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)

        self.integral: np.ndarray | None = None
        self.prev_error: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            kp=config["kp"],
            ki=config["ki"],
            kd=config["kd"],
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,
        x_hat: float | np.ndarray,
    ) -> tuple[float | np.ndarray, PIDControllerLog]:
        """
        Compute control action based on reference and estimated state.

        Args:
            t: Simulation time.
            ref: Reference trajectory vector.
            x_hat: Estimated state vector.
        """
        ref_vec = self.to_col_vec(ref)
        x_hat_vec = self.to_col_vec(x_hat)

        error = ref_vec - x_hat_vec

        if self.integral is None:
            self.integral = np.zeros_like(error)
        if self.prev_error is None:
            self.prev_error = np.zeros_like(error)

        self.integral += error * self.dt

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error.copy()

        u_vec = self.kp @ error + self.ki @ self.integral + self.kd @ derivative

        return self.from_col_vec(u_vec), PIDControllerLog(error=error.copy(), integral=self.integral.copy())
