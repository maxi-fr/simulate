import abc
import dataclasses
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike

from .component import Component


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
class PIControllerLog:
    """Dataclass for internal PIController logging."""

    error: float | np.ndarray
    integral: float | np.ndarray


class PIController(Controller[PIControllerLog]):
    """Generic discrete-time PI controller using matrix gains.

    The control law is ``u = kp @ (ref - x_hat) + ki @ integral`` where ``integral`` accumulates the
    tracking error. Both gains are matrices, so a multi-element state estimate can be fed back: a
    column of ``kp`` acting on an estimated derivative state provides damping in place of a separate
    derivative term (see the DC motor example, which sources that derivative from an observer).
    """

    def __init__(
        self,
        dt: float,
        kp: ArrayLike,
        ki: ArrayLike,
    ) -> None:
        """Initialize the PI controller."""
        super().__init__(dt)

        self.kp = np.atleast_2d(kp)
        self.ki = np.atleast_2d(ki)

        self.integral: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            kp=config["kp"],
            ki=config["ki"],
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,
        x_hat: float | np.ndarray,
    ) -> tuple[float | np.ndarray, PIControllerLog]:
        """Compute control action based on reference and estimated state."""
        error = np.atleast_1d(ref) - np.atleast_1d(x_hat)

        if self.integral is None:
            self.integral = np.zeros_like(error)
        self.integral += error * self.dt

        u = self.kp @ error + self.ki @ self.integral

        return u, PIControllerLog(error=error.copy(), integral=self.integral.copy())
