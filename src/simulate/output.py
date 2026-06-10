import abc
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike

from simulate.component import Component, NoLog


class Output[L](Component[L], abc.ABC):
    """Abstract base class for system output (measurement generation)."""

    def __init__(self, dt: float) -> None:
        """Initialize the output component."""
        super().__init__(dt)

    def evaluate(self, t: float, x: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Evaluate the output from the current state and input (with ZOH)."""
        return self._execute_zoh(t, self.update, x, u)

    @abc.abstractmethod
    def update(self, t: float, x: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update for output. Returns the output y."""


class LinearOutput(Output[NoLog]):
    """Generic linear output implementation using state space matrices C and D."""

    def __init__(
        self,
        dt: float,
        c: ArrayLike,
        d: ArrayLike,
    ) -> None:
        """Initialize the linear output."""
        super().__init__(dt)

        self.c = np.atleast_2d(c)
        self.d = np.atleast_2d(d)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            c=config["c"],
            d=config["d"],
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        x: float | np.ndarray,
        u: float | np.ndarray,
    ) -> tuple[float | np.ndarray, NoLog]:
        """
        Compute output from current state and input.

        Args:
            t: Simulation time.
            x: State vector.
            u: Control input vector.
        """
        x_arr = np.atleast_1d(x)
        u_arr = np.atleast_1d(u)

        y = cast("np.ndarray", self.c @ x_arr + self.d @ u_arr)

        return y, NoLog()
