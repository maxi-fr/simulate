from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict

from simulate.output import Output


class LinearOutputLog(BaseModel):
    """Pydantic model for internal LinearOutput state logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    y: np.ndarray


class LinearOutput(Output[LinearOutputLog]):
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

    def step(
        self, t: float, x: float | np.ndarray, u: float | np.ndarray
    ) -> tuple[float | np.ndarray, LinearOutputLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, x, u)

    def update(
        self,
        t: float,  # noqa: ARG002
        x: float | np.ndarray,
        u: float | np.ndarray,
    ) -> tuple[float | np.ndarray, LinearOutputLog]:
        """
        Compute output from current state and input.

        Args:
            t: Simulation time.
            x: State vector.
            u: Control input vector.
        """
        x_vec = self.to_col_vec(x)
        u_vec = self.to_col_vec(u)

        y_vec = cast("np.ndarray", self.c @ x_vec + self.d @ u_vec)

        return self.from_col_vec(y_vec), LinearOutputLog(y=y_vec.copy())
