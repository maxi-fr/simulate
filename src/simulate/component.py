import abc
import math
from collections.abc import Callable
from typing import Any, Self, TypeVar

import numpy as np
from pydantic import BaseModel

L = TypeVar("L", bound=BaseModel)  # Type for log model


class Component[L: BaseModel](abc.ABC):
    """Abstract base class for all simulation components implementing Zero-Order Hold (ZOH)."""

    def __init__(self, dt: float) -> None:
        """Initialize the component."""
        self.dt = dt
        self.next_update_time: float = 0.0
        self.last_output: float | np.ndarray | None = None
        self.last_log: L | None = None

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""

    @staticmethod
    def to_col_vec(val: float | np.ndarray) -> np.ndarray:
        """Convert a float or array to a 2D column vector (N, 1)."""
        arr = np.atleast_1d(val)
        if arr.ndim == 1:
            return arr.reshape((-1, 1))
        return arr

    @staticmethod
    def from_col_vec(vec: np.ndarray) -> float | np.ndarray:
        """Convert a 2D column vector (N, 1) to a float (if N=1) or 1D array (if N>1)."""
        if vec.size == 1:
            return float(vec.item())
        return vec.flatten()

    def should_update(self, t: float) -> bool:
        """Check if the component should update at the given simulation time."""
        # Use math.isclose to mitigate floating-point precision issues
        return math.isclose(t, self.next_update_time, rel_tol=1e-9, abs_tol=1e-9) or t >= self.next_update_time

    def _execute_zoh(
        self,
        t: float,
        update_fn: Callable[..., tuple[float | np.ndarray, L]],
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[float | np.ndarray, L]:
        """
        Encapsulate Zero-Order Hold logic, delegating to the provided update function.

        This allows subclasses to define explicitly typed `step` and `update` methods
        without duplicating ZOH logic.
        """
        if self.should_update(t) or self.last_output is None or self.last_log is None:
            # Time to update
            primary_output, log_output = update_fn(t, *args, **kwargs)
            self.last_output = primary_output
            self.last_log = log_output
            # Advance next update time
            self.next_update_time += self.dt

        return self.last_output, self.last_log

    @abc.abstractmethod
    def step(self, t: float, *args: Any, **kwargs: Any) -> tuple[float | np.ndarray, L]:  # noqa: ANN401
        """Execute the public step method to be called by the orchestrator. Must be implemented by subclasses."""
