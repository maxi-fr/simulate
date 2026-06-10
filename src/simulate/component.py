import abc
import dataclasses
import math
from collections.abc import Callable
from typing import Any, Self

import numpy as np


@dataclasses.dataclass(frozen=True)
class NoLog:
    """Dataclass placeholder for components that do not log any internal states."""


class Component[L](abc.ABC):
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
            primary_output, log_output = update_fn(t, *args, **kwargs)
            self.last_output = primary_output
            self.last_log = log_output
            self.next_update_time += self.dt

        return self.last_output, self.last_log
