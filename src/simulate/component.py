import abc
import math
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel

from simulate.config import ComponentConfig

L = TypeVar("L", bound=BaseModel)  # Type for log model


class Component[L: BaseModel](abc.ABC):
    """Abstract base class for all simulation components implementing Zero-Order Hold (ZOH)."""

    def __init__(self, config: ComponentConfig) -> None:
        """Initialize the component."""
        self.config = config
        self.next_update_time: float = 0.0
        self.last_output: np.ndarray | None = None
        self.last_log: L | None = None

    def should_update(self, t: float) -> bool:
        """Check if the component should update at the given simulation time."""
        # Use math.isclose to mitigate floating-point precision issues
        return math.isclose(t, self.next_update_time, rel_tol=1e-9, abs_tol=1e-9) or t >= self.next_update_time

    def _execute_zoh(
        self,
        t: float,
        update_fn: Callable[..., tuple[np.ndarray, L]],
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[np.ndarray, L]:
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
            self.next_update_time += self.config.dt

        return self.last_output, self.last_log

    @abc.abstractmethod
    def step(self, t: float, *args: Any, **kwargs: Any) -> tuple[np.ndarray, L]:  # noqa: ANN401
        """Execute the public step method to be called by the orchestrator. Must be implemented by subclasses."""
