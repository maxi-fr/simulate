import math
from typing import Any, TypeVar

from pydantic import BaseModel

from simulate.config import ComponentConfig

T = TypeVar("T")  # Type for primary output
L = TypeVar("L", bound=BaseModel)  # Type for log model


class Component[T, L: BaseModel]:
    """Base class for all simulation components implementing Zero-Order Hold (ZOH)."""

    def __init__(self, config: ComponentConfig) -> None:
        """Initialize the component."""
        self.config = config
        self.next_update_time: float = 0.0
        self.last_output: T | None = None
        self.last_log: L | None = None

    def should_update(self, t: float) -> bool:
        """Check if the component should update at the given simulation time."""
        # Use math.isclose to mitigate floating-point precision issues
        return math.isclose(t, self.next_update_time, rel_tol=1e-9, abs_tol=1e-9) or t >= self.next_update_time

    def step(self, t: float, *args: Any, **kwargs: Any) -> tuple[T, L]:  # noqa: ANN401
        """
        Execute the public step method to be called by the orchestrator.

        Handles ZOH logic and delegates to the internal update method.
        """
        if self.should_update(t) or self.last_output is None or self.last_log is None:
            # Time to update
            primary_output, log_output = self.update(t, *args, **kwargs)
            self.last_output = primary_output
            self.last_log = log_output
            # Advance next update time
            self.next_update_time += self.config.dt

        return self.last_output, self.last_log

    def update(self, t: float, *args: Any, **kwargs: Any) -> tuple[T, L]:  # noqa: ANN401
        """
        Execute the internal update method. Must be implemented by subclasses.

        Returns the primary output and the component's log model.
        """
        msg = "Subclasses must implement the update method"
        raise NotImplementedError(msg)
