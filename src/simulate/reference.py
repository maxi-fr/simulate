import abc
from typing import Any, Self

import numpy as np

from .component import Component, NoLog


class Reference[L](Component[L], abc.ABC):
    """Abstract base class for all reference generators."""

    def __init__(self, dt: float) -> None:
        """Initialize the reference generator."""
        super().__init__(dt)

    def evaluate(self, t: float) -> tuple[float | np.ndarray, L]:
        """Generate the reference signal (or trajectory) for the current time (with ZOH)."""
        return self._execute_zoh(t, self.update)

    @abc.abstractmethod
    def update(self, t: float) -> tuple[float | np.ndarray, L]:
        """Execute internal update dynamics to generate reference. Must be implemented by subclasses."""


class StepReference(Reference[NoLog]):
    """Reference generator that provides a step signal (or trajectory)."""

    def __init__(
        self,
        dt: float,
        step_value: float | np.ndarray = 1.0,
        start_time: float = 0.0,
        horizon: int = 1,
    ) -> None:
        """Initialize the step reference."""
        super().__init__(dt)
        if isinstance(step_value, (list, tuple)):
            self.step_value = np.array(step_value)
        else:
            self.step_value = step_value
        self.start_time = start_time
        self.horizon = horizon

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            step_value=config.get("step_value", 1.0),
            start_time=float(config.get("start_time", 0.0)),
            horizon=int(config.get("horizon", 1)),
        )

    def update(self, t: float) -> tuple[float | np.ndarray, NoLog]:
        """
        Generate a step signal or trajectory.

        Parameters
        ----------
        t : float
            Simulation time.

        Returns
        -------
        reference : float or numpy.ndarray
            Step value (or horizon trajectory) evaluated at time ``t``.
        log : NoLog
            Empty log placeholder.
        """
        if self.horizon == 1:
            if t >= self.start_time:
                res = self.step_value
            else:
                res = 0.0 if isinstance(self.step_value, float) else np.zeros_like(self.step_value)
        else:
            future_times = t + np.arange(self.horizon) * self.dt
            res = np.where(future_times >= self.start_time, self.step_value, 0.0)

        return res, NoLog()
