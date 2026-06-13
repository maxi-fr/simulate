import abc
from typing import Any, Self

import numpy as np

from simulate.component import Component, NoLog


class Estimator[L](Component[L], abc.ABC):
    """Abstract base class for all estimators."""

    def __init__(self, dt: float) -> None:
        """Initialize the estimator."""
        super().__init__(dt)

    def evaluate(self, t: float, y_mea: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Estimate the state based on measurement and control input (with ZOH)."""
        return self._execute_zoh(t, self.update, y_mea, u)

    @abc.abstractmethod
    def update(self, t: float, y_mea: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


class IdentityEstimator(Estimator[NoLog]):
    """Simple estimator that returns the measurement as the state estimate."""

    def __init__(self, dt: float) -> None:
        """Initialize the identity estimator."""
        super().__init__(dt)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]))

    def update(
        self,
        t: float,  # noqa: ARG002
        y_mea: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, NoLog]:
        """
        Return the measurement as the state estimate.

        Parameters
        ----------
        t : float
            Simulation time.
        y_mea : float or numpy.ndarray
            Measured output vector.
        u : float or numpy.ndarray
            Control input vector.

        Returns
        -------
        x_hat : float or numpy.ndarray
            State estimate, equal to the measurement.
        log : NoLog
            Empty log placeholder.
        """
        res = y_mea.copy() if isinstance(y_mea, np.ndarray) else y_mea
        return res, NoLog()
