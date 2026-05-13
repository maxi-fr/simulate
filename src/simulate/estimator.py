import abc
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict

from simulate.component import Component


class Estimator[L: BaseModel](Component[L], abc.ABC):
    """Abstract base class for all estimators."""

    def __init__(self, dt: float) -> None:
        """Initialize the estimator."""
        super().__init__(dt)

    @abc.abstractmethod
    def step(self, t: float, y_mea: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Estimate the state based on measurement and control input. Must be implemented by subclasses."""

    @abc.abstractmethod
    def update(self, t: float, y_mea: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


class IdentityEstimatorLog(BaseModel):
    """Pydantic model for internal IdentityEstimator logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    y_mea: float | np.ndarray


class IdentityEstimator(Estimator[IdentityEstimatorLog]):
    """Simple estimator that returns the measurement as the state estimate."""

    def __init__(self, dt: float) -> None:
        """Initialize the identity estimator."""
        super().__init__(dt)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]))

    def step(
        self, t: float, y_mea: float | np.ndarray, u: float | np.ndarray
    ) -> tuple[float | np.ndarray, IdentityEstimatorLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, y_mea, u)

    def update(
        self,
        t: float,  # noqa: ARG002
        y_mea: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, IdentityEstimatorLog]:
        """
        Return the measurement as the state estimate.

        Args:
            t: Simulation time.
            y_mea: Measured output vector.
            u: Control input vector.
        """
        res = y_mea.copy() if isinstance(y_mea, np.ndarray) else y_mea
        return res, IdentityEstimatorLog(y_mea=res)
