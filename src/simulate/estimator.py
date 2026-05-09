import abc

import numpy as np
from pydantic import BaseModel, ConfigDict

from simulate.component import Component
from simulate.config import EstimatorConfig, IdentityEstimatorConfig


class Estimator[L: BaseModel](Component[L], abc.ABC):
    """Abstract base class for all estimators."""

    def __init__(self, config: EstimatorConfig) -> None:
        """Initialize the estimator."""
        super().__init__(config)

    @abc.abstractmethod
    def step(self, t: float, y_mea: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, L]:
        """Estimate the state based on measurement and control input. Must be implemented by subclasses."""

    @abc.abstractmethod
    def update(self, t: float, y_mea: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


class IdentityEstimatorLog(BaseModel):
    """Pydantic model for internal IdentityEstimator logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    y_mea: np.ndarray


class IdentityEstimator(Estimator[IdentityEstimatorLog]):
    """Simple estimator that returns the measurement as the state estimate."""

    def __init__(self, config: IdentityEstimatorConfig) -> None:
        """Initialize the identity estimator."""
        super().__init__(config)

    def step(self, t: float, y_mea: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, IdentityEstimatorLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, y_mea, u)

    def update(self, t: float, y_mea: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, IdentityEstimatorLog]:  # noqa: ARG002
        """
        Return the measurement as the state estimate.

        Args:
            t: Simulation time.
            y_mea: Measured output vector.
            u: Control input vector.
        """
        return y_mea.copy(), IdentityEstimatorLog(y_mea=y_mea.copy())
