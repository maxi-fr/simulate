import abc

import numpy as np
from pydantic import BaseModel, ConfigDict

from simulate.component import Component
from simulate.config import GaussianSensorConfig, SensorConfig


class Sensor[L: BaseModel](Component[L], abc.ABC):
    """Abstract base class for all sensors."""

    def __init__(self, config: SensorConfig) -> None:
        """Initialize the sensor."""
        super().__init__(config)

    @abc.abstractmethod
    def step(self, t: float, y: np.ndarray) -> tuple[np.ndarray, L]:
        """Measure the plant output. Must be implemented by subclasses."""

    @abc.abstractmethod
    def update(self, t: float, y: np.ndarray) -> tuple[np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


class GaussianSensorLog(BaseModel):
    """Pydantic model for internal GaussianSensor logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    noise: np.ndarray


class GaussianSensor(Sensor[GaussianSensorLog]):
    """Sensor implementation that adds Gaussian noise to the measurement."""

    def __init__(self, config: GaussianSensorConfig) -> None:
        """Initialize the Gaussian sensor."""
        super().__init__(config)
        self.std_dev = config.std_dev
        # Fixed seed for reproducibility in this example
        self.rng = np.random.default_rng(seed=42)

    def step(self, t: float, y: np.ndarray) -> tuple[np.ndarray, GaussianSensorLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, y)

    def update(self, t: float, y: np.ndarray) -> tuple[np.ndarray, GaussianSensorLog]:  # noqa: ARG002
        """
        Add Gaussian noise to the plant output.

        Args:
            t: Simulation time.
            y: True plant output vector.
        """
        noise = self.rng.normal(0, self.std_dev, size=y.shape)
        y_mea = y + noise
        return y_mea, GaussianSensorLog(noise=noise)
