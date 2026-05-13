import abc
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict

from simulate.component import Component


class Sensor[L: BaseModel](Component[L], abc.ABC):
    """Abstract base class for all sensors."""

    def __init__(self, dt: float) -> None:
        """Initialize the sensor."""
        super().__init__(dt)

    @abc.abstractmethod
    def step(self, t: float, y: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Measure the plant output. Must be implemented by subclasses."""

    @abc.abstractmethod
    def update(self, t: float, y: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


class GaussianSensorLog(BaseModel):
    """Pydantic model for internal GaussianSensor logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    noise: float | np.ndarray


class GaussianSensor(Sensor[GaussianSensorLog]):
    """Sensor implementation that adds Gaussian noise to the measurement."""

    def __init__(self, dt: float, std_dev: float = 0.0) -> None:
        """Initialize the Gaussian sensor."""
        super().__init__(dt)
        self.std_dev = std_dev
        self.rng = np.random.default_rng(seed=42)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            std_dev=float(config.get("std_dev", 0.0)),
        )

    def step(self, t: float, y: float | np.ndarray) -> tuple[float | np.ndarray, GaussianSensorLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, y)

    def update(self, t: float, y: float | np.ndarray) -> tuple[float | np.ndarray, GaussianSensorLog]:  # noqa: ARG002
        """
        Add Gaussian noise to the plant output.

        Args:
            t: Simulation time.
            y: True plant output vector.
        """
        y_vec = self.to_col_vec(y)
        noise = self.rng.normal(0, self.std_dev, size=y_vec.shape)
        y_mea_vec = y_vec + noise
        return self.from_col_vec(y_mea_vec), GaussianSensorLog(noise=noise)
