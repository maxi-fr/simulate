import abc
import dataclasses
from typing import Any, Self

import numpy as np

from simulate.component import Component


class Sensor[L](Component[L], abc.ABC):
    """Abstract base class for all sensors."""

    def __init__(self, dt: float) -> None:
        """Initialize the sensor."""
        super().__init__(dt)

    def evaluate(self, t: float, y: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Measure the plant output (with ZOH)."""
        return self._execute_zoh(t, self.update, y)

    @abc.abstractmethod
    def update(self, t: float, y: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


@dataclasses.dataclass(frozen=True)
class GaussianSensorLog:
    """Dataclass for internal GaussianSensor logging."""

    noise: float | np.ndarray
    value: float | np.ndarray  # measured output (true output + noise)


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
        return self.from_col_vec(y_mea_vec), GaussianSensorLog(noise=noise, value=y_mea_vec.copy())
