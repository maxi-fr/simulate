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
        y_arr = np.atleast_1d(y)
        noise = self.rng.normal(0, self.std_dev, size=y_arr.shape)
        y_mea = y_arr + noise
        return y_mea, GaussianSensorLog(noise=noise)


@dataclasses.dataclass(frozen=True)
class RandomWalkBiasSensorLog:
    """Dataclass for internal RandomWalkBiasSensor logging."""

    noise: float | np.ndarray
    bias: float | np.ndarray


class RandomWalkBiasSensor(Sensor[RandomWalkBiasSensorLog]):
    """Sensor implementation that adds Gaussian noise and a random walk bias to the measurement."""

    def __init__(
        self,
        dt: float,
        std_dev_noise: float = 0.0,
        std_dev_bias: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Initialize the RandomWalkBiasSensor.

        Parameters
        ----------
        dt : float
            Sampling time step.
        std_dev_noise : float, optional
            Standard deviation of the Gaussian measurement noise, by default 0.0.
        std_dev_bias : float, optional
            Standard deviation of the random walk bias step, by default 0.0.
        seed : int, optional
            Random number generator seed, by default 42.
        """
        super().__init__(dt)
        self.std_dev_noise = std_dev_noise
        self.std_dev_bias = std_dev_bias
        self.rng = np.random.default_rng(seed=seed)
        self.bias: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        RandomWalkBiasSensor
            An instance of the sensor.
        """
        return cls(
            dt=float(config["dt"]),
            std_dev_noise=float(config.get("std_dev_noise", 0.0)),
            std_dev_bias=float(config.get("std_dev_bias", 0.0)),
            seed=int(config.get("seed", 42)),
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        y: float | np.ndarray,
    ) -> tuple[float | np.ndarray, RandomWalkBiasSensorLog]:
        """Add Gaussian noise and a random walk bias to the plant output.

        Parameters
        ----------
        t : float
            Simulation time.
        y : float or numpy.ndarray
            True plant output vector.

        Returns
        -------
        y_mea : numpy.ndarray
            Measured output vector.
        log : RandomWalkBiasSensorLog
            Detailed component logs containing the generated noise and current bias.
        """
        y_arr = np.atleast_1d(y)
        if self.bias is None or self.bias.shape != y_arr.shape:
            # First (or warm-up) sample: initialise the bias to the measurement width.
            self.bias = np.zeros_like(y_arr, dtype=float)
        else:
            bias_step = self.rng.normal(0, self.std_dev_bias, size=y_arr.shape)
            self.bias += bias_step

        noise = self.rng.normal(0, self.std_dev_noise, size=y_arr.shape)
        y_mea = y_arr + self.bias + noise
        return y_mea, RandomWalkBiasSensorLog(noise=noise, bias=self.bias.copy())
