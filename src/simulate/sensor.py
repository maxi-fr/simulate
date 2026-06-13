import abc
import dataclasses
from typing import Any, Self

import numpy as np

from simulate.component import Component
from simulate.config import build_measurement
from simulate.measurement_model import MeasurementModel


class Sensor[L](Component[L], abc.ABC):
    """Abstract base class for all sensors.

    A sensor maps the plant state to a (noisy) measurement ``y_mea`` at its own ``dt`` with
    Zero-Order Hold. How the measurement is derived is up to the subclass: the generic
    :class:`GaussianSensor` / :class:`RandomWalkBiasSensor` compose a deterministic
    :data:`~simulate.measurement_model.MeasurementModel` with an additive error model, but a
    bespoke sensor may compute ``h(x)`` and its noise directly in :meth:`update`.
    """

    def __init__(self, dt: float) -> None:
        """Initialize the sensor with its sample time."""
        super().__init__(dt)

    def evaluate(self, t: float, x: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Measure the plant state (with ZOH)."""
        return self._execute_zoh(t, self.update, x, u)

    @abc.abstractmethod
    def update(self, t: float, x: float | np.ndarray, u: float | np.ndarray) -> tuple[float | np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


@dataclasses.dataclass(frozen=True)
class GaussianSensorLog:
    """Dataclass for internal GaussianSensor logging."""

    truth: float | np.ndarray
    noise: float | np.ndarray


class GaussianSensor(Sensor[GaussianSensorLog]):
    """Sensor implementation that adds Gaussian noise to the measurement."""

    def __init__(self, dt: float, measurement: MeasurementModel, std_dev: float = 0.0) -> None:
        """Initialize the Gaussian sensor."""
        super().__init__(dt)
        self.measurement = measurement
        self.std_dev = std_dev
        self.rng = np.random.default_rng(seed=42)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            measurement=build_measurement(config["measurement"]),
            std_dev=float(config.get("std_dev", 0.0)),
        )

    def update(
        self, t: float, x: float | np.ndarray, u: float | np.ndarray
    ) -> tuple[float | np.ndarray, GaussianSensorLog]:
        """
        Measure the plant state and add Gaussian noise.

        Parameters
        ----------
        t : float
            Simulation time.
        x : float or numpy.ndarray
            State vector.
        u : float or numpy.ndarray
            Control input vector.

        Returns
        -------
        y_mea : float or numpy.ndarray
            Measured output with additive Gaussian noise.
        log : GaussianSensorLog
            Snapshot of the noise-free truth and the sampled noise.
        """
        y = np.atleast_1d(self.measurement(t, x, u))
        noise = self.rng.normal(0, self.std_dev, size=y.shape)
        y_mea = y + noise
        return y_mea, GaussianSensorLog(truth=y, noise=noise)


@dataclasses.dataclass(frozen=True)
class RandomWalkBiasSensorLog:
    """Dataclass for internal RandomWalkBiasSensor logging."""

    truth: float | np.ndarray
    noise: float | np.ndarray
    bias: float | np.ndarray


class RandomWalkBiasSensor(Sensor[RandomWalkBiasSensorLog]):
    """Sensor implementation that adds Gaussian noise and a random walk bias to the measurement."""

    def __init__(
        self,
        dt: float,
        measurement: MeasurementModel,
        std_dev_noise: float = 0.0,
        std_dev_bias: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Initialize the RandomWalkBiasSensor.

        Parameters
        ----------
        dt : float
            Sampling time step.
        measurement : MeasurementModel
            Deterministic truth model ``y = h(t, x, u)``.
        std_dev_noise : float, optional
            Standard deviation of the Gaussian measurement noise, by default 0.0.
        std_dev_bias : float, optional
            Standard deviation of the random walk bias step, by default 0.0.
        seed : int, optional
            Random number generator seed, by default 42.
        """
        super().__init__(dt)
        self.measurement = measurement
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
            measurement=build_measurement(config["measurement"]),
            std_dev_noise=float(config.get("std_dev_noise", 0.0)),
            std_dev_bias=float(config.get("std_dev_bias", 0.0)),
            seed=int(config.get("seed", 42)),
        )

    def update(
        self,
        t: float,
        x: float | np.ndarray,
        u: float | np.ndarray,
    ) -> tuple[float | np.ndarray, RandomWalkBiasSensorLog]:
        """Measure the plant state and add Gaussian noise and a random walk bias.

        Parameters
        ----------
        t : float
            Simulation time.
        x : float or numpy.ndarray
            State vector.
        u : float or numpy.ndarray
            Control input vector.

        Returns
        -------
        y_mea : numpy.ndarray
            Measured output vector.
        log : RandomWalkBiasSensorLog
            Detailed component logs containing the truth, generated noise and current bias.
        """
        y = np.atleast_1d(self.measurement(t, x, u))
        if self.bias is None or self.bias.shape != y.shape:
            # First (or warm-up) sample: initialise the bias to the measurement width.
            self.bias = np.zeros_like(y, dtype=float)
        else:
            bias_step = self.rng.normal(0, self.std_dev_bias, size=y.shape)
            self.bias += bias_step

        noise = self.rng.normal(0, self.std_dev_noise, size=y.shape)
        y_mea = y + self.bias + noise
        return y_mea, RandomWalkBiasSensorLog(truth=y, noise=noise, bias=self.bias.copy())
