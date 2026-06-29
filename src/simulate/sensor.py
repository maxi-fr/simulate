import abc
import dataclasses
from collections.abc import Callable
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike

from .component import Component
from .config import build_measurement

# A measurement model maps (t, state, input) to a truth output ``y = h(t, x, u)``. Simple
# models can be plain functions; parametrized ones are classes implementing ``__call__``. The
# owning :class:`~simulate.sensor.Sensor` provides the sample rate (``dt``), ZOH and noise.
type MeasurementModel = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


class LinearMeasurement:
    """Generic linear measurement model using state space matrices C and D (``y = C x + D u``)."""

    def __init__(self, C: ArrayLike, D: ArrayLike) -> None:
        """Initialize the linear measurement with the output matrices C and D."""
        self.c = np.atleast_2d(C)
        self.d = np.atleast_2d(D)

    def __call__(
        self,
        _t: float,
        x: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Compute the output from the current state and input."""
        x_arr = x
        u_arr = u
        return cast("np.ndarray", self.c @ x_arr + self.d @ u_arr)


def full_state_measurement(
    t: float,  # noqa: ARG001
    x: np.ndarray,
    u: np.ndarray,  # noqa: ARG001
) -> np.ndarray:
    """Return the full state vector."""
    return x


# --------------------------------------------------------------------------


class Sensor[L](Component[L], abc.ABC):
    """Abstract base class for all sensors.

    A sensor maps the plant state to a (noisy) measurement ``y_mea`` at its own ``dt`` with
    Zero-Order Hold. How the measurement is derived is up to the subclass: the generic
    :class:`GaussianSensor` / :class:`RandomWalkBiasSensor` compose a deterministic
    :data:`~simulate.sensor.MeasurementModel` with an additive error model, but a
    bespoke sensor may compute ``h(x)`` and its noise directly in :meth:`update`.
    """

    def __init__(self, dt: float) -> None:
        """Initialize the sensor with its sample time."""
        super().__init__(dt)

    def evaluate(self, t: float, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, L]:
        """Measure the plant state (with ZOH)."""
        return self._execute_zoh(t, self.update, x, u)

    @abc.abstractmethod
    def update(self, t: float, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, L]:
        """Execute internal update dynamics. Must be implemented by subclasses."""


@dataclasses.dataclass(frozen=True)
class GaussianSensorLog:
    """Dataclass for internal GaussianSensor logging."""

    truth: np.ndarray
    noise: np.ndarray


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

    def update(self, t: float, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, GaussianSensorLog]:
        """
        Measure the plant state and add Gaussian noise.

        Parameters
        ----------
        t : float
            Simulation time.
        x : numpy.ndarray
            State vector.
        u : numpy.ndarray
            Control input vector.

        Returns
        -------
        y_mea : numpy.ndarray
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

    truth: np.ndarray
    noise: np.ndarray
    bias: np.ndarray


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
        x: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, RandomWalkBiasSensorLog]:
        """Measure the plant state and add Gaussian noise and a random walk bias.

        Parameters
        ----------
        t : float
            Simulation time.
        x : numpy.ndarray
            State vector.
        u : numpy.ndarray
            Control input vector.

        Returns
        -------
        y_mea : numpy.ndarray
            Measured output vector.
        log : RandomWalkBiasSensorLog
            Component log containing the truth, generated noise and current bias.
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
