import abc
import importlib
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike

from .component import Component, NoLog
from .integrator import Integrator


class Estimator[L](Component[L], abc.ABC):
    """Abstract base class for all estimators."""

    def __init__(self, dt: float) -> None:
        """Initialize the estimator."""
        super().__init__(dt)

    def evaluate(self, t: float, y_mea: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, L]:
        """Estimate the state based on measurement and control input (with ZOH)."""
        return self._execute_zoh(t, self.update, y_mea, u)

    @abc.abstractmethod
    def update(self, t: float, y_mea: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, L]:
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
        y_mea: np.ndarray,
        u: np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, NoLog]:
        """
        Return the measurement as the state estimate.

        Parameters
        ----------
        t : float
            Simulation time.
        y_mea : numpy.ndarray
            Measured output vector.
        u : numpy.ndarray
            Control input vector.

        Returns
        -------
        x_hat : numpy.ndarray
            State estimate, equal to the measurement.
        log : NoLog
            Empty log placeholder.
        """
        return y_mea.copy(), NoLog()


class LuenbergerObserver(Estimator[NoLog]):
    """Model-based linear observer ``x_hat_dot = A x_hat + B u + L (y - C x_hat)``.

    Reconstructs the full state from a (partial, noisy) measurement using a state-space model and
    the observer gain ``L``. Mirroring :class:`~simulate.dynamics.LinearDynamics`, ``A``, ``B`` and
    ``L`` are interpreted as continuous-time matrices and the observer ODE is integrated over ``dt``
    when an ``integrator`` is given; without one they describe the discrete update directly. The
    observer gain itself is designed by the caller (e.g. pole placement) and passed in.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        L: ArrayLike,
        integrator: Integrator | None = None,
    ) -> None:
        """Initialize the observer from the model matrices, observer gain and optional integrator."""
        super().__init__(dt)
        self.a = np.atleast_2d(A)
        self.b = np.atleast_2d(B)
        self.c = np.atleast_2d(C)
        self.l = np.atleast_2d(L)
        self.integrator = integrator

        self.x_hat = np.zeros(self.a.shape[0], dtype=float)
        self._y = np.zeros(self.c.shape[0], dtype=float)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        integrator = config.get("integrator")
        if isinstance(integrator, str):
            module_name, func_name = integrator.rsplit(".", 1)
            module = importlib.import_module(module_name)
            integrator = getattr(module, func_name)

        return cls(
            dt=float(config["dt"]),
            A=config["A"],
            B=config["B"],
            C=config["C"],
            L=config["L"],
            integrator=integrator,
        )

    def _rhs(self, t: float, x_hat: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Observer kernel ``A x_hat + B u + L (y - C x_hat)`` using the measurement held in ``_y``."""
        return cast("np.ndarray", self.a @ x_hat + self.b @ u + self.l @ (self._y - self.c @ x_hat))

    def update(
        self,
        t: float,
        y_mea: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, NoLog]:
        """
        Advance the observer one step and return the state estimate.

        Parameters
        ----------
        t : float
            Simulation time.
        y_mea : numpy.ndarray
            Measured output vector.
        u : numpy.ndarray
            Control input vector.

        Returns
        -------
        x_hat : numpy.ndarray
            Updated full-state estimate.
        log : NoLog
            Empty log placeholder.
        """
        self._y = y_mea

        if self.integrator is not None:
            self.x_hat = self.integrator(self._rhs, t, self.dt, self.x_hat, u)
        else:
            self.x_hat = self._rhs(t, self.x_hat, u)

        return self.x_hat, NoLog()
