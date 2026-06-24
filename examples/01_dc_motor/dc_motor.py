"""DC Motor dynamics and measurement model for the example notebook."""

import importlib
from typing import Any, Self

import numpy as np

from simulate.component import NoLog
from simulate.dynamics import Dynamics


class DCMotorDynamics(Dynamics[NoLog]):
    """Custom DC Motor dynamics implementation."""

    def __init__(
        self,
        dt: float,
        R: float,
        L: float,
        Ke: float,
        Kt: float,
        J: float,
        b: float,
        integrator: Any = None,  # noqa: ANN401
    ) -> None:
        super().__init__(dt, integrator)
        self.R = R
        self.L = L
        self.Ke = Ke
        self.Kt = Kt
        self.J = J
        self.b = b

        # Initialize state: [omega, i]
        self.x = np.zeros(2)
        self.n_inputs = 1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Load parameters from configuration dictionary."""
        integrator = config.get("integrator")
        if isinstance(integrator, str):
            module_name, func_name = integrator.rsplit(".", 1)
            module = importlib.import_module(module_name)
            integrator = getattr(module, func_name)

        return cls(
            dt=float(config["dt"]),
            R=config["R"],
            L=config["L"],
            Ke=config["Ke"],
            Kt=config["Kt"],
            J=config["J"],
            b=config["b"],
            integrator=integrator,
        )

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Continuous-time dynamics x_dot = f(t, x, u)."""
        omega = x[0]
        i = x[1]
        voltage = np.atleast_1d(u)[0]

        d_omega = (self.Kt * i - self.b * omega) / self.J
        d_i = (-self.R * i - self.Ke * omega + voltage) / self.L

        return np.array([d_omega, d_i])

    def _make_log(self) -> NoLog:
        """Build a snapshot log of the current state."""
        return NoLog()


def dc_motor_measurement(_t: float, x: float | np.ndarray, _u: float | np.ndarray) -> float | np.ndarray:
    """DC motor measurement model: only the angular velocity ``omega`` (the first state) is observed.

    The armature current ``i`` is left unmeasured; a :class:`~simulate.estimator.LuenbergerObserver`
    reconstructs it from this single channel.
    """
    return np.atleast_1d(x)[0]
