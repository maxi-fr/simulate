"""DC Motor dynamics and output components for the example notebook."""

import importlib
from typing import Any, Self

import numpy as np

from simulate import NoLog
from simulate.dynamics import Dynamics
from simulate.output import Output


class DCMotorDynamics(Dynamics[NoLog]):
    """Custom DC Motor dynamics implementation."""

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        R: float,  # noqa: N803
        L: float,  # noqa: N803
        Ke: float,  # noqa: N803
        Kt: float,  # noqa: N803
        J: float,  # noqa: N803
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

        d_omega = (self.Kt * i - self.b * omega) / self.J
        d_i = (-self.R * i - self.Ke * omega + u) / self.L

        return np.array([d_omega, d_i])

    def _make_log(self) -> NoLog:
        """Build a snapshot log of the current state."""
        return NoLog()


class DCMotorOutput(Output[NoLog]):
    """Custom DC Motor output implementation."""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]))

    def update(
        self,
        t: float,  # noqa: ARG002
        x: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, NoLog]:
        """Compute output from current state and input."""
        return x, NoLog()
