"""Pre-built 6-DOF rigid body dynamics with modular actuators.

RigidBodyState layout (single shared column vector)::

    x = [ r(3) | v(3) | q(4) | omega(3) | actuator internal states... ]

with position ``r`` and velocity ``v`` in the inertial frame, attitude quaternion ``q``
(scalar-last, inertial->body, unit norm), and angular velocity ``omega`` in the body
frame. See :mod:`rigid_body.quaternion` and :mod:`rigid_body.effector` for conventions.

Equations of motion (``dynamics`` returns the continuous-time derivative)::

    r_dot     = v
    v_dot     = (1/m) * F
    q_dot     = 0.5 * Omega(omega) @ q
    omega_dot = J^-1 @ ( tau - omega x (J @ omega + h) )

where ``F``/``tau``/``h`` are the summed effector force, body torque, and internal angular
momentum (summed over the composed actuators and environmental effectors). The summed force
``F`` is applied directly in the translational equation (no body->inertial rotation or
gravity term is currently added). This single
formulation covers stateless force/torque actuators, reaction wheels, and environmental
effects alike: total angular momentum ``H = J @ omega + h`` is conserved under zero external
torque.
"""

import importlib
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike

from simulate.component import NoLog
from simulate.dynamics import Dynamics
from simulate.integrator import Integrator
from simulate.output import Output

from .effector import Effector, RigidBodyState
from .quaternion import Quaternion, QuaternionRK4

# Public state-vector layout (for building per-part measurement Outputs).
POSITION = slice(0, 3)
VELOCITY = slice(3, 6)
QUATERNION = slice(6, 10)
ANGULAR_VELOCITY = slice(10, 13)
BASE_STATES = 13  # effector internal states begin here, in composition order

_R = POSITION
_V = VELOCITY
_Q = QUATERNION
_W = ANGULAR_VELOCITY
_BASE_STATES = BASE_STATES


def _load_class(class_path: str) -> type:
    """Resolve a dotted ``module.Class`` path to the class object."""
    module_name, cls_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return cast("type", getattr(module, cls_name))


class RigidBodyDynamics(Dynamics[NoLog]):
    """Coupled attitude + position dynamics for a rigid body with composed effectors."""

    def __init__(
        self,
        dt: float,
        mass: float,
        inertia: ArrayLike,
        effectors: list[Effector] | None = None,
        integrator: Integrator | None = None,
    ) -> None:
        """Initialize the rigid body.

        Args:
            dt: Sample time.
            mass: Body mass.
            inertia: Inertia tensor, either a ``(3, 3)`` matrix or a 3-vector diagonal.
            effectors: Effectors composed into the body (order fixes the command layout).
            integrator: Defaults to :class:`QuaternionRK4` over the quaternion slice.
        """
        super().__init__(dt, integrator if integrator is not None else QuaternionRK4((6, 10)))

        self.mass = float(mass)
        inertia_arr = np.asarray(inertia, dtype=float)
        self.inertia = np.diag(inertia_arr) if inertia_arr.ndim == 1 else inertia_arr
        self.inertia_inv: np.ndarray = np.linalg.inv(self.inertia)
        self.effectors = effectors if effectors is not None else []

        # Precompute the per-effector slices into the state and command vectors.
        self._state_slices: list[slice] = []
        self._cmd_slices: list[slice] = []
        state_idx = _BASE_STATES
        cmd_idx = 0
        for eff in self.effectors:
            eff.bind(self.mass, self.inertia)
            self._state_slices.append(slice(state_idx, state_idx + eff.n_states))
            self._cmd_slices.append(slice(cmd_idx, cmd_idx + eff.n_inputs))
            state_idx += eff.n_states
            cmd_idx += eff.n_inputs

        self.x = np.zeros(state_idx, dtype=float)
        self.x[_Q] = np.array([0.0, 0.0, 0.0, 1.0])  # identity attitude
        for eff, sl in zip(self.effectors, self._state_slices, strict=True):
            self.x[sl] = eff.initial_state()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary.

        ``config["effectors"]`` is a list of ``{"class_path": ..., ...params}`` dicts, each
        built via the effector's own ``from_config``. ``integrator`` may be a dotted path.
        """
        integrator = config.get("integrator")
        if isinstance(integrator, str):
            integrator = cast("Integrator", _load_class(integrator))

        effectors = [
            cast("type[Effector]", _load_class(e["class_path"])).from_config(e) for e in config.get("effectors", [])
        ]

        return cls(
            dt=float(config["dt"]),
            mass=float(config["mass"]),
            inertia=config["inertia"],
            effectors=effectors,
            integrator=integrator,
        )

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time rigid body derivative ``x_dot = f(t, x, u)``."""
        q = Quaternion.from_array(x[_Q])
        omega = x[_W]
        state = RigidBodyState(r_eci=x[_R], v_eci=x[_V], q_bi=q, omega_b_bi=omega)

        force = np.zeros(3, dtype=float)
        torque = np.zeros(3, dtype=float)
        momentum = np.zeros(3, dtype=float)
        for eff, s_sl, c_sl in zip(self.effectors, self._state_slices, self._cmd_slices, strict=True):
            f_eff, tau_eff, h_eff = eff.calc_contributions(t, state, x[s_sl], u[c_sl])
            force += f_eff
            torque += tau_eff
            momentum += h_eff

        r_dot = x[_V]
        v_dot = force / self.mass
        q_dot = q.kinematics(omega)
        omega_dot = self.inertia_inv @ (torque - np.cross(omega, self.inertia @ omega + momentum))

        state_dots: list[np.ndarray] = []
        for eff, s_sl, c_sl in zip(self.effectors, self._state_slices, self._cmd_slices, strict=True):
            s_dot = eff.dynamics(t, state, x[s_sl], u[c_sl], omega_dot)
            state_dots.append(s_dot)

        return np.concatenate([r_dot, v_dot, q_dot, omega_dot, *state_dots])

    def _make_log(self) -> NoLog:
        """Build a snapshot log of the current state."""
        return NoLog()


class RigidBodyOutput(Output[NoLog]):
    """Minimal pose output: position (inertial) and attitude quaternion."""

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
        """Extract pose ``[r(3), q(4)]`` from the full rigid body state."""
        y = np.concatenate([x[_R], x[_Q]])  # ty:ignore[not-subscriptable]
        return y, NoLog()


class RigidBodyAttitudeOutput(Output[NoLog]):
    """Attitude measurement: the body->inertial unit quaternion ``q`` ``(4, 1)``.

    Pair with a :class:`~simulate.sensor.GaussianSensor` to model a star tracker. Note that
    additive noise on ``q`` yields a non-unit quaternion; consumers must renormalize.
    """

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
        """Select the attitude quaternion from the full rigid body state."""
        return x[QUATERNION], NoLog()  # ty:ignore[not-subscriptable]


class RigidBodyRateOutput(Output[NoLog]):
    """Angular-rate measurement: the body-frame angular velocity ``omega`` ``(3, 1)``.

    Pair with a :class:`~simulate.sensor.GaussianSensor` to model a rate gyro.
    """

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
        """Select the body-frame angular velocity from the full rigid body state."""
        return x[ANGULAR_VELOCITY], NoLog()  # ty:ignore[not-subscriptable]


class ReactionWheelTelemetryOutput(Output[NoLog]):
    """Effector telemetry: a single effector internal state (e.g. a wheel's momentum ``h_w``).

    ``index`` is the absolute position of the effector state in the rigid body state vector;
    effector states begin at :data:`BASE_STATES` in composition order, so the first effector's
    first state is ``BASE_STATES``.
    """

    def __init__(self, dt: float, index: int = BASE_STATES) -> None:
        """Initialize with the absolute state-vector index of the effector state to report."""
        super().__init__(dt)
        self.index = index

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), index=int(config.get("index", BASE_STATES)))

    def update(
        self,
        t: float,  # noqa: ARG002
        x: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, NoLog]:
        """Select the effector internal state at ``index`` from the full rigid body state."""
        return x[self.index : self.index + 1], NoLog()  # ty:ignore[not-subscriptable]
