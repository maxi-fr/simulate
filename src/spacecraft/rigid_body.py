"""Pre-built 6-DOF rigid body dynamics with modular actuators.

RigidBodyState layout (single shared column vector)::

    x = [ r(3) | v(3) | q(4) | omega(3) | actuator internal states... ]

with position ``r`` and velocity ``v`` in the inertial frame, attitude quaternion ``q``
(scalar-last, inertial->body, unit norm), and angular velocity ``omega`` in the body
frame. See :mod:`spacecraft.quaternion` and :mod:`spacecraft.effector` for conventions.

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

import datetime
import importlib
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike

from simulate.component import NoLog
from simulate.dynamics import Dynamics
from simulate.integrator import Integrator

from .effector import Effector, RigidBodyState
from .frames import eci_attitude_from_orc
from .orbit_dynamics import SGP4
from .quaternion import Quaternion, QuaternionRK4
from .signals import BASE_STATES, STATE

# Public state-vector layout aliases for external consumers (per-part measurement Outputs,
# tests); the canonical layout lives in :data:`spacecraft.signals.STATE`.
POSITION = STATE.r
VELOCITY = STATE.v
QUATERNION = STATE.q
ANGULAR_VELOCITY = STATE.omega


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
        state_idx = BASE_STATES
        cmd_idx = 0
        for eff in self.effectors:
            eff.bind(self.mass, self.inertia)
            self._state_slices.append(slice(state_idx, state_idx + eff.n_states))
            self._cmd_slices.append(slice(cmd_idx, cmd_idx + eff.n_inputs))
            state_idx += eff.n_states
            cmd_idx += eff.n_inputs

        self.n_inputs = cmd_idx
        self.x = np.zeros(state_idx, dtype=float)
        self.x[STATE.q] = np.array([0.0, 0.0, 0.0, 1.0])  # identity attitude
        for eff, sl in zip(self.effectors, self._state_slices, strict=True):
            self.x[sl] = eff.initial_state()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary.

        ``config["effectors"]`` is a list of ``{"class_path": ..., ...params}`` dicts, each
        built via the effector's own ``from_config``. ``integrator`` may be a dotted path.

        An optional ``initial_state`` block seeds the orbit and attitude from a TLE and an
        ORC-relative attitude::

            initial_state:
              epoch: "2024-01-01T12:00:00"
              tle: ["1 ...", "2 ..."]
              attitude_orc: {roll: ..., pitch: ..., yaw: ...}   # body wrt ORC [deg]
              angular_velocity_orc: [..., ..., ...]             # body rate wrt ORC [deg/s]

        SGP4 propagates the TLE to the epoch for ``r``/``v``; :func:`~spacecraft.frames.eci_attitude_from_orc`
        turns the ORC-relative attitude into the inertial ``q``/``omega``. When ``initial_state`` is
        omitted the state keeps its defaults (zeros, identity quaternion).
        """
        integrator = config.get("integrator")
        if isinstance(integrator, str):
            integrator = cast("Integrator", _load_class(integrator))

        effectors = [
            cast("type[Effector]", _load_class(e["class_path"])).from_config(e) for e in config.get("effectors", [])
        ]

        instance = cls(
            dt=float(config["dt"]),
            mass=float(config["mass"]),
            inertia=config["inertia"],
            effectors=effectors,
            integrator=integrator,
        )

        init = config.get("initial_state")
        if init is not None:
            epoch = datetime.datetime.fromisoformat(init["epoch"])
            r0, v0 = SGP4.from_tle(*init["tle"]).propagate(epoch)
            att = init["attitude_orc"]
            q_bi, omega0 = eci_attitude_from_orc(
                r0,
                v0,
                roll=att["roll"],
                pitch=att["pitch"],
                yaw=att["yaw"],
                omega_bo=init["angular_velocity_orc"],
            )
            instance.x[STATE.r] = r0
            instance.x[STATE.v] = v0
            instance.x[STATE.q] = q_bi.to_array()
            instance.x[STATE.omega] = omega0

        return instance

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time rigid body derivative ``x_dot = f(t, x, u)``."""
        q = Quaternion.from_array(x[STATE.q])
        omega = x[STATE.omega]
        state = RigidBodyState(r_eci=x[STATE.r], v_eci=x[STATE.v], q_bi=q, omega_b_bi=omega)

        force = np.zeros(3, dtype=float)
        torque = np.zeros(3, dtype=float)
        momentum = np.zeros(3, dtype=float)
        for eff, s_sl, c_sl in zip(self.effectors, self._state_slices, self._cmd_slices, strict=True):
            f_eff, tau_eff, h_eff = eff.calc_contributions(t, state, x[s_sl], u[c_sl])
            force += f_eff
            torque += tau_eff
            momentum += h_eff

        r_dot = x[STATE.v]
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


def rigid_body_pose(_t: float, x: float | np.ndarray, _u: float | np.ndarray) -> np.ndarray:
    """Minimal pose measurement: extract ``[r(3), q(4)]`` from the full rigid body state."""
    return np.concatenate([x[STATE.r], x[STATE.q]])  # ty:ignore[not-subscriptable]


def rigid_body_attitude(_t: float, x: float | np.ndarray, _u: float | np.ndarray) -> np.ndarray:
    """Attitude measurement: the body->inertial unit quaternion ``q`` ``(4, 1)``.

    Pair with a :class:`~simulate.sensor.GaussianSensor` to model a star tracker. Note that
    additive noise on ``q`` yields a non-unit quaternion; consumers must renormalize.
    """
    return x[STATE.q]  # ty:ignore[not-subscriptable]


def rigid_body_rate(_t: float, x: float | np.ndarray, _u: float | np.ndarray) -> np.ndarray:
    """Angular-rate measurement: the body-frame angular velocity ``omega`` ``(3, 1)``.

    Pair with a :class:`~simulate.sensor.GaussianSensor` to model a rate gyro.
    """
    return x[STATE.omega]  # ty:ignore[not-subscriptable]


class ReactionWheelTelemetry:
    """Effector telemetry: a single effector internal state (e.g. a wheel's momentum ``h_w``).

    ``index`` is the absolute position of the effector state in the rigid body state vector;
    effector states begin at :data:`BASE_STATES` in composition order, so the first effector's
    first state is ``BASE_STATES``.
    """

    def __init__(self, index: int = BASE_STATES) -> None:
        """Initialize with the absolute state-vector index of the effector state to report."""
        self.index = index

    def __call__(self, _t: float, x: float | np.ndarray, _u: float | np.ndarray) -> np.ndarray:
        """Select the effector internal state at ``index`` from the full rigid body state."""
        return x[self.index : self.index + 1]  # ty:ignore[not-subscriptable]
