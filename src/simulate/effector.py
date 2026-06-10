"""Modular effectors composed into :class:`~simulate.rigid_body.RigidBodyDynamics`.

An effector contributes a body-frame force, a body-frame torque applied to the body
(including any reaction torque), an internal angular momentum it carries (body frame), and
the derivative of its own internal state. The single interface covers three cases:

* **Commanded actuators** (``n_inputs > 0``) driven by the control input ``u`` — e.g.
  :class:`BodyWrench`, :class:`ReactionWheel`.
* **Environmental effects** (``n_inputs == 0``) that are autonomous functions of time and
  the body state — e.g. :class:`GravityGradient`.

The distinction is whether the effector consumes a command, not whether it is stateful
(``n_states`` may be zero or positive in either case). Composing them into one coupled ODE
over ``[body state | effector states]`` keeps state-dependent environmental forces evaluated
at every integrator substage with the intermediate state.

Conventions match :mod:`simulate.attitude`: forces/torques/momenta are body-frame ``(3, 1)``
column vectors.
"""

import abc
import dataclasses
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike

from simulate.attitude import quat_to_rotation_matrix, skew


@dataclasses.dataclass(frozen=True)
class BodyState:
    """Instantaneous kinematic state of the host body, handed to every effector."""

    r: np.ndarray  # (3, 1) position, inertial frame
    v: np.ndarray  # (3, 1) velocity, inertial frame
    q: np.ndarray  # (4, 1) body->inertial unit quaternion (scalar-first)
    omega: np.ndarray  # (3, 1) angular velocity, body frame


@dataclasses.dataclass(frozen=True)
class EffectorOutput:
    """One effector's contribution to the rigid body equations of motion."""

    force: np.ndarray  # (3, 1) body-frame force at the centre of mass
    torque: np.ndarray  # (3, 1) body-frame torque applied to the body (incl. reaction)
    momentum: np.ndarray  # (3, 1) body-frame internal angular momentum carried
    state_dot: np.ndarray  # (n_states, 1) derivative of the effector's internal state


class Effector(abc.ABC):
    """Abstract base class for a rigid body effector.

    Subclasses declare how many command slots they consume (``n_inputs``) and how many
    internal states they contribute to the shared rigid body state (``n_states``).
    """

    n_inputs: int
    n_states: int

    def initial_state(self) -> np.ndarray:
        """Return the initial internal state ``(n_states, 1)`` (zeros by default)."""
        return np.zeros((self.n_states, 1), dtype=float)

    def bind(self, mass: float, inertia: np.ndarray) -> None:  # noqa: B027
        """Receive the host body's mass and inertia. No-op unless the effector needs them."""

    @abc.abstractmethod
    def evaluate(
        self,
        t: float,
        state: BodyState,
        x_eff: np.ndarray,
        cmd: np.ndarray,
    ) -> EffectorOutput:
        """Compute this effector's contribution at the current body state.

        Args:
            t: Simulation time.
            state: Kinematic state of the host body.
            x_eff: This effector's internal state ``(n_states, 1)``.
            cmd: This effector's command slice ``(n_inputs, 1)``.
        """

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the effector from a raw configuration dictionary."""


class BodyWrench(Effector):
    """Stateless actuator applying a commanded body-frame force and torque.

    Command layout: ``cmd = [Fx, Fy, Fz, tau_x, tau_y, tau_z]`` (body frame).
    """

    n_inputs = 6
    n_states = 0

    def evaluate(
        self,
        t: float,  # noqa: ARG002
        state: BodyState,  # noqa: ARG002
        x_eff: np.ndarray,  # noqa: ARG002
        cmd: np.ndarray,
    ) -> EffectorOutput:
        """Map the command directly to a body-frame force and torque."""
        return EffectorOutput(
            force=cmd[0:3].reshape((3, 1)),
            torque=cmd[3:6].reshape((3, 1)),
            momentum=np.zeros((3, 1), dtype=float),
            state_dot=np.zeros((0, 1), dtype=float),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:  # noqa: ARG003
        """Instantiate the component from a raw configuration dictionary."""
        return cls()


class ReactionWheel(Effector):
    """Momentum-exchange device spinning about a body-fixed ``axis``.

    The internal state is the wheel's angular momentum ``h_w`` about ``axis``; the command
    is the motor torque ``tau_m``. The wheel momentum integrates as ``h_w_dot = tau_m`` and,
    by reaction, the body feels ``torque = -tau_m * axis``. The carried momentum
    ``h_w * axis`` is reported so the rigid body adds the gyroscopic coupling ``-omega x h``.
    """

    n_inputs = 1
    n_states = 1

    def __init__(self, axis: ArrayLike) -> None:
        """Initialize the reaction wheel with a (normalized) body-fixed spin axis."""
        axis_arr = np.asarray(axis, dtype=float).reshape((3, 1))
        self.axis = axis_arr / np.linalg.norm(axis_arr)

    def evaluate(
        self,
        t: float,  # noqa: ARG002
        state: BodyState,  # noqa: ARG002
        x_eff: np.ndarray,
        cmd: np.ndarray,
    ) -> EffectorOutput:
        """Report reaction torque, carried momentum, and the momentum derivative."""
        h_w = x_eff[0, 0]
        tau_m = cmd[0, 0]
        return EffectorOutput(
            force=np.zeros((3, 1), dtype=float),
            torque=-tau_m * self.axis,
            momentum=h_w * self.axis,
            state_dot=np.array([[tau_m]], dtype=float),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(axis=config["axis"])


class GravityGradient(Effector):
    """Environmental gravity-gradient torque about a central body.

    ``tau_gg = (3 * mu / R**3) * (o_body x (J @ o_body))`` where ``R = |r|``, the nadir
    direction in the inertial frame is ``-r / R``, and ``o_body`` is that direction rotated
    into the body frame. Command-free (``n_inputs = 0``); the inertia ``J`` is supplied by
    the host body via :meth:`bind`.
    """

    n_inputs = 0
    n_states = 0

    def __init__(self, mu: float) -> None:
        """Initialize with the central body's gravitational parameter ``mu``."""
        self.mu = float(mu)
        self.inertia: np.ndarray | None = None

    def bind(self, mass: float, inertia: np.ndarray) -> None:  # noqa: ARG002
        """Capture the host body's inertia tensor."""
        self.inertia = inertia

    def evaluate(
        self,
        t: float,  # noqa: ARG002
        state: BodyState,
        x_eff: np.ndarray,  # noqa: ARG002
        cmd: np.ndarray,  # noqa: ARG002
    ) -> EffectorOutput:
        """Compute the gravity-gradient torque from position and attitude."""
        if self.inertia is None:
            msg = "GravityGradient inertia is unbound; compose it into a RigidBodyDynamics."
            raise RuntimeError(msg)

        r_norm = np.linalg.norm(state.r)
        nadir_inertial = -state.r / r_norm
        o_body = quat_to_rotation_matrix(state.q).T @ nadir_inertial
        torque = (3.0 * self.mu / r_norm**3) * (skew(o_body) @ (self.inertia @ o_body))

        return EffectorOutput(
            force=np.zeros((3, 1), dtype=float),
            torque=torque,
            momentum=np.zeros((3, 1), dtype=float),
            state_dot=np.zeros((0, 1), dtype=float),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(mu=float(config["mu"]))
