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

Conventions match :mod:`rigid_body.quaternion`: forces/torques/momenta are body-frame ``(3, 1)``
column vectors.
"""

import abc
import dataclasses
import importlib
from collections.abc import Callable
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike

import rigid_body.disturbances as dis
from rigid_body.quaternion import Quaternion


@dataclasses.dataclass(frozen=True)
class RigidBodyState:
    """Instantaneous kinematic state of the host body, handed to every effector."""

    r_eci: np.ndarray  # (3) position, inertial frame
    v_eci: np.ndarray  # (3) velocity, inertial frame
    q_bi: Quaternion
    omega_b_bi: np.ndarray  # (3) angular velocity, body frame


class Effector(abc.ABC):
    """Abstract base class for a rigid body effector.

    Subclasses declare how many command slots they consume (``n_inputs``) and how many
    internal states they contribute to the shared rigid body state (``n_states``).
    """

    n_inputs: int
    n_states: int

    def initial_state(self) -> np.ndarray:
        """Return the initial internal state ``(n_states,)`` (zeros by default)."""
        return np.zeros(self.n_states, dtype=float)

    def bind(self, mass: float, inertia: np.ndarray) -> None:  # noqa: B027
        """Receive the host body's mass and inertia. No-op unless the effector needs them."""

    @abc.abstractmethod
    def calc_contributions(
        self,
        t: float,
        state: RigidBodyState,
        x_eff: np.ndarray,
        cmd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute this effector's contribution at the current body state.

        Args:
            t: Simulation time.
            state: Kinematic state of the host body.
            x_eff: This effector's internal state ``(n_states, 1)``.
            cmd: This effector's command slice ``(n_inputs, 1)``.

        Returns
        -------
        A tuple of (force, torque, momentum) body-frame vectors.
        """

    def dynamics(
        self,
        t: float,  # noqa: ARG002
        state: RigidBodyState,  # noqa: ARG002
        x_eff: np.ndarray,  # noqa: ARG002
        cmd: np.ndarray,  # noqa: ARG002
        omega_dot: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray:
        """Compute the derivative of the effector's internal state.

        Args:
            t: Simulation time.
            state: Kinematic state of the host body.
            x_eff: This effector's internal state ``(n_states, 1)``.
            cmd: This effector's command slice ``(n_inputs, 1)``.
            omega_dot: Resolved angular acceleration of the spacecraft ``(3,)``.
        """
        return np.zeros(self.n_states, dtype=float)

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

    def calc_contributions(
        self,
        t: float,  # noqa: ARG002
        state: RigidBodyState,  # noqa: ARG002
        x_eff: np.ndarray,  # noqa: ARG002
        cmd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map the command directly to a body-frame force and torque."""
        return (
            cmd[0:3],
            cmd[3:6],
            np.zeros(3, dtype=float),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:  # noqa: ARG003
        """Instantiate the component from a raw configuration dictionary."""
        return cls()


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

    def calc_contributions(
        self,
        t: float,  # noqa: ARG002
        state: RigidBodyState,
        x_eff: np.ndarray,  # noqa: ARG002
        cmd: np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the gravity-gradient torque from position and attitude."""
        if self.inertia is None:
            msg = "GravityGradient inertia is unbound; compose it into a RigidBodyDynamics."
            raise RuntimeError(msg)

        torque = dis.gravity_gradient(state.r_eci, state.q_bi, self.inertia)

        return (
            np.zeros(3, dtype=float),
            torque,
            np.zeros(3, dtype=float),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(mu=float(config["mu"]))


def _to_array(val: ArrayLike, n: int, name: str) -> np.ndarray:
    """Convert scalar/array to a 1D float array of size n."""
    arr = np.atleast_1d(val)
    if len(arr) == 1:
        return np.repeat(arr, n)
    if len(arr) != n:
        msg = f"Length of {name} ({len(arr)}) must be 1 or equal to the number of elements ({n})."
        raise ValueError(msg)
    return arr.astype(float)


class ReactionWheelArray(Effector):
    """Array of reaction wheels for satellite attitude control."""

    def __init__(  # noqa: PLR0913
        self,
        axes: ArrayLike,
        inertia: ArrayLike,
        torque_constant: ArrayLike,
        time_constant: ArrayLike,
        max_current: ArrayLike,
        max_rpm: ArrayLike = 6000.0,
        initial_currents: ArrayLike | None = None,
        initial_omega: ArrayLike | None = None,
    ) -> None:
        """Initialize the reaction wheel array.

        Args:
            axes: (N, 3) or (3, N) spin axes in the body frame. Normalized internally.
            inertia: (N,) axial inertias J_w (or scalar if identical).
            torque_constant: (N,) torque constants K_w (or scalar if identical).
            time_constant: (N,) current-loop time constants T_cur (or scalar if identical).
            max_current: (N,) current saturation limits i_max (or scalar if identical).
            max_rpm: (N,) maximum speeds in RPM (or scalar if identical).
            initial_currents: (N,) initial motor currents. Defaults to zeros.
            initial_omega: (N,) initial wheel spin rates. Defaults to zeros.
        """
        axes_arr = np.asarray(axes, dtype=float)
        if axes_arr.ndim == 1:
            axes_arr = axes_arr.reshape(1, -1)
        if axes_arr.shape[1] != 3 and axes_arr.shape[0] == 3:  # noqa: PLR2004
            axes_arr = axes_arr.T
        if axes_arr.shape[1] != 3:  # noqa: PLR2004
            msg = f"axes must have shape (N, 3), got {axes_arr.shape}"
            raise ValueError(msg)

        n = len(axes_arr)
        norms = np.linalg.norm(axes_arr, axis=1, keepdims=True)
        self.axes = axes_arr / norms

        self.n_inputs = n
        self.n_states = 2 * n

        self.inertia = _to_array(inertia, n, "inertia")
        self.torque_constant = _to_array(torque_constant, n, "torque_constant")
        self.time_constant = _to_array(time_constant, n, "time_constant")
        self.max_current = _to_array(max_current, n, "max_current")
        self.max_rpm = _to_array(max_rpm, n, "max_rpm")
        self.max_omega: np.ndarray = 2.0 * np.pi * self.max_rpm / 60.0

        if initial_currents is None:
            self.initial_currents = np.zeros(n, dtype=float)
        else:
            self.initial_currents = _to_array(initial_currents, n, "initial_currents")

        if initial_omega is None:
            self.initial_omega = np.zeros(n, dtype=float)
        else:
            self.initial_omega = _to_array(initial_omega, n, "initial_omega")

    def initial_state(self) -> np.ndarray:
        """Return the initial internal state vector."""
        return np.concatenate([self.initial_currents, self.initial_omega])

    def calc_contributions(
        self,
        t: float,  # noqa: ARG002
        state: RigidBodyState,
        x_eff: np.ndarray,
        cmd: np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate reaction torque and carried angular momentum."""
        currents = x_eff[: self.n_inputs]
        omega_rel = x_eff[self.n_inputs :]

        # Clamp currents for torque calculation
        currents_clamped = np.clip(currents, -self.max_current, self.max_current)
        tau_w = self.torque_constant * currents_clamped

        # Apply motor torque limits based on wheel spin rate (saturation)
        tau_w = np.where((omega_rel >= self.max_omega) & (tau_w > 0.0), 0.0, tau_w)
        tau_w = np.where((omega_rel <= -self.max_omega) & (tau_w < 0.0), 0.0, tau_w)

        # Reaction torque on body: sum_k -tau_w_k * axis_k
        torque = -self.axes.T @ tau_w

        # Carried angular momentum: sum_k J_w_k * (omega_rel_k + axis_k^T @ omega_body) * axis_k
        omega_abs = omega_rel + self.axes @ state.omega_b_bi
        h_w = self.inertia * omega_abs
        momentum = self.axes.T @ h_w

        return (
            np.zeros(3, dtype=float),
            torque,
            momentum,
        )

    def dynamics(
        self,
        t: float,  # noqa: ARG002
        state: RigidBodyState,  # noqa: ARG002
        x_eff: np.ndarray,
        cmd: np.ndarray,
        omega_dot: np.ndarray,
    ) -> np.ndarray:
        """Compute current loop dynamics and relative wheel speeds dynamics."""
        currents = x_eff[: self.n_inputs]
        omega_rel = x_eff[self.n_inputs :]

        i_cmd = cmd.flatten()
        i_cmd_clamped = np.clip(i_cmd, -self.max_current, self.max_current)

        didt = (i_cmd_clamped - currents) / self.time_constant
        # Apply current derivative limits to prevent integrator windup
        didt = np.where((currents >= self.max_current) & (didt > 0.0), 0.0, didt)
        didt = np.where((currents <= -self.max_current) & (didt < 0.0), 0.0, didt)

        currents_clamped = np.clip(currents, -self.max_current, self.max_current)
        tau_w = self.torque_constant * currents_clamped

        # Apply motor torque limits based on wheel spin rate (saturation)
        tau_w = np.where((omega_rel >= self.max_omega) & (tau_w > 0.0), 0.0, tau_w)
        tau_w = np.where((omega_rel <= -self.max_omega) & (tau_w < 0.0), 0.0, tau_w)

        omega_dot_arr = np.asarray(omega_dot).flatten()
        domega_dt = tau_w / self.inertia - self.axes @ omega_dot_arr

        return np.concatenate([didt, domega_dt])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Load reaction wheel array from configuration."""
        return cls(
            axes=config["axes"],
            inertia=config["inertia"],
            torque_constant=config["torque_constant"],
            time_constant=config["time_constant"],
            max_current=config["max_current"],
            max_rpm=config.get("max_rpm", 6000.0),
            initial_currents=config.get("initial_currents"),
            initial_omega=config.get("initial_omega"),
        )


class MagnetorquerArray(Effector):
    """Array of magnetorquers for satellite attitude control/momentum management."""

    def __init__(  # noqa: PLR0913
        self,
        axes: ArrayLike,
        dipole_constant: ArrayLike,
        time_constant: ArrayLike,
        max_current: ArrayLike,
        b_field_model: Callable[[float, RigidBodyState], np.ndarray] | ArrayLike | None = None,
        initial_currents: ArrayLike | None = None,
    ) -> None:
        """Initialize the magnetorquer array.

        Args:
            axes: (M, 3) or (3, M) coil normal axes. Normalized internally.
            dipole_constant: (M,) dipole constants K_m (or scalar if identical).
            time_constant: (M,) current-loop time constants T_cur (or scalar if identical).
            max_current: (M,) current saturation limits i_max (or scalar if identical).
            b_field_model: A callable (t, state) -> B_body, a constant B_body 3-vector,
                or None (defaults to zero field).
            initial_currents: (M,) initial currents. Defaults to zeros.
        """
        axes_arr = np.asarray(axes, dtype=float)
        if axes_arr.ndim == 1:
            axes_arr = axes_arr.reshape(1, -1)
        if axes_arr.shape[1] != 3 and axes_arr.shape[0] == 3:  # noqa: PLR2004
            axes_arr = axes_arr.T
        if axes_arr.shape[1] != 3:  # noqa: PLR2004
            msg = f"axes must have shape (M, 3), got {axes_arr.shape}"
            raise ValueError(msg)

        m = len(axes_arr)
        norms = np.linalg.norm(axes_arr, axis=1, keepdims=True)
        self.axes = axes_arr / norms

        self.n_inputs = m
        self.n_states = m

        self.dipole_constant = _to_array(dipole_constant, m, "dipole_constant")
        self.time_constant = _to_array(time_constant, m, "time_constant")
        self.max_current = _to_array(max_current, m, "max_current")

        if initial_currents is None:
            self.initial_currents = np.zeros(m, dtype=float)
        else:
            self.initial_currents = _to_array(initial_currents, m, "initial_currents")

        if b_field_model is None:
            self.b_field_model: Callable[[float, RigidBodyState], np.ndarray] = lambda _t, _state: np.zeros(
                3, dtype=float
            )
        elif callable(b_field_model):
            self.b_field_model = cast("Callable[[float, RigidBodyState], np.ndarray]", b_field_model)
        else:
            const_b = np.asarray(b_field_model, dtype=float).flatten()
            if len(const_b) != 3:  # noqa: PLR2004
                msg = "b_field_model must be a callable or a 3-element vector."
                raise ValueError(msg)
            self.b_field_model = lambda _t, _state: const_b

    def initial_state(self) -> np.ndarray:
        """Return the initial internal state vector."""
        return self.initial_currents

    def calc_contributions(
        self,
        t: float,
        state: RigidBodyState,
        x_eff: np.ndarray,
        cmd: np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate magnetorquer torque."""
        currents = x_eff

        # Clamp currents for dipole calculation
        currents_clamped = np.clip(currents, -self.max_current, self.max_current)
        dipole_m = self.dipole_constant * currents_clamped

        # Total dipole moment vector in the body frame: sum_k m_k * axis_k
        m_vec = self.axes.T @ dipole_m

        # Get local B-field in the body frame
        b_body = self.b_field_model(t, state)

        # Control torque: m x B
        torque = np.cross(m_vec, b_body)

        return (
            np.zeros(3, dtype=float),
            torque,
            np.zeros(3, dtype=float),
        )

    def dynamics(
        self,
        t: float,  # noqa: ARG002
        state: RigidBodyState,  # noqa: ARG002
        x_eff: np.ndarray,
        cmd: np.ndarray,
        omega_dot: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray:
        """Compute coil current loop dynamics."""
        currents = x_eff
        i_cmd = cmd.flatten()
        i_cmd_clamped = np.clip(i_cmd, -self.max_current, self.max_current)

        didt = (i_cmd_clamped - currents) / self.time_constant
        didt = np.where((currents >= self.max_current) & (didt > 0.0), 0.0, didt)

        return np.where((currents <= -self.max_current) & (didt < 0.0), 0.0, didt)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Load magnetorquer array from configuration."""
        b_field_model = config.get("b_field_model")
        if isinstance(b_field_model, str):
            try:
                module_name, cls_name = b_field_model.rsplit(".", 1)
                module = importlib.import_module(module_name)
                b_field_model = getattr(module, cls_name)
            except (ValueError, ImportError, AttributeError):
                pass

        return cls(
            axes=config["axes"],
            dipole_constant=config["dipole_constant"],
            time_constant=config["time_constant"],
            max_current=config["max_current"],
            b_field_model=b_field_model,
            initial_currents=config.get("initial_currents"),
        )
