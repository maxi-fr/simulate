"""Quadrocopter effectors and models for the example notebook."""

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike

from simulate.component import NoLog
from simulate.controller import Controller
from spacecraft.effector import Effector, RigidBodyState


class FlatGravity(Effector):
    """Environmental constant gravity force in the inertial frame.

    Command-free (``n_inputs = 0``); host mass ``m`` is supplied via :meth:`bind`.
    """

    n_inputs = 0
    n_states = 0

    def __init__(self, gravity_acceleration: ArrayLike = (0.0, 0.0, -9.81)) -> None:
        """Initialize with the gravitational acceleration vector [m/s**2]."""
        self.g_inertial = np.asarray(gravity_acceleration, dtype=float)
        self.mass: float | None = None

    def bind(self, mass: float, inertia: np.ndarray) -> None:  # noqa: ARG002
        """Capture the host body's mass."""
        self.mass = float(mass)

    def calc_contributions(
        self,
        t: float,  # noqa: ARG002
        state: RigidBodyState,  # noqa: ARG002
        x_eff: np.ndarray,  # noqa: ARG002
        cmd: np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the inertial-frame gravity force from mass."""
        if self.mass is None:
            msg = "FlatGravity mass is unbound; compose it into a RigidBodyDynamics."
            raise RuntimeError(msg)

        force_inertial = self.mass * self.g_inertial
        return force_inertial, np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(gravity_acceleration=config.get("gravity_acceleration", (0.0, 0.0, -9.81)))


class Quadrocopter(Effector):
    """Stateless effector modeling a quadrocopter's 4 rotors.

    Command layout: ``cmd = [F1, F2, F3, F4]`` (thrust force of each rotor in Newtons).
    """

    n_inputs = 4
    n_states = 0

    def __init__(
        self,
        rotor_positions: ArrayLike = ((0.2, 0.2, 0.0), (-0.2, -0.2, 0.0), (0.2, -0.2, 0.0), (-0.2, 0.2, 0.0)),
        rotor_directions: ArrayLike = (-1, -1, 1, 1),  # 1: CCW, -1: CW
        torque_to_thrust_ratio: float = 0.015,  # c_q [m]
        thrust_axis: ArrayLike = (0.0, 0.0, 1.0),  # default thrust direction in body frame
    ) -> None:
        """Initialize the quadrocopter effector."""
        self.rotor_positions = np.asarray(rotor_positions, dtype=float)
        self.rotor_directions = np.asarray(rotor_directions, dtype=float)
        self.torque_to_thrust_ratio = float(torque_to_thrust_ratio)
        self.thrust_axis = np.asarray(thrust_axis, dtype=float)
        self.thrust_axis = self.thrust_axis / np.linalg.norm(self.thrust_axis)

    def calc_contributions(
        self,
        t: float,  # noqa: ARG002
        state: RigidBodyState,
        x_eff: np.ndarray,  # noqa: ARG002
        cmd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map rotor thrust commands to inertial force and body torque."""
        forces = cmd.flatten()

        # Total force in the body frame: sum(F_i) * thrust_axis
        total_force_body = np.sum(forces) * self.thrust_axis

        # Rotate body-frame force to inertial frame using conjugate of q_bi
        force_inertial = state.q_bi.conjugate().apply(total_force_body)

        # Calculate torque:
        # Thrust torque: sum( r_i x (F_i * thrust_axis) )
        # Reaction torque: sum( -dir_i * c_q * F_i * thrust_axis )
        torque_body = np.zeros(3, dtype=float)
        for r_i, d_i, f_i in zip(self.rotor_positions, self.rotor_directions, forces, strict=True):
            f_vec = f_i * self.thrust_axis
            tau_thrust = np.cross(r_i, f_vec)
            tau_react = -d_i * self.torque_to_thrust_ratio * f_i * self.thrust_axis
            torque_body += tau_thrust + tau_react

        return force_inertial, torque_body, np.zeros(3, dtype=float)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            rotor_positions=config.get(
                "rotor_positions", ((0.2, 0.2, 0.0), (-0.2, -0.2, 0.0), (0.2, -0.2, 0.0), (-0.2, 0.2, 0.0))
            ),
            rotor_directions=config.get("rotor_directions", (-1, -1, 1, 1)),
            torque_to_thrust_ratio=config.get("torque_to_thrust_ratio", 0.015),
            thrust_axis=config.get("thrust_axis", (0.0, 0.0, 1.0)),
        )


class AerodynamicDragQuad(Effector):
    """Stateless environmental aerodynamic drag for a quadrocopter.

    F_drag = -C_d * v_body (linear drag on velocity in body frame)
    tau_drag = -C_rot * omega (linear drag on angular rate in body frame)
    """

    n_inputs = 0
    n_states = 0

    def __init__(self, c_d: float = 0.1, c_rot: float = 0.05) -> None:
        """Initialize with translational and rotational drag coefficients."""
        self.c_d = float(c_d)
        self.c_rot = float(c_rot)

    def calc_contributions(
        self,
        t: float,  # noqa: ARG002
        state: RigidBodyState,
        x_eff: np.ndarray,  # noqa: ARG002
        cmd: np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the drag forces in the inertial frame and torques in the body frame."""
        # Convert velocity to body frame
        v_body = state.q_bi.apply(state.v_eci)
        f_drag_body = -self.c_d * v_body

        # Rotate body-frame drag force back to inertial frame
        f_drag_inertial = state.q_bi.conjugate().apply(f_drag_body)

        # Rotational drag in body frame
        tau_drag_body = -self.c_rot * state.omega_b_bi

        return f_drag_inertial, tau_drag_body, np.zeros(3, dtype=float)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            c_d=config.get("c_d", 0.1),
            c_rot=config.get("c_rot", 0.05),
        )


class OpenLoopPitchController(Controller[NoLog]):
    """Open-loop controller that commands unequal thrust to pitch the vehicle."""

    def __init__(self, dt: float, f_hover: float) -> None:
        super().__init__(dt)
        self.f_hover = f_hover

    def update(
        self,
        t: float,
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, NoLog]:
        """Compute open loop thrust commands to pitch the vehicle forward."""
        if 1.0 <= t < 2.0:  # noqa: PLR2004
            u = np.array([self.f_hover + 0.5, self.f_hover - 0.5, self.f_hover + 0.5, self.f_hover - 0.5])
        else:
            u = np.array([self.f_hover, self.f_hover, self.f_hover, self.f_hover])
        return u, NoLog()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), f_hover=float(config["f_hover"]))


class ZeroController(Controller[NoLog]):
    """Open-loop controller that commands zero/no inputs (e.g. for drag simulation)."""

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, NoLog]:
        """Return a zero-length command vector."""
        return np.zeros(0), NoLog()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]))


class HoverController(Controller[NoLog]):
    """Controller that commands constant hover thrust on all rotors."""

    def __init__(self, dt: float, f_hover: float) -> None:
        super().__init__(dt)
        self.f_hover = f_hover

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, NoLog]:
        """Compute constant hover thrust commands for all rotors."""
        return np.array([self.f_hover, self.f_hover, self.f_hover, self.f_hover]), NoLog()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), f_hover=float(config["f_hover"]))
