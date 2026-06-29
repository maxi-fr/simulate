"""Quadrocopter effectors and models for the example notebook."""

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike

from simulate.component import NoLog
from simulate.controller import Controller
from spacecraft.effector import Effector, RigidBodyState
from spacecraft.frames import euler_from_quaternion
from spacecraft.quaternion import Quaternion
from spacecraft.rigid_body import STATE


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


class CascadedController(Controller[NoLog]):
    """Linear cascaded PID controller for quadrocopter position and attitude."""

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        mass: float,
        inertia: ArrayLike,
        k_p_pos: ArrayLike,
        k_d_pos: ArrayLike,
        k_p_att: ArrayLike,
        k_d_att: ArrayLike,
        thrust_axis: ArrayLike = (0.0, 0.0, 1.0),
        rotor_positions: ArrayLike = ((0.2, 0.2, 0.0), (-0.2, -0.2, 0.0), (0.2, -0.2, 0.0), (-0.2, 0.2, 0.0)),
        rotor_directions: ArrayLike = (-1, -1, 1, 1),
        torque_to_thrust_ratio: float = 0.015,
        max_thrust_per_rotor: float = 10.0,
    ) -> None:
        """Initialize the cascaded controller."""
        super().__init__(dt)
        self.mass = float(mass)
        inertia_arr = np.asarray(inertia, dtype=float)
        self.inertia = np.diag(inertia_arr) if inertia_arr.ndim == 1 else inertia_arr

        self.k_p_pos = np.asarray(k_p_pos, dtype=float)
        self.k_d_pos = np.asarray(k_d_pos, dtype=float)
        self.k_p_att = np.asarray(k_p_att, dtype=float)
        self.k_d_att = np.asarray(k_d_att, dtype=float)

        self.g = 9.81
        self.thrust_axis = np.asarray(thrust_axis, dtype=float)
        self.thrust_axis = self.thrust_axis / np.linalg.norm(self.thrust_axis)

        self.max_thrust_per_rotor = float(max_thrust_per_rotor)

        # Build mixer matrix
        m_mat = np.zeros((4, 4), dtype=float)
        positions = np.asarray(rotor_positions, dtype=float)
        directions = np.asarray(rotor_directions, dtype=float)

        for i in range(4):
            r_i = positions[i]
            d_i = directions[i]

            # F_total is sum of forces
            m_mat[0, i] = 1.0

            # Torques: r_i x (thrust_axis * f_i) + reaction_torque
            tau_thrust_dir = np.cross(r_i, self.thrust_axis)
            tau_react_dir = -d_i * torque_to_thrust_ratio * self.thrust_axis
            tau_dir = tau_thrust_dir + tau_react_dir

            m_mat[1:4, i] = tau_dir

        self.mixer_inv = np.linalg.inv(m_mat)

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: np.ndarray,
        x_hat: np.ndarray,
    ) -> tuple[np.ndarray, NoLog]:
        """Compute the rotor thrust commands."""
        x = np.asarray(x_hat)
        r = x[STATE.r]
        v = x[STATE.v]
        q = Quaternion.from_array(x[STATE.q])
        omega = x[STATE.omega]

        ref_pos = np.asarray(ref) if isinstance(ref, np.ndarray) else np.array([ref, ref, ref])

        # 1. Outer loop: Position control
        pos_error = ref_pos - r
        vel_error = -v  # assume reference velocity is 0

        # Desired acceleration
        a_des = self.k_p_pos * pos_error + self.k_d_pos * vel_error

        # Add gravity compensation (assuming Z is up, gravity is -9.81 in Z)
        a_des[2] += self.g

        # Desired thrust force vector in inertial frame
        f_des_inertial = self.mass * a_des

        # Map desired inertial force to body frame
        f_des_body = q.apply(f_des_inertial)

        t_des = np.dot(f_des_body, self.thrust_axis)
        t_des = np.clip(t_des, 0.0, 4 * self.max_thrust_per_rotor)

        pitch, roll, yaw = euler_from_quaternion(q)

        theta_des = (-a_des[0] * np.cos(yaw) + a_des[1] * np.sin(yaw)) / self.g
        phi_des = (a_des[0] * np.sin(yaw) + a_des[1] * np.cos(yaw)) / self.g
        psi_des = 0.0  # command zero yaw

        phi_des = np.clip(phi_des, -0.5, 0.5)
        theta_des = np.clip(theta_des, -0.5, 0.5)

        # 2. Inner loop: Attitude control
        phi_err = phi_des - roll
        theta_err = theta_des - pitch
        psi_err = psi_des - yaw

        att_err = np.array([phi_err, theta_err, psi_err])

        # Desired body torque (roll -> x, pitch -> y, yaw -> z)
        # Note: Euler angle rates and body rates have opposite signs with this convention!
        tau_des = self.inertia @ (-self.k_p_att * att_err - self.k_d_att * omega)

        wrench = np.array([t_des, tau_des[0], tau_des[1], tau_des[2]])
        f_rotors = self.mixer_inv @ wrench

        f_rotors = np.clip(f_rotors, 0.0, self.max_thrust_per_rotor)

        return f_rotors, NoLog()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            mass=float(config["mass"]),
            inertia=config["inertia"],
            k_p_pos=config["k_p_pos"],
            k_d_pos=config["k_d_pos"],
            k_p_att=config["k_p_att"],
            k_d_att=config["k_d_att"],
            rotor_positions=config.get(
                "rotor_positions", ((0.2, 0.2, 0.0), (-0.2, -0.2, 0.0), (0.2, -0.2, 0.0), (-0.2, 0.2, 0.0))
            ),
            rotor_directions=config.get("rotor_directions", (-1, -1, 1, 1)),
            torque_to_thrust_ratio=config.get("torque_to_thrust_ratio", 0.015),
            max_thrust_per_rotor=config.get("max_thrust_per_rotor", 10.0),
        )
