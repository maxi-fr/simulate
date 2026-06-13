# ruff: noqa: N803, N806, ANN401
"""Symbolic dynamics and linearization helpers using CasADi.

Ported from the legacy `flight_software.controller_models`. This module defines the
continuous-time equations of motion for the satellite attitude and reaction wheels,
and provides functions to build symbolic linearized discrete-time models via CasADi autodiff.
"""

from collections.abc import Callable
from typing import Any

import casadi as ca
import numpy as np

from .signals import MODEL


def integrator(
    f: Callable[[ca.SX, ca.SX, ca.SX], ca.SX],
    x: ca.SX,
    u: ca.SX,
    p: ca.SX,
    dt: float,
) -> ca.SX:
    """Perform a single RK2 integration step and normalize the quaternion part of the state.

    Parameters
    ----------
    f : Callable[[ca.SX, ca.SX, ca.SX], ca.SX]
        The system dynamics function with signature f(x, u, p) -> dx.
    x : ca.SX
        Current state vector.
    u : ca.SX
        Current control input vector.
    p : ca.SX
        Parameters (e.g. magnetic field).
    dt : float
        Time step.

    Returns
    -------
    ca.SX
        The state vector at the next time step.
    """
    k1 = f(x, u, p)
    k2 = f(x + dt * k1, u, p)
    x_next = x + 0.5 * dt * (k1 + k2)

    x_next[:4] = x_next[:4] / ca.norm_2(x_next[:4])
    return x_next


def quaternion_conjugate(q: ca.SX) -> ca.SX:
    """Compute the conjugate of a JPL scalar-last quaternion.

    Parameters
    ----------
    q : ca.SX
        Quaternion (4x1), [qx, qy, qz, qw].

    Returns
    -------
    ca.SX
        Conjugate quaternion (4x1).
    """
    return ca.vertcat(-q[:3], q[3])


def quaternion_product(q_lhs: ca.SX, q_rhs: ca.SX) -> ca.SX:
    """Compute the JPL quaternion product of two scalar-last quaternions.

    Parameters
    ----------
    q_lhs : ca.SX
        Left-hand side quaternion (4x1).
    q_rhs : ca.SX
        Right-hand side quaternion (4x1).

    Returns
    -------
    ca.SX
        The product quaternion (4x1).
    """
    qv = q_lhs[:3]
    w = q_lhs[3]

    qv_other = q_rhs[:3]
    w_other = q_rhs[3]

    v_ret = w_other * qv + w * qv_other - ca.cross(qv, qv_other)
    w_ret = w * w_other - ca.dot(qv, qv_other)

    return ca.vertcat(v_ret, w_ret)


def quaternion_rotation(q: ca.SX, v: ca.SX) -> ca.SX:
    """Rotate a 3D vector v by a JPL scalar-last quaternion q.

    Parameters
    ----------
    q : ca.SX
        Rotation quaternion (4x1).
    v : ca.SX
        Vector to rotate (3x1).

    Returns
    -------
    ca.SX
        Rotated vector (3x1).
    """
    qv = q[:3]
    w = q[3]
    aux = -2.0 * ca.cross(qv, v)
    return v + w * aux - ca.cross(qv, aux)


def attitude_jacobian(q: ca.SX) -> ca.SX:
    """Build the Xi matrix mapping angular velocity to quaternion derivative.

    Parameters
    ----------
    q : ca.SX
        Quaternion (4x1).

    Returns
    -------
    ca.SX
        The Xi matrix (4x3).
    """
    q_vec = q[:3]
    q_w = q[3]

    return ca.vertcat(
        ca.horzcat(q_w, -q_vec[2], q_vec[1]),
        ca.horzcat(q_vec[2], q_w, -q_vec[0]),
        ca.horzcat(-q_vec[1], q_vec[0], q_w),
        -q_vec.T,
    )


def kinematics(q: ca.SX, w: ca.SX) -> ca.SX:
    """Compute the time derivative of a quaternion based on angular velocity.

    Parameters
    ----------
    q : ca.SX
        Current quaternion (4x1).
    w : ca.SX
        Angular velocity in body frame (3x1).

    Returns
    -------
    ca.SX
        Quaternion derivative (4x1).
    """
    return 0.5 * (attitude_jacobian(q) @ w)


def rotational_dynamics(  # noqa: PLR0913
    omega: ca.SX,
    u_rw: ca.SX,
    u_mag: ca.SX,
    h_w: ca.SX,
    B_B: ca.SX,
    J_hat: Any,
) -> ca.SX:
    """Compute angular acceleration (Euler's equations of motion).

    Parameters
    ----------
    omega : ca.SX
        Angular velocity (3x1).
    u_rw : ca.SX
        Reaction wheel control torque command (3x1).
    u_mag : ca.SX
        Commanded magnetic torque vector (3x1).
    h_w : ca.SX
        Reaction wheel angular momentum (3x1).
    B_B : ca.SX
        Magnetic field vector in body frame (3x1).
    J_hat : Any
        Spacecraft inertia matrix (3x3).

    Returns
    -------
    ca.SX
        Angular acceleration (3x1).
    """
    tau_rw = -u_rw
    b = B_B / ca.norm_2(B_B)
    tau_mag = (ca.SX.eye(3) - b @ b.T) @ u_mag

    cross_term = ca.cross(omega, J_hat @ omega + h_w)
    total_torque = tau_mag - tau_rw - cross_term

    return ca.solve(J_hat, total_torque)


def satellite_dynamics(x: ca.SX, u: ca.SX, B_eci: ca.SX, J_hat: Any) -> ca.SX:
    """Compute the continuous-time state derivative dx/dt for ECI system dynamics.

    State vector x: [q_BI(4), omega(3), h_w(3)]
    Input vector u: [u_mag(3), u_rw(3)]

    Parameters
    ----------
    x : ca.SX
        State vector (10x1).
    u : ca.SX
        Input vector (6x1).
    B_eci : ca.SX
        Magnetic field vector in ECI frame (3x1).
    J_hat : Any
        Spacecraft inertia matrix (3x3).

    Returns
    -------
    ca.SX
        State derivative (10x1).
    """
    q_BI = x[MODEL.q]
    omega = x[MODEL.omega]
    h_w = x[MODEL.h_w]

    u_mag = u[MODEL.u_mag]
    u_rw = u[MODEL.u_rw]

    d_q_BI = kinematics(q_BI, omega)
    d_omega = rotational_dynamics(omega, u_rw, u_mag, h_w, quaternion_rotation(q_BI, B_eci), J_hat)
    d_h_w = -u_rw

    return ca.vertcat(d_q_BI, d_omega, d_h_w)


def build_reduced_system_dynamics(
    dt: float,
    J_hat: np.ndarray,
) -> tuple[ca.Function, ca.Function, ca.Function]:
    """Build symbolic discrete dynamics and linearizations for the reduced system state.

    Parameters
    ----------
    dt : float
        Integration time step.
    J_hat : np.ndarray
        Spacecraft inertia matrix (3, 3).

    Returns
    -------
    F : ca.Function
        Discrete dynamics function `x_next = F(x, u, B_eci)`.
    A_func : ca.Function
        Linearized state matrix function `A_tilde = A_func(x, u, B_eci)`.
    B_func : ca.Function
        Linearized input matrix function `B_tilde = B_func(x, u, B_eci)`.
    """
    q_BI = ca.SX.sym("q", 4)
    omega = ca.SX.sym("omega", 3)
    h_w = ca.SX.sym("h_w", 3)
    x = ca.vertcat(q_BI, omega, h_w)

    u_rw = ca.SX.sym("u_rw", 3)
    u_mag = ca.SX.sym("u_mag", 3)
    u = ca.vertcat(u_mag, u_rw)

    B_eci = ca.SX.sym("B_eci", 3)

    def dyn_fn(x_s: ca.SX, u_s: ca.SX, p_s: ca.SX) -> ca.SX:
        return satellite_dynamics(x_s, u_s, p_s, ca.SX(J_hat))

    x_next = integrator(dyn_fn, x, u, B_eci, dt)

    A = ca.jacobian(x_next, x)
    B = ca.jacobian(x_next, u)
    xi = attitude_jacobian(q_BI)
    E = ca.diagcat(xi, ca.SX.eye(6))

    A_tilde = E.T @ A @ E
    B_tilde = E.T @ B

    A_func = ca.Function(
        "get_discrete_linearized_dynamics",
        [x, u, B_eci],
        [A_tilde],
        ["x_star", "u_star", "B_eci_star"],
        ["A_tilde"],
    )
    B_func = ca.Function(
        "get_discrete_linearized_dynamics",
        [x, u, B_eci],
        [B_tilde],
        ["x_star", "u_star", "B_eci_star"],
        ["B_tilde"],
    )
    F = ca.Function(
        "discrete_dynamics",
        [x, u, B_eci],
        [x_next],
        ["x", "u", "B_eci"],
        ["x_next"],
    )

    return F, A_func, B_func
