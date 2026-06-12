from collections.abc import Callable

import casadi as ca
import numpy as np


def integrator(f: Callable[[ca.SX, ca.SX, ca.SX], ca.SX], x: ca.SX, u: ca.SX, p: ca.SX, dt: ca.SX | float) -> ca.SX:
    """
    Performs a single RK2 integration step using a Python callable for dynamics.

    This function also normalizes the quaternion part of the state vector (first 4 elements)
    after the integration step to ensure unit norm.

    Parameters
    ----------
    f : Callable[[ca.SX, ca.SX, ca.SX], ca.SX]
        The system dynamics function with the signature f(x, u, p) -> dx.
    x : ca.SX
        The current state vector.
    u : ca.SX
        The current control input vector.
    p : ca.SX
        Time-varying parameters (e.g., magnetic field).
    dt : ca.SX | float
        The integration time step.

    Returns
    -------
    ca.SX
        The state vector at the next time step.
    """
    # k1 = f(x, u, p)
    # k2 = f(x + 0.5 * dt * k1, u, p)
    # k3 = f(x + 0.5 * dt * k2, u, p)
    # k4 = f(x + dt * k3, u, p)
    # x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) #type: ignore
    k1 = f(x, u, p)
    k2 = f(x + dt * k1, u, p)

    x_next = x + 0.5 * dt * (k1 + k2)

    x_next[:4] = x_next[:4] / ca.norm_2(x_next[:4])

    return x_next


def quaternion_conjugate(q: ca.SX) -> ca.SX:
    """
    Symbolic quaternion conjugate.

    Computes the conjugate of a quaternion.
    Assumes scalar-last format: q = [qx, qy, qz, qw].

    Parameters
    ----------
    q : ca.SX
        Quaternion (4x1).

    Returns
    -------
    ca.SX
        Conjugate quaternion (4x1).
    """
    return ca.vertcat(-q[:3], q[3])


def quaternion_product(q_lhs: ca.SX, q_rhs: ca.SX) -> ca.SX:
    """
    Symbolic quaternion multiplication.

    Computes the JPL quaternion product: q_ret = q_lhs * q_rhs.
    Assumes scalar-last format: q = [qx, qy, qz, qw].

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
    """
    Rotates a 3D vector v by a quaternion q.

    Implements the operation v' = q * v * q_conj.

    Parameters
    ----------
    q : ca.SX
        Rotation quaternion (4x1), scalar-last.
    v : ca.SX
        3D vector to rotate (3x1).

    Returns
    -------
    ca.SX
        Rotated 3D vector (3x1).
    """
    qv = q[:3]
    w = q[3]

    aux = -2.0 * ca.cross(qv, v)

    return v + w * aux - ca.cross(qv, aux)


def attitude_jacobian(q: ca.SX) -> ca.SX:
    """
    Builds the Xi matrix symbolically.

    The Xi matrix maps angular velocity to quaternion derivative:
    dq/dt = 0.5 * Xi(q) * omega.

    Xi(q) = | q_4 * I_3 + [q_{1:3} x] |
            | -q_{1:3}^T              |

    Parameters
    ----------
    q : ca.SX
        Quaternion (4x1), scalar-last.

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
    """
    Symbolic quaternion kinematics.

    Computes the time derivative of a quaternion based on angular velocity.
    q_dot = 0.5 * q (x) w  (quaternion multiplication)

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


def rotational_dynamics(
    omega: ca.SX, u_rw: ca.SX, u_mag: ca.SX, h_w: ca.SX, B_B: ca.SX, J_hat: np.ndarray | ca.SX | ca.DM
) -> ca.SX:
    """
    Symbolic rotational dynamics (Euler's equation).

    Computes angular acceleration considering control torques and gyroscopic terms.

    Parameters
    ----------
    omega : ca.SX
        Angular velocity (3x1).
    u_rw : ca.SX
        Reaction wheel control torque (3x1).
    u_mag : ca.SX
        Commanded magnetic torque vector (3x1).
    h_w : ca.SX
        Reaction wheel angular momentum (3x1).
    B_B : ca.SX
        Magnetic field vector in body frame (3x1).
    J_hat : Union[np.ndarray, ca.SX, ca.DM]
        Estimated inertia tensor of the satellite (3x3).

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


def satellite_dynamics(x: ca.SX, u: ca.SX, B_eci: ca.SX, J_hat: np.ndarray | ca.SX | ca.DM) -> ca.SX:
    """
    Computes the state derivative dx/dt for the satellite system.

    State vector x: [q_BI (4), omega (3), h_w (3)] (size 10).
    Input vector u: [u_rw (3), u_mag (3)] (size 6).

    Parameters
    ----------
    x : ca.SX
        State vector.
    u : ca.SX
        Control input vector.
    B_eci : ca.SX
        Magnetic field vector in inertial frame (3x1).
    J_hat : Union[np.ndarray, ca.SX, ca.DM]
        Inertia matrix.

    Returns
    -------
    ca.SX
        State derivative dx/dt (10x1).
    """
    q_BI = x[:4]
    omega = x[4:7]
    h_w = x[7:10]

    u_mag = u[:3]
    u_rw = u[3:]

    d_q_BI = kinematics(q_BI, omega)

    d_omega = rotational_dynamics(omega, u_rw, u_mag, h_w, quaternion_rotation(q_BI, B_eci), J_hat)
    d_h_w = -u_rw

    return ca.vertcat(d_q_BI, d_omega, d_h_w)


def error_dynamics(
    x: ca.SX, u: ca.SX, B_orc: ca.SX, omega_c: np.ndarray | ca.SX, J_hat: np.ndarray | ca.SX | ca.DM
) -> ca.SX:
    """
    Computes the state derivative dx/dt for the attitude error dynamics.

    State vector x: [q_err (4), omega_err (3), h_w (3)] (size 10).
    Input vector u: [u_rw (3), u_mag (3)] (size 6).

    Parameters
    ----------
    x : ca.SX
        State vector (error state).
    u : ca.SX
        Control input vector.
    B_orc : ca.SX
        Magnetic field vector in the Orbit Reference Frame (ORC) (3x1).
    omega_c : Union[np.ndarray, ca.SX]
        Orbital angular velocity vector (3x1).
    J_hat : Union[np.ndarray, ca.SX, ca.DM]
        Estimated inertia tensor of the satellite (3x3).

    Returns
    -------
    ca.SX
        State derivative dx/dt (10x1).
    """
    q_err = x[:4]
    omega_err = x[4:7]
    h_w = x[7:10]

    u_mag = u[:3]
    u_rw = u[3:]

    omega_c_b = quaternion_rotation(q_err, omega_c)
    omega = omega_err + omega_c_b

    d_omega = rotational_dynamics(omega, u_rw, u_mag, h_w, quaternion_rotation(q_err, B_orc), J_hat)

    d_q_err = kinematics(q_err, omega_err)
    d_omega_err = d_omega + ca.cross(omega_err, omega_c_b)
    d_h_w = -u_rw

    return ca.vertcat(d_q_err, d_omega_err, d_h_w)


def build_reduced_error_dynamics(
    dt: float, omega_c: np.ndarray, J_hat: np.ndarray
) -> tuple[ca.Function, ca.Function, ca.Function]:
    """
    Builds the discrete-time linearized dynamics for the reduced attitude error system.

    This function performs RK4 integration symbolically to obtain discrete dynamics,
    linearizes the system around a symbolic operating point, and applies a coordinate
    change using the attitude Jacobian (Xi) to handle the quaternion constraint.

    Parameters
    ----------
    dt : float
        Sampling time step.
    omega_c : np.ndarray
        Orbital angular velocity vector (3x1).
    J_hat : np.ndarray
        Inertia tensor.

    Returns
    -------
    Tuple[ca.Function, ca.Function, ca.Function]
        - F: Discrete dynamics function `x_next = F(x, u, B_orc)`.
        - A_func: Linearized state matrix function `A_tilde = A_func(x, u, B_orc)`.
        - B_func: Linearized input matrix function `B_tilde = B_func(x, u, B_orc)`.
    """
    q = ca.SX.sym("q", 4)
    omega = ca.SX.sym("omega", 3)
    h_w = ca.SX.sym("h_w", 3)
    x = ca.vertcat(q, omega, h_w)

    u_rw = ca.SX.sym("u_rw", 3)
    u_mag = ca.SX.sym("u_mag", 3)
    u = ca.vertcat(u_mag, u_rw)

    B_orc = ca.SX.sym("B_orc", 3)

    # Define a lambda for the dynamics to pass to the integrator
    def dyn_fn(x_s, u_s, p_s):
        return error_dynamics(x_s, u_s, p_s, ca.SX(omega_c), ca.SX(J_hat))

    x_next = integrator(dyn_fn, x, u, B_orc, dt)

    A = ca.jacobian(x_next, x)

    B = ca.jacobian(x_next, u)

    xi = attitude_jacobian(q)

    E = ca.diagcat(xi, ca.SX.eye(6))

    A_tilde = E.T @ A @ E

    B_tilde = E.T @ B

    A_func = ca.Function(
        "get_discrete_linearized_dynamics", [x, u, B_orc], [A_tilde], ["x_star", "u_star", "B_orc_star"], ["A_tilde"]
    )
    B_func = ca.Function(
        "get_discrete_linearized_dynamics", [x, u, B_orc], [B_tilde], ["x_star", "u_star", "B_orc_star"], ["B_tilde"]
    )
    F = ca.Function("discrete_dynamics", [x, u, B_orc], [x_next], ["x", "u", "B_orc"], ["x_next"])

    return F, A_func, B_func


def build_reduced_system_dynamics(dt: float, J_hat: np.ndarray) -> tuple[ca.Function, ca.Function, ca.Function]:
    """
    Builds the discrete-time linearized dynamics for the reduced attitude system.

    Parameters
    ----------
    dt : float
        Sampling time step.
    J_hat : np.ndarray
        Inertia tensor.

    Returns
    -------
    Tuple[ca.Function, ca.Function, ca.Function]
        - F: Discrete dynamics function `x_next = F(x, u, B_eci)`.
        - A_func: Linearized state matrix function `A_tilde = A_func(x, u, B_eci)`.
        - B_func: Linearized input matrix function `B_tilde = B_func(x, u, B_eci)`.
    """
    q_BI = ca.SX.sym("q", 4)
    omega = ca.SX.sym("omega", 3)
    h_w = ca.SX.sym("h_w", 3)
    x = ca.vertcat(q_BI, omega, h_w)

    u_rw = ca.SX.sym("u_rw", 3)
    u_mag = ca.SX.sym("u_mag", 3)
    u = ca.vertcat(u_mag, u_rw)

    B_eci = ca.SX.sym("B_eci", 3)

    # Define a lambda for the dynamics to pass to the integrator
    def dyn_fn(x_s, u_s, p_s):
        return satellite_dynamics(x_s, u_s, p_s, ca.SX(J_hat))

    x_next = integrator(dyn_fn, x, u, B_eci, dt)

    A = ca.jacobian(x_next, x)

    B = ca.jacobian(x_next, u)

    xi = attitude_jacobian(q_BI)

    E = ca.diagcat(xi, ca.SX.eye(6))

    A_tilde = E.T @ A @ E

    B_tilde = E.T @ B

    A_func = ca.Function(
        "get_discrete_linearized_dynamics", [x, u, B_eci], [A_tilde], ["x_star", "u_star", "B_eci_star"], ["A_tilde"]
    )
    B_func = ca.Function(
        "get_discrete_linearized_dynamics", [x, u, B_eci], [B_tilde], ["x_star", "u_star", "B_eci_star"], ["B_tilde"]
    )
    F = ca.Function("discrete_dynamics", [x, u, B_eci], [x_next], ["x", "u", "B_eci"], ["x_next"])

    return F, A_func, B_func
