r"""Reduced discrete-time error dynamics and its linearization for the LQR controllers.

Ported from the legacy ``controller_models.py`` (CasADi) to plain NumPy. The nonlinear attitude
error dynamics are integrated one step with RK2 (quaternion renormalized), then differentiated by
central finite differences to obtain the discrete Jacobians ``A`` and ``B`` about the regulation
point. Finally the 4-component error-quaternion block is collapsed to a 3-component small-angle
state with the attitude Jacobian :math:`\Xi`, giving the reduced model

.. math:: \tilde x_{k+1} = \tilde A \tilde x_k + \tilde B u_k

with state ``[delta_theta(3), delta_omega(3)]`` (6) and input ``[m(3), tau_rw(3)]`` (6), where ``m``
is the magnetorquer dipole and ``tau_rw`` the reaction-wheel body torque. The magnetic field ``B``
enters the input matrix through the magnetorquer torque ``m x B``, which is what the adaptive
controller re-solves on as ``B`` changes around the orbit. The reaction wheels make the model fully
controllable, so the discrete Riccati equation is well-posed for a fixed field (a momentum state is
deliberately *not* included: with a frozen ``B`` the total-momentum component along ``B`` is
uncontrollable -- magnetic momentum dumping is handled by
:class:`~rigid_body.controller.QuaternionFeedbackController`).
"""

from collections.abc import Callable

import numpy as np

from .quaternion import Quaternion

_STATE = 7  # full error state: [q_err(4), delta_omega(3)]
_INPUT = 6  # [m(3), tau_rw(3)]
_REDUCED = 6  # [delta_theta(3), delta_omega(3)]
_FD_STEP = 1e-6


def error_dynamics(  # noqa: PLR0913
    x: np.ndarray,
    u: np.ndarray,
    b_field: np.ndarray,
    omega_c: np.ndarray,
    inertia: np.ndarray,
    inertia_inv: np.ndarray,
) -> np.ndarray:
    """Continuous-time attitude error dynamics ``x_dot = f(x, u, B)``.

    Parameters
    ----------
    x : np.ndarray
        Error state ``[q_err(4), delta_omega(3)]``, shape (7,).
    u : np.ndarray
        Input ``[m(3), tau_rw(3)]`` (magnetorquer dipole, reaction-wheel body torque), shape (6,).
    b_field : np.ndarray
        Magnetic field in the body frame [T], shape (3,).
    omega_c : np.ndarray
        Reference (orbital) angular velocity [rad/s], shape (3,).
    inertia, inertia_inv : np.ndarray
        Spacecraft inertia and its inverse, shape (3, 3).

    Returns
    -------
    np.ndarray
        State derivative, shape (7,).
    """
    q_err = x[0:4]
    delta_omega = x[4:7]
    m = u[0:3]
    tau_rw = u[3:6]

    omega = delta_omega + omega_c
    torque = np.cross(m, b_field) + tau_rw
    omega_dot = inertia_inv @ (torque - np.cross(omega, inertia @ omega))
    q_dot = Quaternion.from_array(q_err).kinematics(delta_omega)
    return np.concatenate([q_dot, omega_dot])


def rk2_step(  # noqa: PLR0913
    x: np.ndarray,
    u: np.ndarray,
    b_field: np.ndarray,
    dt: float,
    omega_c: np.ndarray,
    inertia: np.ndarray,
    inertia_inv: np.ndarray,
) -> np.ndarray:
    """Advance the error state one step with RK2 and renormalize the error quaternion."""
    k1 = error_dynamics(x, u, b_field, omega_c, inertia, inertia_inv)
    k2 = error_dynamics(x + dt * k1, u, b_field, omega_c, inertia, inertia_inv)
    x_next = x + 0.5 * dt * (k1 + k2)
    x_next[0:4] = x_next[0:4] / np.linalg.norm(x_next[0:4])
    return x_next


def _jacobian(f: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, n_out: int) -> np.ndarray:
    """Central finite-difference Jacobian of ``f`` at ``x0`` (output dimension ``n_out``)."""
    jac = np.zeros((n_out, x0.shape[0]))
    for i in range(x0.shape[0]):
        dx = np.zeros_like(x0)
        dx[i] = _FD_STEP
        jac[:, i] = (f(x0 + dx) - f(x0 - dx)) / (2.0 * _FD_STEP)
    return jac


def discrete_jacobians(
    b_field: np.ndarray,
    dt: float,
    omega_c: np.ndarray,
    inertia: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Full discrete Jacobians ``(A, B)`` of the one-step dynamics about the regulation point.

    The regulation point is the identity error quaternion, zero rate error and zero input.
    ``A`` is (7, 7) and ``B`` is (7, 6).
    """
    inertia_inv = np.linalg.inv(inertia)

    x_star = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    u_star = np.zeros(_INPUT)

    a_full = _jacobian(lambda x: rk2_step(x, u_star, b_field, dt, omega_c, inertia, inertia_inv), x_star, _STATE)
    b_full = _jacobian(lambda u: rk2_step(x_star, u, b_field, dt, omega_c, inertia, inertia_inv), u_star, _STATE)
    return a_full, b_full


def reduced_model(
    b_field: np.ndarray,
    dt: float,
    omega_c: np.ndarray,
    inertia: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduced discrete model ``(A_tilde, B_tilde)`` (6x6, 6x6) via the attitude-Jacobian reduction.

    Parameters
    ----------
    b_field : np.ndarray
        Magnetic field in the body frame [T], shape (3,).
    dt : float
        Discretization step [s].
    omega_c : np.ndarray
        Reference (orbital) angular velocity [rad/s], shape (3,).
    inertia : np.ndarray
        Spacecraft inertia, shape (3, 3).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The reduced ``(A_tilde, B_tilde)`` matrices.
    """
    a_full, b_full = discrete_jacobians(b_field, dt, omega_c, inertia)
    xi = Quaternion.from_array(np.array([0.0, 0.0, 0.0, 1.0])).xi
    e = np.zeros((_STATE, _REDUCED))
    e[0:4, 0:3] = xi
    e[4:, 3:] = np.eye(3)
    a_tilde = e.T @ a_full @ e
    b_tilde = e.T @ b_full
    return a_tilde, b_tilde
