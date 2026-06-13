# ruff: noqa: N803, N806
"""Attitude controllers driving reaction wheels (+ magnetorquers) for nadir pointing.

These controllers consume the estimator's ``x_hat`` and the nadir reference and return the control
input ``u`` that the simulation feeds straight to the actuator effectors. Because
:class:`~spacecraft.effector.ReactionWheelArray` and :class:`~spacecraft.effector.MagnetorquerArray`
interpret their command slice as **current commands** (amperes), the feedback law's desired control
*torque* is converted to currents with :func:`to_current_commands` before being returned -- mirroring
the legacy ``PI.calc_input_cmds`` flow.

``x_hat`` layout (see :mod:`spacecraft.estimator`)::

    [ r(3), v(3), q(4), omega(3), b_body(3), h_wheel(3) ]   # length 19
      0:3   3:6   6:10  10:13     13:16      16:19

where ``b_body`` is the estimated magnetic field in the body frame [T] (needed for magnetorquer
allocation) and ``h_wheel`` the estimated reaction-wheel angular momentum in the body frame
[N*m*s] (dumped by the magnetorquers). The reference is ``[q_des(4), omega_des(3)]``.
"""

import dataclasses
import datetime
from typing import Any, Self

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike

from simulate.controller import Controller

from .controller_models import build_reduced_system_dynamics
from .frames import orbital_rate, orc_from_orbit
from .orbit_dynamics import MU, SGP4
from .quaternion import Quaternion
from .signals import CONTROL, ESTIMATE, REFERENCE

_EPS = 1e-12


def _gain_matrix(value: ArrayLike) -> np.ndarray:
    """Coerce a gain to a (3, 3) matrix: scalar -> k*I, length-3 -> diag, else as given."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr) * np.eye(3)
    if arr.ndim == 1 and arr.shape[0] == 3:  # noqa: PLR2004
        return np.diag(arr)
    return arr.reshape(3, 3)


def _attitude_error(ref: np.ndarray, x_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Attitude and rate error of the body relative to the nadir (orbital/ORC) frame.

    The reference ``[q_bo(4), omega_bo(3)]`` is expressed **relative to the orbital frame**: ``q_bo``
    is the desired ORC->body rotation (identity for nadir pointing) and ``omega_bo`` the desired rate
    *relative to* ORC (zero for nadir). The orbital frame is reconstructed from the estimated orbit
    ``r, v`` carried in ``x_hat`` (the controller never sees the inertial reference directly):

    * the body's actual ORC->body rotation is ``q_bo_act = q_bi (x) q_oi^-1`` with
      ``q_oi = orc_from_orbit(r, v)`` (inertial->ORC), so the body-frame attitude error is
      ``q_err = q_bo_act (x) q_bo^-1`` (identity when the body is at the reference attitude),
    * the orbital feedforward body rate is ``omega_des = q_bo_act.apply(orbital_rate(r, v)) + omega_bo``
      (the ORC frame's rate rotated into the body frame, plus the reference's ORC-relative rate).

    Returns ``(q_err_vec(3), delta_omega(3))`` where ``q_err`` is the small-angle attitude error and
    ``delta_omega = omega - omega_des`` the body-rate error, both in the body frame.
    """
    r = x_hat[ESTIMATE.r]
    v = x_hat[ESTIMATE.v]
    q_bi = Quaternion.from_array(x_hat[ESTIMATE.q])
    omega = x_hat[ESTIMATE.omega]

    q_oi = orc_from_orbit(r, v)
    q_bo_act = q_bi * q_oi.conjugate()
    q_bo_des = Quaternion.from_array(ref[REFERENCE.q_des])
    q_err = q_bo_act.error_to(q_bo_des)
    q_err_vec = q_err.vec * np.sign(q_err.scalar)  # take the short rotation path

    omega_des = q_bo_act.apply(orbital_rate(r, v)) + ref[REFERENCE.omega_des]
    return q_err_vec, omega - omega_des


def allocation_matrix(axes: ArrayLike, constants: ArrayLike) -> np.ndarray:
    """Actuator allocation matrix ``Alpha`` with ``Alpha[:, k] = constant_k * axis_k`` (shape (3, N)).

    For reaction wheels ``constants`` are the torque constants and ``Alpha @ i`` is the
    (negated) body torque; for magnetorquers ``constants`` are the dipole constants and
    ``Alpha @ i`` is the body-frame dipole moment.
    """
    axes_arr = np.asarray(axes, dtype=float)
    axes_arr = axes_arr / np.linalg.norm(axes_arr, axis=1, keepdims=True)
    const_arr = np.asarray(constants, dtype=float)
    return (axes_arr * const_arr[:, None]).T


def _solve_allocation(alpha: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``alpha @ i = rhs`` for the current vector (exact solve when square, else least squares)."""
    if alpha.shape[0] == alpha.shape[1]:
        return np.linalg.solve(alpha, rhs)
    return np.linalg.lstsq(alpha, rhs, rcond=None)[0]


def to_current_commands(
    tau_rw: np.ndarray,
    tau_mtq: np.ndarray,
    b_body: np.ndarray,
    alpha_rw: np.ndarray,
    alpha_mtq: np.ndarray | None,
) -> np.ndarray:
    """Convert desired control torques to actuator current commands.

    Reaction wheels produce a body torque ``-alpha_rw @ i_rw``, so the currents achieving the
    desired ``tau_rw`` solve ``alpha_rw @ i_rw = -tau_rw``. Magnetorquers produce ``m x B`` with
    dipole ``m = alpha_mtq @ i_mtq``; the minimum-norm dipole for ``tau_mtq`` is
    ``m_cmd = (B x tau_mtq) / |B|^2``, then ``alpha_mtq @ i_mtq = m_cmd``.

    Returns ``[i_mtq, i_rw]`` (magnetorquers first) to match the effector composition order; when
    ``alpha_mtq`` is ``None`` only the reaction-wheel currents are returned.

    Parameters
    ----------
    tau_rw, tau_mtq : np.ndarray
        Desired reaction-wheel and magnetorquer body torques [N*m], shape (3,).
    b_body : np.ndarray
        Magnetic field in the body frame [T], shape (3,).
    alpha_rw : np.ndarray
        Reaction-wheel allocation matrix, shape (3, N).
    alpha_mtq : np.ndarray | None
        Magnetorquer allocation matrix, shape (3, M), or ``None`` if there are no magnetorquers.

    Returns
    -------
    np.ndarray
        Concatenated current commands ``[i_mtq, i_rw]`` (or just ``i_rw``).
    """
    i_rw = _solve_allocation(alpha_rw, -tau_rw)
    if alpha_mtq is None:
        return i_rw

    b_norm = float(np.linalg.norm(b_body))
    if b_norm < _EPS:
        i_mtq = np.zeros(alpha_mtq.shape[1])
    else:
        m_cmd = np.cross(b_body, tau_mtq) / b_norm**2
        i_mtq = _solve_allocation(alpha_mtq, m_cmd)
    return np.concatenate([i_mtq, i_rw])


@dataclasses.dataclass(frozen=True)
class QuaternionFeedbackControllerLog:
    """Internal log for :class:`QuaternionFeedbackController`."""

    q_err: np.ndarray
    tau_rw: np.ndarray
    tau_mtq: np.ndarray
    currents: np.ndarray


class QuaternionFeedbackController(Controller[QuaternionFeedbackControllerLog]):
    """Quaternion-feedback PD attitude control with optional magnetorquer momentum dumping.

    The reaction wheels track the reference attitude with ``tau_rw = -Kp q_err - Kd (omega -
    omega_des)`` and the magnetorquers bleed off stored wheel momentum
    with ``tau_mtq = -k_m h_wheel``. Both torques are allocated to actuator currents via
    :func:`to_current_commands`.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        kp: ArrayLike,
        kd: ArrayLike,
        alpha_rw: np.ndarray,
        alpha_mtq: np.ndarray,
        k_m: float = 0.0,
    ) -> None:
        """Initialize with the sample time, PD gains, actuator allocation matrices and dumping gain."""
        super().__init__(dt)
        self.kp = _gain_matrix(kp)
        self.kd = _gain_matrix(kd)
        self.alpha_rw = alpha_rw
        self.alpha_mtq = alpha_mtq
        self.k_m = float(k_m)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from config (PD gains + reaction-wheel/magnetorquer allocation blocks)."""
        rw_cfg = config["reaction_wheels"]
        alpha_rw = allocation_matrix(rw_cfg["axes"], rw_cfg["torque_constant"])

        mtq_cfg = config["magnetorquers"]
        alpha_mtq = allocation_matrix(mtq_cfg["axes"], mtq_cfg["dipole_constant"])

        return cls(
            dt=float(config["dt"]),
            kp=config["kp"],
            kd=config["kd"],
            alpha_rw=alpha_rw,
            alpha_mtq=alpha_mtq,
            k_m=float(config.get("k_m", 0.0)),
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,
        x_hat: float | np.ndarray,
    ) -> tuple[float | np.ndarray, QuaternionFeedbackControllerLog]:
        """Compute the quaternion-feedback control current commands."""
        x = np.asarray(x_hat)

        b_body = x[ESTIMATE.b_body]
        h_wheel = x[ESTIMATE.h_wheel]

        q_err, delta_omega = _attitude_error(np.asarray(ref), x)
        tau_rw = -self.kp @ q_err - self.kd @ delta_omega
        tau_mtq = -self.k_m * h_wheel

        u = to_current_commands(tau_rw, tau_mtq, b_body, self.alpha_rw, self.alpha_mtq)
        return u, QuaternionFeedbackControllerLog(q_err=q_err, tau_rw=tau_rw, tau_mtq=tau_mtq, currents=u)


def _ensure_utc(epoch: datetime.datetime) -> datetime.datetime:
    """Return ``epoch`` as a timezone-aware UTC datetime (naive inputs are assumed UTC)."""
    return epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=datetime.UTC)


def average_rate(
    propagator: SGP4,
    epoch: datetime.datetime,
    n_samples: int = 24,
) -> np.ndarray:
    """Orbit-averaged magnetic field (in the nadir/ORC frame) and reference rate over one orbit.

    Samples the SGP4 orbit ``n_samples`` times over one orbital period (estimated from the initial
    state via vis-viva) and averages the IGRF field rotated into the nadir-pointing frame together
    with the orbital rate. These feed the field-averaged LQR model.

    Parameters
    ----------
    propagator : SGP4
        Orbit propagator.
    epoch : datetime.datetime
        Reference time (``t = 0``).
    n_samples : int
        Number of samples over the orbit.

    Returns
    -------
    np.ndarray
        ``omega_c`` -- averaged rate [rad/s], each shape (3,).
    """
    epoch = _ensure_utc(epoch)
    r0, v0 = propagator.propagate(epoch)
    a = 1.0 / (2.0 / np.linalg.norm(r0) - float(np.dot(v0, v0)) / MU)
    period = 2.0 * np.pi * np.sqrt(a**3 / MU)

    w_acc = np.zeros(3)
    for k in range(n_samples):
        t_k = epoch + datetime.timedelta(seconds=period * k / n_samples)
        r, v = propagator.propagate(t_k)
        w_acc += orbital_rate(r, v)

    return w_acc / n_samples


def _dlqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discrete LQR gain ``K`` and Riccati solution ``P`` for ``min sum x'Qx + u'Ru``."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return K, P


def _dlqr_warm_start(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update the LQR gain using a single Newton-Kleinman iteration (Warm Start).

    Parameters
    ----------
    A : np.ndarray
        Current time-varying state transition matrix A(t)
    B : np.ndarray
        Current time-varying input matrix B(t).
    Q : np.ndarray
        State cost matrix.
    R : np.ndarray
        Input cost matrix.
    P : np.ndarray
        The solution P from the previous time step.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        K_new: The updated control gain.
        P_new: The updated Riccati solution.
    """
    BtP = B.T @ P
    R_total = R + BtP @ B
    K_0 = np.linalg.solve(R_total, BtP @ A)

    A_cl = A - B @ K_0
    S = Q + K_0.T @ R @ K_0
    P_new = scipy.linalg.solve_discrete_lyapunov(A_cl.T, S)

    BtP_new = B.T @ P_new

    R_total_new = R + BtP_new @ B
    K_new = np.linalg.solve(R_total_new, BtP_new @ A)

    return K_new, P_new


@dataclasses.dataclass(frozen=True)
class AdaptiveLQRLog:
    """Internal log for the LQR controllers."""

    error: np.ndarray
    dipole: np.ndarray
    tau_rw: np.ndarray
    currents: np.ndarray


class AdaptiveLQR(Controller[AdaptiveLQRLog]):
    """LQR that re-solves its gain each step to deal with the model changing.

    The reduced model is rebuilt with the magnetic field carried in ``x_hat`` and the Riccati
    equation re-solved with a Newton-Kleinman warm start from the previous solution, so the gain
    adapts as ``B`` changes around the orbit.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        Q: ArrayLike,
        R: ArrayLike,
        inertia: ArrayLike,
        omega_c: ArrayLike,
        alpha_rw: ArrayLike,
        alpha_mtq: ArrayLike,
    ) -> None:
        """Initialize and solve the LQR gain for the given weights and (averaged) model."""
        super().__init__(dt)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.inertia = np.asarray(inertia, dtype=float)
        self.omega_c = np.asarray(omega_c, dtype=float)
        self.alpha_rw = np.asarray(alpha_rw, dtype=float)
        self.alpha_mtq = np.asarray(alpha_mtq, dtype=float)
        self.n_inputs = self.alpha_mtq.shape[0] + self.alpha_rw.shape[0]

        self.P = None

        _, self.A_func, self.B_func = build_reduced_system_dynamics(dt, self.inertia)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from config (weights, inertia, model field/rate, actuator allocation)."""
        tle1, tle2 = config["tle"]
        epoch = datetime.datetime.fromisoformat(config["epoch"])
        omega_c = average_rate(SGP4.from_tle(tle1, tle2), epoch)

        rw_cfg = config["reaction_wheels"]
        mtq_cfg = config["magnetorquers"]
        return cls(
            dt=float(config["dt"]),
            Q=np.asarray(config["Q"], dtype=float),
            R=np.asarray(config["R"], dtype=float),
            inertia=np.asarray(config["inertia"], dtype=float),
            omega_c=omega_c,
            alpha_rw=allocation_matrix(rw_cfg["axes"], rw_cfg["torque_constant"]),
            alpha_mtq=allocation_matrix(mtq_cfg["axes"], mtq_cfg["dipole_constant"]),
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,
        x_hat: float | np.ndarray,
    ) -> tuple[float | np.ndarray, AdaptiveLQRLog]:
        """Re-solve the gain at the current field, then compute the LQR current commands."""
        ref_arr = np.asarray(ref)
        x = np.asarray(x_hat)
        r = x[ESTIMATE.r]
        v = x[ESTIMATE.v]
        q_bi = Quaternion.from_array(x[ESTIMATE.q])
        omega = x[ESTIMATE.omega]
        h_w = x[ESTIMATE.h_wheel]

        q_oi = orc_from_orbit(r, v)
        q_bo_act = q_bi * q_oi.conjugate()

        b_body = x[ESTIMATE.b_body]
        b_eci = q_bi.conjugate().apply(b_body)

        q_bo_ref = Quaternion.from_array(ref_arr[REFERENCE.q_des])
        q_bi_ref = (q_bo_ref * q_oi).to_array()
        omega_ref = q_bo_act.apply(orbital_rate(r, v)) + ref_arr[REFERENCE.omega_des]
        h_w_ref = -self.inertia @ omega_ref

        x_ref = np.concatenate((q_bi_ref, omega_ref, h_w_ref))
        u_ref = np.zeros(self.n_inputs)

        A = np.array(self.A_func(x_ref, u_ref, b_eci))
        B = np.array(self.B_func(x_ref, u_ref, b_eci))

        self.K, self.P = (
            _dlqr_gain(A, B, self.Q, self.R) if self.P is None else _dlqr_warm_start(A, B, self.Q, self.R, self.P)
        )

        q_err = q_bo_act.error_to(q_bo_ref)
        q_err_vec = q_err.vec * np.sign(q_err.scalar)
        omega_err = omega - omega_ref
        h_w_err = h_w - h_w_ref
        error = np.concatenate([q_err_vec, omega_err, h_w_err])

        control = -self.K @ error

        u = to_current_commands(
            tau_rw=control[CONTROL.tau_rw],
            tau_mtq=control[CONTROL.tau_mtq],
            b_body=b_body,
            alpha_rw=self.alpha_rw,
            alpha_mtq=self.alpha_mtq,
        )

        b_norm_sq = np.dot(b_body, b_body)
        dipole = np.cross(b_body, control[CONTROL.tau_mtq]) / b_norm_sq if b_norm_sq > _EPS else np.zeros(3)

        return u, AdaptiveLQRLog(
            error=error,
            dipole=dipole,
            tau_rw=control[CONTROL.tau_rw],
            currents=u,
        )
