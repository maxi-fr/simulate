# ruff: noqa: N803, N806
"""Attitude controllers driving reaction wheels (+ magnetorquers) for nadir pointing.

These controllers consume the estimator's ``x_hat`` and the nadir reference and return the control
input ``u`` that the simulation feeds straight to the actuator effectors. Because
:class:`~rigid_body.effector.ReactionWheelArray` and :class:`~rigid_body.effector.MagnetorquerArray`
interpret their command slice as **current commands** (amperes), the feedback law's desired control
*torque* is converted to currents with :func:`to_current_commands` before being returned -- mirroring
the legacy ``PI.calc_input_cmds`` flow.

``x_hat`` layout (see :mod:`rigid_body.estimator`)::

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

from .environment import magnetic_field_vector
from .frames import eci_to_geodedic, orbital_rate, orc_from_orbit
from .linearization import reduced_model
from .orbit_dynamics import MU, SGP4
from .quaternion import Quaternion

_EPS = 1e-12

# Slices into x_hat.
_Q = slice(6, 10)
_W = slice(10, 13)
_B = slice(13, 16)
_H = slice(16, 19)


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
      ``q_err = q_bo^-1 (x) q_bo_act`` (identity when the body is at the reference attitude),
    * the orbital feedforward body rate is ``omega_des = q_bo_act.apply(orbital_rate(r, v)) + omega_bo``
      (the ORC frame's rate rotated into the body frame, plus the reference's ORC-relative rate).

    Returns ``(q_err_vec(3), delta_omega(3))`` where ``q_err`` is the small-angle attitude error and
    ``delta_omega = omega - omega_des`` the body-rate error, both in the body frame.
    """
    r = x_hat[0:3]
    v = x_hat[3:6]
    q_bi = Quaternion.from_array(x_hat[_Q])
    omega = x_hat[_W]

    q_oi = orc_from_orbit(r, v)
    q_bo_act = q_bi * q_oi.conjugate()
    q_bo_des = Quaternion.from_array(ref[0:4])
    q_err = q_bo_act.error_to(q_bo_des).vec

    omega_des = q_bo_act.apply(orbital_rate(r, v)) + ref[4:7]
    return q_err, omega - omega_des


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

        b_body = x[_B]
        h_wheel = x[_H]

        q_err, delta_omega = _attitude_error(np.asarray(ref), x)
        tau_rw = -self.kp @ q_err - self.kd @ delta_omega
        tau_mtq = -self.k_m * h_wheel

        u = to_current_commands(tau_rw, tau_mtq, b_body, self.alpha_rw, self.alpha_mtq)
        return u, QuaternionFeedbackControllerLog(q_err=q_err, tau_rw=tau_rw, tau_mtq=tau_mtq, currents=u)


def _ensure_utc(epoch: datetime.datetime) -> datetime.datetime:
    """Return ``epoch`` as a timezone-aware UTC datetime (naive inputs are assumed UTC)."""
    return epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=datetime.UTC)


def average_field_and_rate(
    propagator: SGP4,
    epoch: datetime.datetime,
    n_samples: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
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
    tuple[np.ndarray, np.ndarray]
        ``(b_field_orc, omega_c)`` -- averaged field [T] and rate [rad/s], each shape (3,).
    """
    epoch = _ensure_utc(epoch)
    r0, v0 = propagator.propagate(epoch)
    a = 1.0 / (2.0 / np.linalg.norm(r0) - float(np.dot(v0, v0)) / MU)
    period = 2.0 * np.pi * np.sqrt(a**3 / MU)

    b_acc = np.zeros(3)
    w_acc = np.zeros(3)
    for k in range(n_samples):
        t_k = epoch + datetime.timedelta(seconds=period * k / n_samples)
        r, v = propagator.propagate(t_k)
        q_orc = orc_from_orbit(r, v)
        lat, lon, alt = eci_to_geodedic(r)
        b_eci = magnetic_field_vector(t_k.replace(tzinfo=None), float(lat), float(lon), float(alt))
        b_acc += q_orc.apply(b_eci)
        w_acc += orbital_rate(r, v)
    return b_acc / n_samples, w_acc / n_samples


def _dlqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discrete LQR gain ``K`` and Riccati solution ``P`` for ``min sum x'Qx + u'Ru``."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return K, P


def _dlqr_warm_start(  # noqa: PLR0913
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    *,
    iters: int = 50,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-solve the discrete Riccati equation by Newton-Kleinman value iteration from ``P``."""
    for _ in range(iters):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P_next = A.T @ P @ A - A.T @ P @ B @ K + Q
        P_next = 0.5 * (P_next + P_next.T)
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return K, P


def _build_model_inputs(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ``(b_field, omega_c)`` for the LQR model from explicit values or a TLE."""
    if "tle" in config:
        tle1, tle2 = config["tle"]
        epoch = datetime.datetime.fromisoformat(config["epoch"])
        return average_field_and_rate(SGP4.from_tle(tle1, tle2), epoch)
    return np.asarray(config["b_field"], dtype=float), np.asarray(config["omega_c"], dtype=float)


@dataclasses.dataclass(frozen=True)
class LQRControllerLog:
    """Internal log for the LQR controllers."""

    error: np.ndarray
    dipole: np.ndarray
    tau_rw: np.ndarray
    currents: np.ndarray


class AdaptiveLQRController(Controller[LQRControllerLog]):
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
        b_field: ArrayLike,
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

        A, B = reduced_model(np.asarray(b_field, dtype=float), dt, self.omega_c, self.inertia)
        self.A = A
        self.B = B
        self.K, self.P = _dlqr_gain(A, B, self.Q, self.R)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from config (weights, inertia, model field/rate, actuator allocation)."""
        b_field, omega_c = _build_model_inputs(config)
        rw_cfg = config["reaction_wheels"]
        mtq_cfg = config["magnetorquers"]
        return cls(
            dt=float(config["dt"]),
            Q=np.asarray(config["Q"], dtype=float),
            R=np.asarray(config["R"], dtype=float),
            inertia=np.asarray(config["inertia"], dtype=float),
            omega_c=omega_c,
            b_field=b_field,
            alpha_rw=allocation_matrix(rw_cfg["axes"], rw_cfg["torque_constant"]),
            alpha_mtq=allocation_matrix(mtq_cfg["axes"], mtq_cfg["dipole_constant"]),
        )

    def _error(self, ref: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Reduced error state ``[delta_theta, delta_omega]`` (relative to the nadir/ORC frame)."""
        q_err, delta_omega = _attitude_error(ref, x)
        return np.concatenate([q_err, delta_omega])  # TODO: LQR feedback also on wheel momentum

    def _allocate(self, control: np.ndarray) -> np.ndarray:
        """Allocate the model input ``[m, tau_rw]`` to ``[i_mtq, i_rw]`` current commands."""
        i_mtq = _solve_allocation(self.alpha_mtq, control[0:3])
        i_rw = _solve_allocation(self.alpha_rw, -control[3:6])
        return np.concatenate([i_mtq, i_rw])

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,
        x_hat: float | np.ndarray,
    ) -> tuple[float | np.ndarray, LQRControllerLog]:
        """Re-solve the gain at the current field, then compute the LQR current commands."""
        x = np.asarray(x_hat)
        self.A, self.B = reduced_model(x[_B], self.dt, self.omega_c, self.inertia)
        self.K, self.P = _dlqr_warm_start(self.A, self.B, self.Q, self.R, self.P)

        error = self._error(np.asarray(ref), x)
        control = -self.K @ error
        u = self._allocate(control)
        return u, LQRControllerLog(error=error, dipole=control[0:3], tau_rw=control[3:6], currents=u)
