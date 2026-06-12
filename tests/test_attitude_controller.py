"""Tests for the Phase-5 attitude controllers and their supporting linearization."""

import numpy as np

from rigid_body.controller import (
    AdaptiveLQRController,
    LQRController,
    QuaternionFeedbackController,
    allocation_matrix,
    to_current_commands,
)
from rigid_body.effector import EarthGravity, MagnetorquerArray, ReactionWheelArray
from rigid_body.frames import orc_from_orbit
from rigid_body.linearization import discrete_jacobians, reduced_model, rk2_step
from rigid_body.orbit_dynamics import MU
from rigid_body.quaternion import Quaternion
from rigid_body.rigid_body import RigidBodyDynamics

_AXES = np.eye(3)
_KT = 0.02  # reaction-wheel torque constant
_KM = 1.5  # magnetorquer dipole constant
_INERTIA = np.diag([4.0, 5.0, 6.0])
_B_BODY = np.array([1.8e-5, -1.2e-5, 3.0e-5])  # representative LEO field [T]
_R0 = np.array([7.0e6, 0.0, 0.0])  # LEO position [m]
_V0 = np.array([0.0, float(np.sqrt(MU / 7.0e6)), 0.0])  # circular velocity [m/s]


def _make_plant(*, rw_initial_omega: np.ndarray | None = None) -> tuple[RigidBodyDynamics, ReactionWheelArray]:
    """Rigid body on a LEO orbit: magnetorquers, then reaction wheels (the [i_mtq, i_rw] order), then gravity.

    ``EarthGravity`` makes the integrated position follow a Keplerian orbit so the nadir (ORC) frame
    -- which the controllers reconstruct from ``x_hat``'s ``r, v`` -- evolves physically.
    """
    mtq = MagnetorquerArray(axes=_AXES, dipole_constant=_KM, time_constant=0.3, max_current=50.0, b_field_model=_B_BODY)
    rw = ReactionWheelArray(
        axes=_AXES,
        inertia=0.01,
        torque_constant=_KT,
        time_constant=0.3,
        max_current=5.0,
        max_rpm=8000.0,
        initial_omega=rw_initial_omega,
    )
    dynamics = RigidBodyDynamics(dt=0.1, mass=50.0, inertia=_INERTIA, effectors=[mtq, rw, EarthGravity()])
    dynamics.x[0:3] = _R0
    dynamics.x[3:6] = _V0
    return dynamics, rw


def _nadir_angle(dynamics: RigidBodyDynamics) -> float:
    """Geodesic angle [rad] between the body attitude and the nadir (ORC) frame from the plant state."""
    q_oi = orc_from_orbit(dynamics.x[0:3], dynamics.x[3:6])
    q_err = Quaternion.from_array(dynamics.x[6:10]).error_to(q_oi)  # desired q_bo = identity (nadir)
    return float(2.0 * np.arctan2(np.linalg.norm(q_err.vec), abs(q_err.scalar)))


def _wheel_momentum(dynamics: RigidBodyDynamics, rw: ReactionWheelArray, omega: np.ndarray) -> np.ndarray:
    """Body-frame reaction-wheel momentum from the plant state (rw is the second effector)."""
    sl = dynamics._state_slices[1]  # noqa: SLF001
    omega_rel = dynamics.x[sl][rw.n_inputs :]
    return rw.axes.T @ (rw.inertia * (omega_rel + rw.axes @ omega))


def _x_hat(dynamics: RigidBodyDynamics, rw: ReactionWheelArray) -> np.ndarray:
    """Assemble a 19-element x_hat from the true plant state (estimator stand-in)."""
    omega = dynamics.x[10:13]
    h_wheel = _wheel_momentum(dynamics, rw, omega)
    return np.concatenate([dynamics.x[0:13], _B_BODY, h_wheel])


# --------------------------------------------------------------------------------------------- #
# Allocation helper (ported to_current_commands)
# --------------------------------------------------------------------------------------------- #
def test_to_current_commands_round_trip() -> None:
    alpha_rw = allocation_matrix(_AXES, _KT * np.ones(3))
    alpha_mtq = allocation_matrix(_AXES, _KM * np.ones(3))
    tau_rw = np.array([0.10, -0.20, 0.05])
    tau_mtq = np.array([1e-4, 2e-4, -1.5e-4])

    u = to_current_commands(tau_rw, tau_mtq, _B_BODY, alpha_rw, alpha_mtq)
    i_mtq, i_rw = u[:3], u[3:]

    # Reaction wheels reproduce the desired torque exactly (body torque = -alpha @ i).
    np.testing.assert_allclose(-alpha_rw @ i_rw, tau_rw, rtol=1e-12)

    # Magnetorquers reproduce the component of the desired torque perpendicular to B.
    b_hat = _B_BODY / np.linalg.norm(_B_BODY)
    tau_perp = tau_mtq - np.dot(tau_mtq, b_hat) * b_hat
    np.testing.assert_allclose(np.cross(alpha_mtq @ i_mtq, _B_BODY), tau_perp, atol=1e-12)


def test_to_current_commands_zero_field_skips_magnetorquers() -> None:
    alpha_rw = allocation_matrix(_AXES, _KT * np.ones(3))
    alpha_mtq = allocation_matrix(_AXES, _KM * np.ones(3))
    u = to_current_commands(np.ones(3), np.ones(3), np.zeros(3), alpha_rw, alpha_mtq)
    np.testing.assert_array_equal(u[:3], np.zeros(3))  # no field -> no dipole commanded


# --------------------------------------------------------------------------------------------- #
# Step 5.1 - QuaternionFeedbackController
# --------------------------------------------------------------------------------------------- #
def _quaternion_controller(*, k_m: float) -> QuaternionFeedbackController:
    return QuaternionFeedbackController(
        dt=0.1,
        kp=0.6,
        kd=3.0,
        alpha_rw=allocation_matrix(_AXES, _KT * np.ones(3)),
        alpha_mtq=allocation_matrix(_AXES, _KM * np.ones(3)),
        k_m=k_m,
    )


def test_quaternion_feedback_drives_attitude_error_to_zero() -> None:
    dynamics, rw = _make_plant()
    # Start ~10 deg off nadir (a small body-frame offset from the ORC frame).
    offset = Quaternion.from_array(np.array([0.06, -0.05, 0.04, 1.0]))
    q0 = offset * orc_from_orbit(_R0, _V0)
    dynamics.x[6:10] = q0.to_array() / np.linalg.norm(q0.to_array())

    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # q_bo = identity (nadir), zero orbit-relative rate
    controller = _quaternion_controller(k_m=0.0)  # dumping off

    angle = np.inf
    for k in range(800):
        t = k * 0.1
        u, _ = controller.evaluate(t, ref, _x_hat(dynamics, rw))
        dynamics.evaluate(t, u)
        angle = _nadir_angle(dynamics)

    assert angle < np.deg2rad(1.0)


def test_quaternion_feedback_dumping_bounds_wheel_momentum() -> None:
    # Spin the wheels up so there is momentum to dump.
    dynamics, rw = _make_plant(rw_initial_omega=np.array([200.0, -150.0, 100.0]))
    dynamics.x[6:10] = orc_from_orbit(_R0, _V0).to_array()  # start pointed at nadir
    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    controller = _quaternion_controller(k_m=5e-4)

    h0 = np.linalg.norm(_wheel_momentum(dynamics, rw, dynamics.x[10:13]))
    for k in range(4000):
        t = k * 0.1
        u, _ = controller.evaluate(t, ref, _x_hat(dynamics, rw))
        dynamics.evaluate(t, u)
    h_final = np.linalg.norm(_wheel_momentum(dynamics, rw, dynamics.x[10:13]))

    assert h_final < h0  # magnetorquer dumping bled off stored momentum


# --------------------------------------------------------------------------------------------- #
# Step 5.2 - Linearization
# --------------------------------------------------------------------------------------------- #
def test_linearization_jacobian_matches_nonlinear_step() -> None:
    omega_c = np.array([0.0, -1.0e-3, 0.0])
    dt = 0.1
    inertia_inv = np.linalg.inv(_INERTIA)
    a_full, b_full = discrete_jacobians(_B_BODY, dt, omega_c, _INERTIA)

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    u0 = np.zeros(6)
    rng = np.random.default_rng(0)
    dx = 1e-4 * rng.standard_normal(7)
    dx[0:4] -= np.dot(dx[0:4], x0[0:4]) * x0[0:4]  # keep the perturbation on the unit sphere
    du = 1e-4 * rng.standard_normal(6)

    f0 = rk2_step(x0, u0, _B_BODY, dt, omega_c, _INERTIA, inertia_inv)
    fx = rk2_step(x0 + dx, u0, _B_BODY, dt, omega_c, _INERTIA, inertia_inv)
    fu = rk2_step(x0, u0 + du, _B_BODY, dt, omega_c, _INERTIA, inertia_inv)

    np.testing.assert_allclose(a_full @ dx, fx - f0, atol=1e-8)
    np.testing.assert_allclose(b_full @ du, fu - f0, atol=1e-8)


# --------------------------------------------------------------------------------------------- #
# Step 5.3 - LQRController
# --------------------------------------------------------------------------------------------- #
def _lqr_kwargs(b_field: np.ndarray) -> dict:
    return {
        "dt": 0.1,
        "Q": np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0]),
        "R": np.eye(6),
        "inertia": _INERTIA,
        "omega_c": np.array([0.0, -1.0e-3, 0.0]),
        "b_field": b_field,
        "alpha_rw": allocation_matrix(_AXES, _KT * np.ones(3)),
        "alpha_mtq": allocation_matrix(_AXES, _KM * np.ones(3)),
    }


def test_lqr_closed_loop_is_stable() -> None:
    controller = LQRController(**_lqr_kwargs(_B_BODY))
    eigvals = np.linalg.eigvals(controller.A - controller.B @ controller.K)
    assert np.all(np.abs(eigvals) < 1.0)


def test_lqr_drives_nonlinear_plant_to_tolerance() -> None:
    controller = LQRController(**_lqr_kwargs(_B_BODY))
    dynamics, rw = _make_plant()
    offset = Quaternion.from_array(np.array([0.05, -0.04, 0.03, 1.0]))
    q0 = offset * orc_from_orbit(_R0, _V0)
    dynamics.x[6:10] = q0.to_array() / np.linalg.norm(q0.to_array())
    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    angle = np.inf
    for k in range(800):
        t = k * 0.1
        u, _ = controller.evaluate(t, ref, _x_hat(dynamics, rw))
        dynamics.evaluate(t, u)
        angle = _nadir_angle(dynamics)

    assert angle < np.deg2rad(2.0)


# --------------------------------------------------------------------------------------------- #
# Step 5.4 - AdaptiveLQRController
# --------------------------------------------------------------------------------------------- #
def test_adaptive_lqr_matches_static_lqr() -> None:
    static = LQRController(**_lqr_kwargs(_B_BODY))
    adaptive = AdaptiveLQRController(**_lqr_kwargs(_B_BODY))

    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    x_hat = np.concatenate([_R0, _V0, [0.0, 0.0, 0.0, 1.0], np.zeros(3), _B_BODY, np.zeros(3)])
    adaptive.update(0.0, ref, x_hat)

    np.testing.assert_allclose(adaptive.K, static.K, rtol=1e-6, atol=1e-9)


def test_adaptive_lqr_adapts_when_field_changes() -> None:
    adaptive = AdaptiveLQRController(**_lqr_kwargs(_B_BODY))
    k_initial = adaptive.K.copy()

    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    b_other = np.array([3.0e-5, 2.0e-5, -1.0e-5])
    x_hat = np.concatenate([_R0, _V0, [0.0, 0.0, 0.0, 1.0], np.zeros(3), b_other, np.zeros(3)])
    adaptive.update(0.0, ref, x_hat)

    assert not np.allclose(adaptive.K, k_initial)
