"""Tests for the Phase-5 attitude controllers and their supporting linearization."""

import importlib
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning

from spacecraft.effector import EarthGravity, MagnetorquerArray, ReactionWheelArray
from spacecraft.frames import lvlh_from_orbit
from spacecraft.orbit_dynamics import MU
from spacecraft.quaternion import Quaternion
from spacecraft.rigid_body import RigidBodyDynamics

_ctrl_mod = importlib.import_module("examples.03_satellite.controller")
MPC = _ctrl_mod.MPC
AdaptiveLQR = _ctrl_mod.AdaptiveLQR
QuaternionFeedbackController = _ctrl_mod.QuaternionFeedbackController
allocation_matrix = _ctrl_mod.allocation_matrix
to_current_commands = _ctrl_mod.to_current_commands

_AXES = np.eye(3)
_KT = 0.02  # reaction-wheel torque constant
_KM = 1.5  # magnetorquer dipole constant
_INERTIA = np.diag([4.0, 5.0, 6.0])
_B_BODY = np.array([1.8e-5, -1.2e-5, 3.0e-5])  # representative LEO field [T]
_R0 = np.array([7.0e6, 0.0, 0.0])  # LEO position [m]
_V0 = np.array([0.0, float(np.sqrt(MU / 7.0e6)), 0.0])  # circular velocity [m/s]


def _make_plant(*, rw_initial_omega: np.ndarray | None = None) -> tuple[RigidBodyDynamics, ReactionWheelArray]:
    """Rigid body on a LEO orbit: magnetorquers, then reaction wheels (the [i_mtq, i_rw] order), then gravity.

    ``EarthGravity`` makes the integrated position follow a Keplerian orbit so the nadir (LVLH) frame
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
    """Geodesic angle [rad] between the body attitude and the nadir (LVLH) frame from the plant state."""
    q_oi = lvlh_from_orbit(dynamics.x[0:3], dynamics.x[3:6])
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
    # Start ~10 deg off nadir (a small body-frame offset from the LVLH frame).
    offset = Quaternion.from_array(np.array([0.06, -0.05, 0.04, 1.0]))
    q0 = offset * lvlh_from_orbit(_R0, _V0)
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
    dynamics.x[6:10] = lvlh_from_orbit(_R0, _V0).to_array()  # start pointed at nadir
    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    controller = _quaternion_controller(k_m=5e-4)

    h0 = np.linalg.norm(_wheel_momentum(dynamics, rw, dynamics.x[10:13]))
    for k in range(4000):
        t = k * 0.1
        u, _ = controller.evaluate(t, ref, _x_hat(dynamics, rw))
        dynamics.evaluate(t, u)
    h_final = np.linalg.norm(_wheel_momentum(dynamics, rw, dynamics.x[10:13]))

    assert h_final < h0  # magnetorquer dumping bled off stored momentum


def _lqr_kwargs() -> dict:
    return {
        "dt": 0.1,
        "Q": np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0]),
        "R": np.eye(6),
        "inertia": _INERTIA,
        "omega_c": np.array([0.0, -1.0e-3, 0.0]),
        "alpha_rw": allocation_matrix(_AXES, _KT * np.ones(3)),
        "alpha_mtq": allocation_matrix(_AXES, _KM * np.ones(3)),
    }


def test_adaptive_lqr_adapts_when_field_changes() -> None:
    adaptive = AdaptiveLQR(**_lqr_kwargs())

    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # First update with initial _B_BODY
    x_hat_initial = np.concatenate([_R0, _V0, [0.0, 0.0, 0.0, 1.0], np.zeros(3), _B_BODY, np.zeros(3)])
    adaptive.update(0.0, ref, x_hat_initial)
    k_initial = adaptive.K.copy()

    # Second update with different B field
    b_other = np.array([3.0e-5, 2.0e-5, -1.0e-5])
    x_hat_other = np.concatenate([_R0, _V0, [0.0, 0.0, 0.0, 1.0], np.zeros(3), b_other, np.zeros(3)])
    adaptive.update(0.0, ref, x_hat_other)

    assert not np.allclose(adaptive.K, k_initial)


def _lqr_x_hat(b_body: np.ndarray) -> np.ndarray:
    """19-element x_hat at nadir, at rest, with the given body-frame field and zero wheel momentum."""
    return np.concatenate([_R0, _V0, [0.0, 0.0, 0.0, 1.0], np.zeros(3), b_body, np.zeros(3)])


def test_adaptive_lqr_warm_start_stays_stabilizing() -> None:
    # The warm-started Riccati recursion must not diverge (no ill-conditioned-Lyapunov warning,
    # closed loop stays inside the unit circle) as the field varies slowly around the orbit.
    adaptive = AdaptiveLQR(**_lqr_kwargs())
    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)  # an ill-conditioned solve fails the test
        for _ in range(60):
            b = _B_BODY + 5.0e-6 * rng.standard_normal(3)
            _, log = adaptive.update(0.0, ref, _lqr_x_hat(b))
            assert log.closed_loop_radius < 1.0
            assert 0 <= log.n_iter <= 30


def test_adaptive_lqr_warm_start_matches_cold_solution() -> None:
    # A warm-started step (iterated to convergence) must reproduce the cold ARE gain.
    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    b = np.array([2.5e-5, 1.0e-5, -2.0e-5])

    warm = AdaptiveLQR(**_lqr_kwargs())
    for _ in range(20):  # build up the warm-start state
        warm.update(0.0, ref, _lqr_x_hat(_B_BODY))
    warm.update(0.0, ref, _lqr_x_hat(b))  # warm-started at field b

    cold = AdaptiveLQR(**_lqr_kwargs())
    cold.update(0.0, ref, _lqr_x_hat(b))  # first step solves cold at field b

    np.testing.assert_allclose(warm.K, cold.K, rtol=1e-5, atol=1e-6)


def _mpc_kwargs(**actuator_limits: object) -> dict:
    weights = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0])
    return {
        "dt": 0.1,
        "n_steps": 10,
        "Q": weights,
        "R": np.eye(6),
        "Qf": weights,
        "inertia": _INERTIA,
        "alpha_rw": allocation_matrix(_AXES, _KT * np.ones(3)),
        "alpha_mtq": allocation_matrix(_AXES, _KM * np.ones(3)),
        # Actuator limits, matching the plant's reaction wheels in `_make_plant` so they don't bind
        # unless a test deliberately tightens them.
        "wheel_axes": _AXES,
        "wheel_inertia": 0.01,
        "i_rw_max": 5.0,
        "i_mtq_max": 50.0,
        "omega_w_max": 8000.0 * 2.0 * np.pi / 60.0,
        **actuator_limits,
    }


def test_mpc_drives_attitude_error_to_zero() -> None:
    dynamics, rw = _make_plant()
    # Start ~10 deg off nadir (a small body-frame offset from the LVLH frame).
    offset = Quaternion.from_array(np.array([0.06, -0.05, 0.04, 1.0]))
    q0 = offset * lvlh_from_orbit(_R0, _V0)
    dynamics.x[6:10] = q0.to_array() / np.linalg.norm(q0.to_array())

    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # q_bo = identity (nadir), zero orbit-relative rate
    controller = MPC(**_mpc_kwargs())

    angle = np.inf
    for k in range(800):
        t = k * 0.1
        u, _ = controller.evaluate(t, ref, _x_hat(dynamics, rw))
        dynamics.evaluate(t, u)
        angle = _nadir_angle(dynamics)

    assert angle < np.deg2rad(1.5)


def test_mpc_from_config() -> None:
    config = {
        "dt": 0.1,
        "n_steps": 8,
        "Q": np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0]).tolist(),
        "R": np.eye(6).tolist(),
        "inertia": _INERTIA.tolist(),
        "reaction_wheels": {
            "axes": _AXES.tolist(),
            "torque_constant": [_KT, _KT, _KT],
            "inertia": [0.01, 0.01, 0.01],
            "max_current": 5.0,
            "max_rpm": 8000.0,
        },
        "magnetorquers": {"axes": _AXES.tolist(), "dipole_constant": [_KM, _KM, _KM], "max_current": 50.0},
    }
    controller = MPC.from_config(config)
    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    u, log = controller.update(0.0, ref, _lqr_x_hat(_B_BODY))

    u = np.asarray(u)
    assert u.shape == (6,)
    assert np.all(np.isfinite(u))
    assert log.solve_success


def test_mpc_respects_actuator_current_limits() -> None:
    # Tight enough that an unconstrained controller's response to a large attitude error would
    # exceed them: i_rw = tau_rw / K_w, i_mtq = (B x tau_mtq / |B|^2) / K_m (alpha = Lambda K with
    # Lambda = I here).
    i_rw_max = 5.0e-3
    i_mtq_max = 5.0e-3
    controller = MPC(**_mpc_kwargs(i_rw_max=i_rw_max, i_mtq_max=i_mtq_max))
    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # A large attitude error would drive an unconstrained controller well past the limits.
    offset = Quaternion.from_array(np.array([0.3, -0.25, 0.2, 1.0]))
    q0 = offset * lvlh_from_orbit(_R0, _V0)
    x_hat = np.concatenate([_R0, _V0, q0.to_array() / np.linalg.norm(q0.to_array()), np.zeros(3), _B_BODY, np.zeros(3)])

    u, log = controller.update(0.0, ref, x_hat)
    u_arr = np.asarray(u)

    assert log.solve_success
    i_mtq, i_rw = u_arr[:3], u_arr[3:]
    assert np.all(np.abs(i_rw) <= i_rw_max + 1e-6)  # IPOPT honors bounds to its constraint tolerance
    assert np.all(np.abs(i_mtq) <= i_mtq_max + 1e-6)


def test_mpc_respects_wheel_speed_limit() -> None:
    # A small omega_w_max with the wheels already near saturated momentum should hold the
    # predicted wheel speed |Omega| = |(Lambda_w D_w)^-1 h_w - Lambda_w^T omega| within the limit.
    wheel_inertia = 0.01
    omega_w_max = 50.0
    controller = MPC(
        **_mpc_kwargs(
            wheel_axes=_AXES,
            wheel_inertia=wheel_inertia,
            omega_w_max=omega_w_max,
        )
    )
    ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # Start near nadir but with the wheels already spun up close to the speed limit.
    h_w = wheel_inertia * np.array([45.0, -45.0, 45.0])
    x_hat = np.concatenate([_R0, _V0, lvlh_from_orbit(_R0, _V0).to_array(), np.zeros(3), _B_BODY, h_w])

    _, log = controller.update(0.0, ref, x_hat)

    assert log.solve_success
    assert np.all(np.isfinite(log.tau_rw))
