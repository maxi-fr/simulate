import datetime
import importlib

import numpy as np

from spacecraft.effector import (
    EarthGravity,
    ReactionWheelArray,
    RigidBodyState,
    Wrench,
)
from spacecraft.frames import eci_attitude_from_lvlh, lvlh_from_orbit
from spacecraft.orbit_dynamics import SGP4
from spacecraft.quaternion import Quaternion, QuaternionRK4
from spacecraft.rigid_body import RigidBodyDynamics

_qc_module = importlib.import_module("examples.02_quadrocopter.quadrocopter")
AerodynamicDragQuad = _qc_module.AerodynamicDragQuad
FlatGravity = _qc_module.FlatGravity
Quadrocopter = _qc_module.Quadrocopter


def _run(dynamics: RigidBodyDynamics, cmd: np.ndarray, n_steps: int) -> None:
    """Step the dynamics ``n_steps`` times with a constant command ``cmd``."""
    for k in range(n_steps):
        dynamics.evaluate(k * dynamics.dt, cmd)


def test_constant_inertial_force_translates() -> None:
    """A constant inertial-frame force gives constant acceleration."""
    dt = 0.001
    mass = 2.0
    fx = 4.0
    dynamics = RigidBodyDynamics(dt=dt, mass=mass, inertia=[1.0, 1.0, 1.0], effectors=[Wrench()])

    n = 1000
    cmd = np.array([fx, 0.0, 0.0, 0.0, 0.0, 0.0])
    _run(dynamics, cmd, n)

    t = n * dt
    accel = fx / mass
    assert np.isclose(dynamics.x[0], 0.5 * accel * t**2, rtol=1e-4)
    assert np.isclose(dynamics.x[3], accel * t, rtol=1e-4)
    # No rotation should be induced.
    assert np.allclose(dynamics.x[6:10], np.array([0.0, 0.0, 0.0, 1.0]))


def test_torque_free_symmetric_spin_preserves_omega_and_unit_quaternion() -> None:
    """With a symmetric inertia and no torque, omega is constant and |q| stays 1."""
    dt = 0.001
    dynamics = RigidBodyDynamics(dt=dt, mass=1.0, inertia=[1.0, 1.0, 1.0], effectors=[])
    omega0 = np.array([0.3, -0.2, 0.5])
    dynamics.x[10:13] = omega0

    _run(dynamics, np.zeros(0), 2000)

    assert np.allclose(dynamics.x[10:13], omega0)
    assert np.isclose(np.linalg.norm(dynamics.x[6:10]), 1.0)


def test_torque_free_asymmetric_conserves_angular_momentum() -> None:
    """A torque-free asymmetric body conserves the angular-momentum magnitude |J omega|."""
    dt = 0.001
    inertia = np.diag([1.0, 2.0, 3.0])
    dynamics = RigidBodyDynamics(dt=dt, mass=1.0, inertia=inertia, effectors=[])
    omega0 = np.array([1.0, 1.0, 1.0])
    dynamics.x[10:13] = omega0

    h0 = np.linalg.norm(inertia @ omega0)
    _run(dynamics, np.zeros(0), 3000)
    h_final = np.linalg.norm(inertia @ dynamics.x[10:13])

    assert np.isclose(h_final, h0, rtol=1e-6)
    assert np.isclose(np.linalg.norm(dynamics.x[6:10]), 1.0)


def test_from_config_round_trip() -> None:
    """from_config builds the effectors and a quaternion-aware integrator, and steps."""
    config = {
        "dt": 0.01,
        "mass": 3.0,
        "inertia": [1.0, 2.0, 3.0],
        "effectors": [
            {"class_path": "spacecraft.effector.Wrench"},
            {
                "class_path": "spacecraft.effector.ReactionWheelArray",
                "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "inertia": 0.05,
                "torque_constant": 0.08,
                "time_constant": 0.04,
                "max_current": 2.5,
            },
        ],
    }
    dynamics = RigidBodyDynamics.from_config(config)

    assert len(dynamics.effectors) == 2
    assert isinstance(dynamics.effectors[0], Wrench)
    assert isinstance(dynamics.effectors[1], ReactionWheelArray)
    assert isinstance(dynamics.integrator, QuaternionRK4)
    assert dynamics.mass == 3.0

    # Command layout is 6 (wrench) + 3 (wheels) = 9; stepping must not raise.
    dynamics.evaluate(0.0, np.zeros(9))
    assert dynamics.x.shape == (19,)


def test_from_config_initial_state_from_tle_and_lvlh_attitude() -> None:
    """An ``initial_state`` block seeds r/v from SGP4 and q/omega from the LVLH-relative attitude."""
    tle = (
        "1 25544U 98067A   24001.50000000  .00000000  00000-0  00000-0 2    07",
        "2 25544 097.6000 010.0000 0001000 000.0000 000.0000 15.25000000000009",
    )
    epoch = "2024-01-01T12:00:00"
    config = {
        "dt": 0.2,
        "mass": 2.0,
        "inertia": [1.0, 1.0, 1.0],
        "initial_state": {
            "epoch": epoch,
            "tle": list(tle),
            "attitude_lvlh": {"roll": 0.0, "pitch": 0.0, "yaw": -15.0},
            "angular_velocity_lvlh": [0.0, 0.0, 0.0],
        },
    }

    dynamics = RigidBodyDynamics.from_config(config)

    r0, v0 = SGP4.from_tle(*tle).propagate(datetime.datetime.fromisoformat(epoch))
    q_bi, omega0 = eci_attitude_from_lvlh(r0, v0, roll=0.0, pitch=0.0, yaw=-15.0, omega_bo=np.zeros(3))
    np.testing.assert_allclose(dynamics.x[0:3], r0)
    np.testing.assert_allclose(dynamics.x[3:6], v0)
    np.testing.assert_allclose(dynamics.x[6:10], q_bi.to_array())
    np.testing.assert_allclose(np.linalg.norm(dynamics.x[6:10]), 1.0)
    np.testing.assert_allclose(dynamics.x[10:13], omega0)

    # The seeded attitude is 15 deg off nadir (a pure yaw about the LVLH frame).
    q_bo = Quaternion.from_array(dynamics.x[6:10]) * lvlh_from_orbit(r0, v0).conjugate()
    nadir_angle = 2.0 * np.arctan2(np.linalg.norm(q_bo.vec), abs(q_bo.scalar))
    np.testing.assert_allclose(np.degrees(nadir_angle), 15.0, atol=1e-6)


def test_from_config_initial_state_direct() -> None:
    """An ``initial_state`` block can directly seed r, v, q, and omega."""
    config = {
        "dt": 0.01,
        "mass": 1.5,
        "inertia": [1.0, 1.0, 1.0],
        "initial_state": {
            "r": [1.0, 2.0, 3.0],
            "v": [4.0, 5.0, 6.0],
            "q": [0.0, 0.0, 0.0, 1.0],
            "omega": [7.0, 8.0, 9.0],
        },
    }
    dynamics = RigidBodyDynamics.from_config(config)
    np.testing.assert_allclose(dynamics.x[0:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(dynamics.x[3:6], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(dynamics.x[6:10], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(dynamics.x[10:13], [7.0, 8.0, 9.0])


def test_gravity_gradient_torque_acts_through_ode() -> None:
    """A gravity-gradient-only body released off-equilibrium develops angular velocity."""
    dt = 0.5
    inertia = np.diag([100.0, 200.0, 300.0])
    body = RigidBodyDynamics(
        dt=dt,
        mass=500.0,
        inertia=inertia,
        effectors=[EarthGravity(mu=3.986e14)],
    )
    # Orbital radius along inertial x, tilted attitude so the torque is nonzero.
    body.x[0:3] = np.array([7.0e6, 0.0, 0.0])
    half = np.pi / 8  # 22.5 deg about body z
    body.x[6:10] = np.array([0.0, 0.0, np.sin(half), np.cos(half)])

    for k in range(200):
        body.evaluate(k * dt, np.zeros(0))

    assert np.linalg.norm(body.x[10:13]) > 0.0
    assert np.isclose(np.linalg.norm(body.x[6:10]), 1.0)


def test_flat_gravity() -> None:
    dt = 0.01
    mass = 1.5
    g_acc = np.array([0.0, 0.0, -9.81])
    dynamics = RigidBodyDynamics(dt=dt, mass=mass, inertia=[1.0, 1.0, 1.0], effectors=[FlatGravity(g_acc)])

    # Run one step
    dynamics.evaluate(0.0, np.zeros(0))

    # Velocity should be g_acc * dt
    expected_v = g_acc * dt
    assert np.allclose(dynamics.x[3:6], expected_v)


def test_quadrocopter_forces_and_torques() -> None:
    dt = 0.01
    mass = 1.5
    quad = Quadrocopter(
        rotor_positions=[[0.2, 0.2, 0.0], [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [-0.2, 0.2, 0.0]],
        rotor_directions=[-1, -1, 1, 1],
        torque_to_thrust_ratio=0.015,
        thrust_axis=[0.0, 0.0, 1.0],
    )
    RigidBodyDynamics(dt=dt, mass=mass, inertia=[1.0, 1.0, 1.0], effectors=[quad])

    # 1. Symmetric thrust command (hover)
    cmd_symmetric = np.array([3.67875, 3.67875, 3.67875, 3.67875])
    # Total force in body = 4 * 3.67875 = 14.715 along z
    # Since q is identity, force in ECI is [0, 0, 14.715]
    state = RigidBodyState(
        r_eci=np.zeros(3), v_eci=np.zeros(3), q_bi=Quaternion.from_array([0.0, 0.0, 0.0, 1.0]), omega_b_bi=np.zeros(3)
    )
    f, tau, h = quad.calc_contributions(0.0, state, np.zeros(0), cmd_symmetric)
    assert np.allclose(f, [0.0, 0.0, 14.715])
    assert np.allclose(tau, np.zeros(3))
    assert np.allclose(h, np.zeros(3))

    # 2. Asymmetric thrust: roll/pitch torque
    # Add 1.0 N to Rotor 1 (0.2, 0.2, 0.0) -> CCW/CW direction -1
    cmd_asymmetric = np.array([4.67875, 3.67875, 3.67875, 3.67875])
    f, tau, h = quad.calc_contributions(0.0, state, np.zeros(0), cmd_asymmetric)
    # Rotor 1 is at [0.2, 0.2, 0.0]. Thrust force of rotor 1 is 4.67875 N along [0, 0, 1].
    # Total torque delta:
    # delta_thrust = 1.0 N
    # delta_tau_thrust = [0.2, 0.2, 0.0] x [0, 0, 1.0] = [0.2, -0.2, 0.0]
    # delta_tau_react = -(-1) * 0.015 * 1.0 * [0, 0, 1] = [0, 0, 0.015]
    # Total torque should be [0.2, -0.2, 0.015]
    assert np.allclose(tau, [0.2, -0.2, 0.015])


def test_aerodynamic_drag_quad() -> None:
    drag = AerodynamicDragQuad(c_d=0.2, c_rot=0.1)

    state = RigidBodyState(
        r_eci=np.zeros(3),
        v_eci=np.array([5.0, 0.0, 0.0]),
        q_bi=Quaternion.from_array([0.0, 0.0, 0.0, 1.0]),
        omega_b_bi=np.array([0.0, 0.0, 2.0]),
    )

    f, tau, h = drag.calc_contributions(0.0, state, np.zeros(0), np.zeros(0))
    assert np.allclose(f, [-1.0, 0.0, 0.0])
    assert np.allclose(tau, [0.0, 0.0, -0.2])
    assert np.allclose(h, np.zeros(3))


def test_quad_from_config() -> None:
    config = {
        "dt": 0.01,
        "mass": 1.5,
        "inertia": [0.015, 0.015, 0.03],
        "effectors": [
            {
                "class_path": "examples.02_quadrocopter.quadrocopter.Quadrocopter",
                "rotor_positions": [[0.2, 0.2, 0.0], [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [-0.2, 0.2, 0.0]],
                "rotor_directions": [-1, -1, 1, 1],
                "torque_to_thrust_ratio": 0.015,
                "thrust_axis": [0.0, 0.0, 1.0],
            },
            {
                "class_path": "examples.02_quadrocopter.quadrocopter.FlatGravity",
                "gravity_acceleration": [0.0, 0.0, -9.81],
            },
            {"class_path": "examples.02_quadrocopter.quadrocopter.AerodynamicDragQuad", "c_d": 0.1, "c_rot": 0.05},
        ],
    }
    dynamics = RigidBodyDynamics.from_config(config)
    assert len(dynamics.effectors) == 3

    assert isinstance(dynamics.effectors[0], Quadrocopter)
    assert isinstance(dynamics.effectors[1], FlatGravity)
    assert isinstance(dynamics.effectors[2], AerodynamicDragQuad)

    # Command length should be 4
    assert dynamics.n_inputs == 4
    dynamics.evaluate(0.0, np.array([3.6, 3.6, 3.6, 3.6]))
    assert dynamics.x.shape == (13,)
