import datetime

import numpy as np

from rigid_body.effector import EarthGravity, ReactionWheelArray, Wrench
from rigid_body.frames import eci_attitude_from_orc, orc_from_orbit
from rigid_body.orbit_dynamics import SGP4
from rigid_body.quaternion import Quaternion, QuaternionRK4
from rigid_body.rigid_body import RigidBodyDynamics


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


def test_reaction_wheel_conserves_total_angular_momentum() -> None:
    """Spinning up wheels counter-rotates the body; total H = J omega + h stays 0."""
    dt = 0.001
    inertia = np.diag([1.0, 1.0, 1.0])
    rw_array = ReactionWheelArray(
        axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        inertia=0.05,
        torque_constant=0.08,
        time_constant=0.04,
        max_current=2.5,
    )
    dynamics = RigidBodyDynamics(dt=dt, mass=1.0, inertia=inertia, effectors=[rw_array])

    cmd = np.array([1.5, -1.0, 2.0])
    _run(dynamics, cmd, 1000)

    omega = dynamics.x[10:13]
    dynamics.x[13:16]
    omega_rel = dynamics.x[16:19]
    h_w = rw_array.inertia * (omega_rel + rw_array.axes @ omega)
    momentum = rw_array.axes.T @ h_w
    total_h = inertia @ omega + momentum

    assert np.allclose(total_h, 0.0, atol=1e-9)


def test_from_config_round_trip() -> None:
    """from_config builds the effectors and a quaternion-aware integrator, and steps."""
    config = {
        "dt": 0.01,
        "mass": 3.0,
        "inertia": [1.0, 2.0, 3.0],
        "effectors": [
            {"class_path": "rigid_body.effector.Wrench"},
            {
                "class_path": "rigid_body.effector.ReactionWheelArray",
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


def test_from_config_initial_state_from_tle_and_orc_attitude() -> None:
    """An ``initial_state`` block seeds r/v from SGP4 and q/omega from the ORC-relative attitude."""
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
            "attitude_orc": {"roll": 0.0, "pitch": 0.0, "yaw": -15.0},
            "angular_velocity_orc": [0.0, 0.0, 0.0],
        },
    }

    dynamics = RigidBodyDynamics.from_config(config)

    r0, v0 = SGP4.from_tle(*tle).propagate(datetime.datetime.fromisoformat(epoch))
    q_bi, omega0 = eci_attitude_from_orc(r0, v0, roll=0.0, pitch=0.0, yaw=-15.0, omega_bo=np.zeros(3))
    np.testing.assert_allclose(dynamics.x[0:3], r0)
    np.testing.assert_allclose(dynamics.x[3:6], v0)
    np.testing.assert_allclose(dynamics.x[6:10], q_bi.to_array())
    np.testing.assert_allclose(np.linalg.norm(dynamics.x[6:10]), 1.0)
    np.testing.assert_allclose(dynamics.x[10:13], omega0)

    # The seeded attitude is 15 deg off nadir (a pure yaw about the ORC frame).
    q_bo = Quaternion.from_array(dynamics.x[6:10]) * orc_from_orbit(r0, v0).conjugate()
    nadir_angle = 2.0 * np.arctan2(np.linalg.norm(q_bo.vec), abs(q_bo.scalar))
    np.testing.assert_allclose(np.degrees(nadir_angle), 15.0, atol=1e-6)


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
