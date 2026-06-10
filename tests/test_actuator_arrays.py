import numpy as np

from simulate.attitude import quat_to_rotation_matrix
from simulate.effector import BodyState, MagnetorquerArray, ReactionWheelArray
from simulate.rigid_body import RigidBodyDynamics


def test_reaction_wheel_array_momentum_conservation() -> None:
    """Spinning up wheels counter-rotates the body; total angular momentum is conserved."""
    dt = 0.005
    inertia = np.diag([1.5, 2.0, 2.5])

    # 3 orthogonal reaction wheels
    axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    wheel_inertia = 0.05
    torque_constant = 0.08
    time_constant = 0.04
    max_current = 2.5

    rw_array = ReactionWheelArray(
        axes=axes,
        inertia=wheel_inertia,
        torque_constant=torque_constant,
        time_constant=time_constant,
        max_current=max_current,
        initial_currents=[0.0, 0.0, 0.0],
        initial_omega=[1.0, -2.0, 3.0],
    )

    dynamics = RigidBodyDynamics(
        dt=dt,
        mass=10.0,
        inertia=inertia,
        effectors=[rw_array],
    )

    # State layout: [r(3) | v(3) | q(4) | omega(3) | currents(3) | omega_rel(3)]
    # Verify initial state has correct omega_rel
    assert np.allclose(dynamics.x[16:19], np.array([1.0, -2.0, 3.0]))

    # Step the dynamics with a non-zero current command
    cmd = np.array([1.5, -1.0, 2.0])

    # Calculate initial total momentum in inertial frame
    q_0 = dynamics.x[6:10]
    omega_0 = dynamics.x[10:13]
    omega_rel_0 = dynamics.x[16:19]

    axes_mat = rw_array.axes  # (3, 3) identity
    h_wheels_0 = wheel_inertia * (omega_rel_0 + axes_mat @ omega_0)
    h_total_0_body = inertia @ omega_0 + h_wheels_0
    h_total_0_inertial = quat_to_rotation_matrix(q_0) @ h_total_0_body

    # Step 100 times
    for k in range(100):
        dynamics.evaluate(k * dt, cmd)

    # Calculate final total momentum in inertial frame
    q_f = dynamics.x[6:10]
    omega_f = dynamics.x[10:13]
    omega_rel_f = dynamics.x[16:19]
    h_wheels_f = wheel_inertia * (omega_rel_f + axes_mat @ omega_f)
    h_total_f_body = inertia @ omega_f + h_wheels_f
    h_total_f_inertial = quat_to_rotation_matrix(q_f) @ h_total_f_body

    # Total angular momentum must be conserved (constant) in the inertial frame
    assert np.allclose(h_total_f_inertial, h_total_0_inertial, rtol=1e-6, atol=1e-9)


def test_reaction_wheel_array_current_saturation() -> None:
    """Commanding a current past saturation limit clamps current state to max_current."""
    dt = 0.01
    rw_array = ReactionWheelArray(
        axes=[[1.0, 0.0, 0.0]],
        inertia=0.01,
        torque_constant=0.02,
        time_constant=0.05,
        max_current=0.5,
    )

    dynamics = RigidBodyDynamics(
        dt=dt,
        mass=1.0,
        inertia=[1.0, 1.0, 1.0],
        effectors=[rw_array],
    )

    # Command a huge current limit (2.0 A) which exceeds 0.5 A saturation limit
    cmd = np.array([2.0])
    for k in range(100):
        dynamics.evaluate(k * dt, cmd)

    # Verify state of current is bounded by max_current
    current = dynamics.x[13]
    assert np.isclose(current, 0.5, atol=1e-4)


def test_magnetorquer_array_torque_generation() -> None:
    """Magnetorquer torque is generated exactly as m x B."""
    axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    dipole_constant = 2.0
    time_constant = 0.08
    max_current = 1.5
    b_field = np.array([1e-5, -2e-5, 3e-5])

    mtq_array = MagnetorquerArray(
        axes=axes,
        dipole_constant=dipole_constant,
        time_constant=time_constant,
        max_current=max_current,
        b_field_model=b_field,
        initial_currents=[1.0, -1.0, 0.5],
    )

    # Evaluate directly
    state = BodyState(r=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.zeros(3))
    x_eff = np.array([1.0, -1.0, 0.5])
    cmd = np.array([0.0, 0.0, 0.0])

    _, torque, momentum = mtq_array.calc_contributions(0.0, state, x_eff, cmd)

    # Expected dipole: K_m * i = 2.0 * [1.0, -1.0, 0.5] = [2.0, -2.0, 1.0]
    expected_m = dipole_constant * x_eff
    expected_torque = np.cross(expected_m, b_field)

    assert np.allclose(torque, expected_torque)
    assert np.allclose(momentum, 0.0)


def test_from_config_loading() -> None:
    """Both effector arrays can be loaded successfully from a configuration dict."""
    config = {
        "dt": 0.01,
        "mass": 2.0,
        "inertia": [1.0, 1.0, 1.0],
        "effectors": [
            {
                "class_path": "simulate.effector.ReactionWheelArray",
                "axes": [[1, 0, 0], [0, 1, 0]],
                "inertia": 0.01,
                "torque_constant": 0.05,
                "time_constant": 0.1,
                "max_current": 1.0,
                "initial_omega": [10.0, -10.0],
            },
            {
                "class_path": "simulate.effector.MagnetorquerArray",
                "axes": [[0, 0, 1]],
                "dipole_constant": 1.5,
                "time_constant": 0.05,
                "max_current": 2.0,
                "b_field_model": [0, 0, 5e-5],
            },
        ],
    }

    dynamics = RigidBodyDynamics.from_config(config)
    assert len(dynamics.effectors) == 2
    assert isinstance(dynamics.effectors[0], ReactionWheelArray)
    assert isinstance(dynamics.effectors[1], MagnetorquerArray)

    # State length: 13 (base) + 4 (RW array: 2 currents, 2 omega) + 1 (mtq current) = 18
    assert dynamics.x.shape == (18,)


def test_reaction_wheel_speed_saturation() -> None:
    """Motor torque saturates to zero if accelerating beyond max_rpm limit."""
    rw_array = ReactionWheelArray(
        axes=[[1.0, 0.0, 0.0]],
        inertia=0.01,
        torque_constant=0.05,
        time_constant=0.1,
        max_current=1.0,
        max_rpm=6000.0,  # max_omega = 200 * pi ≈ 628.3185
    )

    # State has a current of 1.0A (positive torque) and speed above limit: 630.0 rad/s
    x_eff = np.array([1.0, 630.0])
    state = BodyState(r=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.zeros(3))

    # calc_contributions should return zero torque because the speed is above limit and torque is positive
    _, torque, _ = rw_array.calc_contributions(0.0, state, x_eff, np.array([1.0]))
    assert np.allclose(torque, 0.0)

    # If the current is negative (-1.0A), torque should NOT be saturated to 0 because it opposes the speed
    x_eff_neg = np.array([-1.0, 630.0])
    _, torque_neg, _ = rw_array.calc_contributions(0.0, state, x_eff_neg, np.array([-1.0]))
    assert np.allclose(torque_neg, np.array([0.05, 0.0, 0.0]))

    # Check dynamics (domega_dt) behavior:
    # If speed is above limit and torque would be positive (accelerating), domega_dt should have zero motor torque term
    didt, domega_dt = np.split(rw_array.dynamics(0.0, state, x_eff, np.array([1.0]), np.zeros(3)), 2)
    assert np.allclose(domega_dt, 0.0)

    # If speed is below -limit (-630.0) and torque is negative (accelerating negative), domega_dt should have zero motor torque term
    x_eff_neg_speed = np.array([-1.0, -630.0])
    didt, domega_dt = np.split(rw_array.dynamics(0.0, state, x_eff_neg_speed, np.array([-1.0]), np.zeros(3)), 2)
    assert np.allclose(domega_dt, 0.0)


def test_reaction_wheel_didt_limits() -> None:
    """Anti-windup: didt is zeroed when current is at max limit and command pushes it further."""
    rw_array = ReactionWheelArray(
        axes=[[1.0, 0.0, 0.0]],
        inertia=0.01,
        torque_constant=0.05,
        time_constant=0.1,
        max_current=1.0,
    )

    state = BodyState(r=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.zeros(3))

    # Current is at max limit: 1.0A, and command is 2.0A (pushes further positive)
    x_eff = np.array([1.0, 0.0])
    didt, _ = np.split(rw_array.dynamics(0.0, state, x_eff, np.array([2.0]), np.zeros(3)), 2)
    assert np.allclose(didt, 0.0)

    # If command is 0.5A (decreases current), didt should not be zeroed
    didt, _ = np.split(rw_array.dynamics(0.0, state, x_eff, np.array([0.5]), np.zeros(3)), 2)
    assert np.isclose(didt[0], -5.0)

    # Current is at negative limit: -1.0A, and command is -2.0A (pushes further negative)
    x_eff_neg = np.array([-1.0, 0.0])
    didt, _ = np.split(rw_array.dynamics(0.0, state, x_eff_neg, np.array([-2.0]), np.zeros(3)), 2)
    assert np.allclose(didt, 0.0)


def test_magnetorquer_array_didt_limits() -> None:
    """Anti-windup: didt is zeroed when MTQ current is at max limit and command pushes it further."""
    mtq_array = MagnetorquerArray(
        axes=[[1.0, 0.0, 0.0]],
        dipole_constant=1.0,
        time_constant=0.05,
        max_current=1.5,
    )

    state = BodyState(r=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.zeros(3))

    # Current is at max limit: 1.5A, and command is 2.5A (pushes further positive)
    x_eff = np.array([1.5])
    didt = mtq_array.dynamics(0.0, state, x_eff, np.array([2.5]), np.zeros(3))
    assert np.allclose(didt, 0.0)

    # If command is 0.5A (decreases current), didt should not be zeroed
    # expected didt = (0.5 - 1.5) / 0.05 = -20.0
    didt = mtq_array.dynamics(0.0, state, x_eff, np.array([0.5]), np.zeros(3))
    assert np.isclose(didt[0], -20.0)

    # Current is at negative limit: -1.5A, and command is -2.5A (pushes further negative)
    x_eff_neg = np.array([-1.5])
    didt = mtq_array.dynamics(0.0, state, x_eff_neg, np.array([-2.5]), np.zeros(3))
    assert np.allclose(didt, 0.0)
