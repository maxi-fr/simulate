import numpy as np

from simulate.effector import BodyState, GravityGradient

MU_EARTH = 3.986e14


def _state(r: np.ndarray, q: np.ndarray) -> BodyState:
    """Build a BodyState with the given position and attitude (velocity/omega zero)."""
    return BodyState(r=r, v=np.zeros(3), q=q, omega=np.zeros(3))


def _bound_gg(inertia: np.ndarray) -> GravityGradient:
    """A GravityGradient with its inertia bound, as RigidBodyDynamics would."""
    gg = GravityGradient(mu=MU_EARTH)
    gg.bind(mass=500.0, inertia=inertia)
    return gg


def test_zero_torque_at_principal_axis_equilibrium() -> None:
    """With a principal axis aligned with nadir, the gravity-gradient torque vanishes."""
    inertia = np.diag([100.0, 200.0, 300.0])
    gg = _bound_gg(inertia)
    r = np.array([7.0e6, 0.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])  # identity: nadir = -x = body principal axis

    out = gg.evaluate(0.0, _state(r, q), np.zeros(0), np.zeros(0))
    assert np.allclose(out.torque, 0.0, atol=1e-12)


def test_known_torque_value() -> None:
    """Gravity-gradient torque matches the hand-computed analytic value (30 deg about z)."""
    inertia = np.diag([100.0, 200.0, 300.0])
    gg = _bound_gg(inertia)
    r = np.array([7.0e6, 0.0, 0.0])
    half = np.deg2rad(15.0)  # 30 deg rotation about body z
    q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    out = gg.evaluate(0.0, _state(r, q), np.zeros(0), np.zeros(0))
    assert np.allclose(out.torque, np.array([0.0, 0.0, -1.5096e-4]), rtol=1e-3, atol=1e-9)


def test_torque_scales_as_inverse_r_cubed() -> None:
    """Doubling the orbital radius reduces the torque magnitude by a factor of 8."""
    inertia = np.diag([100.0, 200.0, 300.0])
    gg = _bound_gg(inertia)
    half = np.deg2rad(15.0)
    q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    near = gg.evaluate(0.0, _state(np.array([7.0e6, 0.0, 0.0]), q), np.zeros(0), np.zeros(0))
    far = gg.evaluate(0.0, _state(np.array([1.4e7, 0.0, 0.0]), q), np.zeros(0), np.zeros(0))

    assert np.isclose(np.linalg.norm(near.torque) / np.linalg.norm(far.torque), 8.0, rtol=1e-9)
