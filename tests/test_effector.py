import numpy as np

from rigid_body.effector import GravityGradient, RigidBodyState
from rigid_body.quaternion import Quaternion

MU_EARTH = 3.986e14


def _state(r: np.ndarray, q: np.ndarray) -> RigidBodyState:
    """Build a RigidBodyState with the given position and attitude (velocity/omega zero)."""
    return RigidBodyState(
        r_eci=r, v_eci=np.zeros(3), q_bi=Quaternion.from_array(q, scalar_first=False), omega_b_bi=np.zeros(3)
    )


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
    q = np.array([0.0, 0.0, 0.0, 1.0])  # identity: nadir = -x = body principal axis

    _, torque, _ = gg.calc_contributions(0.0, _state(r, q), np.zeros(0), np.zeros(0))
    assert np.allclose(torque, 0.0, atol=1e-12)


def test_known_torque_value() -> None:
    """Gravity-gradient torque matches the hand-computed analytic value (30 deg about z)."""
    inertia = np.diag([100.0, 200.0, 300.0])
    gg = _bound_gg(inertia)
    r = np.array([7.0e6, 0.0, 0.0])
    half = np.deg2rad(15.0)  # 30 deg rotation about body z
    q = np.array([0.0, 0.0, np.sin(half), np.cos(half)])

    _, torque, _ = gg.calc_contributions(0.0, _state(r, q), np.zeros(0), np.zeros(0))
    assert np.allclose(torque, np.array([0.0, 0.0, -1.5096e-4]), rtol=1e-3, atol=1e-9)


def test_torque_scales_as_inverse_r_cubed() -> None:
    """Doubling the orbital radius reduces the torque magnitude by a factor of 8."""
    inertia = np.diag([100.0, 200.0, 300.0])
    gg = _bound_gg(inertia)
    half = np.deg2rad(15.0)
    q = np.array([0.0, 0.0, np.sin(half), np.cos(half)])

    _, torque_near, _ = gg.calc_contributions(0.0, _state(np.array([7.0e6, 0.0, 0.0]), q), np.zeros(0), np.zeros(0))
    _, torque_far, _ = gg.calc_contributions(0.0, _state(np.array([1.4e7, 0.0, 0.0]), q), np.zeros(0), np.zeros(0))

    assert np.isclose(np.linalg.norm(torque_near) / np.linalg.norm(torque_far), 8.0, rtol=1e-9)
