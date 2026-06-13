"""Differential tests for the CasADi symbolic kernels ported to plain NumPy.

The quaternion primitives in the old ``flight_software.controller_models`` (CasADi ``SX``) and the new
:class:`~rigid_body.quaternion.Quaternion` (NumPy) are independent implementations of the same JPL
algebra; we evaluate the symbolic versions numerically and compare.

The reduced LQR *model*, by contrast, was deliberately re-derived rather than ported one-to-one: the
new ``rigid_body.linearization`` drops the wheel-momentum state and uses a different magnetorquer
torque model. We document this by checking that the reduced state dimension changed (old keeps
``h_w`` -> 9x9; new drops it -> 6x6), so a value-level comparison is intentionally not attempted.
"""

import numpy as np
import pytest
from diffhelpers import rand_inertia, rand_quat_array, rand_unit_vec

from rigid_body.controller_models import build_reduced_system_dynamics
from rigid_body.quaternion import Quaternion


def _val(expr: object) -> np.ndarray:
    """Evaluate a CasADi DM/SX numeric result into a flat NumPy array."""
    import casadi as ca

    return np.array(ca.DM(expr)).squeeze()


def test_quaternion_conjugate_matches(rng: np.random.Generator) -> None:
    """The CasADi symbolic ``quaternion_conjugate`` evaluates to the numpy ``Quaternion.conjugate``."""
    ca = pytest.importorskip("casadi")
    cm = pytest.importorskip("flight_software.controller_models")
    for _ in range(20):
        q = rand_quat_array(rng)
        expected = Quaternion.from_array(q).conjugate().to_array()
        np.testing.assert_allclose(_val(cm.quaternion_conjugate(ca.DM(q))), expected, rtol=1e-12, atol=1e-14)


def test_quaternion_product_matches(rng: np.random.Generator) -> None:
    """Symbolic JPL ``quaternion_product`` matches the numpy ``Quaternion.__mul__`` on random pairs."""
    ca = pytest.importorskip("casadi")
    cm = pytest.importorskip("flight_software.controller_models")
    for _ in range(20):
        q1 = rand_quat_array(rng)
        q2 = rand_quat_array(rng)
        expected = (Quaternion.from_array(q1) * Quaternion.from_array(q2)).to_array()
        np.testing.assert_allclose(_val(cm.quaternion_product(ca.DM(q1), ca.DM(q2))), expected, rtol=1e-12, atol=1e-14)


def test_quaternion_rotation_matches(rng: np.random.Generator) -> None:
    """Symbolic ``quaternion_rotation`` (rotate a vector) matches the numpy ``Quaternion.apply``."""
    ca = pytest.importorskip("casadi")
    cm = pytest.importorskip("flight_software.controller_models")
    for _ in range(20):
        q = rand_quat_array(rng)
        v = rng.standard_normal(3)
        expected = Quaternion.from_array(q).apply(v)
        np.testing.assert_allclose(_val(cm.quaternion_rotation(ca.DM(q), ca.DM(v))), expected, rtol=1e-12, atol=1e-13)


def test_attitude_jacobian_matches(rng: np.random.Generator) -> None:
    """The symbolic ``attitude_jacobian`` (the Xi matrix) matches the numpy ``Quaternion.xi``."""
    ca = pytest.importorskip("casadi")
    cm = pytest.importorskip("flight_software.controller_models")
    for _ in range(20):
        q = rand_quat_array(rng)
        np.testing.assert_allclose(_val(cm.attitude_jacobian(ca.DM(q))), Quaternion.from_array(q).xi, rtol=1e-12)


def test_symbolic_kinematics_matches(rng: np.random.Generator) -> None:
    """Symbolic quaternion ``kinematics`` (``0.5 Xi(q) w``) matches the numpy ``Quaternion.kinematics``."""
    ca = pytest.importorskip("casadi")
    cm = pytest.importorskip("flight_software.controller_models")
    for _ in range(20):
        q = rand_quat_array(rng)
        w = rng.standard_normal(3)
        expected = Quaternion.from_array(q).kinematics(w)
        np.testing.assert_allclose(_val(cm.kinematics(ca.DM(q), ca.DM(w))), expected, rtol=1e-12, atol=1e-13)


def test_reduced_model_matches(rng: np.random.Generator) -> None:
    """The new CasADi-based reduced_model matches the old build_reduced_system_dynamics."""
    pytest.importorskip("casadi")
    cm = pytest.importorskip("flight_software.controller_models")

    inertia = rand_inertia(rng)
    dt = 0.5
    omega_ref = np.array([0.0, -1.0e-3, 0.0])
    q_ref = rand_quat_array(rng)
    b_eci = rng.standard_normal(3) * 3e-5

    _, a_func, b_func = cm.build_reduced_system_dynamics(dt, inertia)
    x_star = np.concatenate([q_ref, omega_ref, np.zeros(3)])
    u_star = np.zeros(6)
    expected_a = np.array(a_func(x_star, u_star, b_eci))
    expected_b = np.array(b_func(x_star, u_star, b_eci))

    _, new_a_func, new_b_func = build_reduced_system_dynamics(dt, inertia)
    new_a = np.array(new_a_func(x_star, u_star, b_eci))
    new_b = np.array(new_b_func(x_star, u_star, b_eci))

    assert new_a.shape == (9, 9)
    assert new_b.shape == (9, 6)
    # Both repositories now use RK2 integration under the hood, matching exactly.
    np.testing.assert_allclose(new_a, expected_a, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(new_b, expected_b, rtol=1e-10, atol=1e-12)
