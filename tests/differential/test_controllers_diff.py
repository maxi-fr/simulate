"""Differential tests for the controller helper functions.

* ``to_current_commands`` is mathematically identical across the port; only the signature changed
  from actuator-object lists (``rw.K_t``, ``rw.axis``) to pre-built allocation matrices. We feed both
  forms the same axes/constants and compare the current vector.

* ``update_lqr_warm_start`` (old) and ``_dlqr_warm_start`` (new) are DIFFERENT algorithms: the old
  does a single Newton-Kleinman step (one discrete-Lyapunov solve); the new runs up to 50 Riccati
  value-iteration sweeps. They only coincide at the DARE fixed point -- which is exactly the property
  we assert, while also documenting that, away from the fixed point, their one-call output diverges.
"""

import types
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from diffhelpers import rand_unit_vec

from rigid_body.controller import _dlqr_warm_start, allocation_matrix, to_current_commands
from rigid_body.linearization import reduced_model


def _actuator(k_t: float, axis: np.ndarray) -> Any:
    """Duck-typed stand-in for the old ReactionWheel/Magnetorquer (only ``.K_t`` and ``.axis`` used)."""
    return types.SimpleNamespace(K_t=k_t, axis=axis)


def test_to_current_commands_matches(rng: np.random.Generator) -> None:
    """Torque-to-current allocation is identical despite the signature change.

    Old took lists of actuator objects (reading ``.K_t``/``.axis``); new takes pre-built allocation
    matrices. Feeding both the same 3 reaction-wheel and 3 magnetorquer axes/constants, field and
    desired torques, the resulting ``[i_mtq, i_rw]`` current vectors must agree.
    """
    old_mod = pytest.importorskip("flight_software.controllers")

    for _ in range(20):
        rw_axes = np.array([rand_unit_vec(rng) for _ in range(3)])
        mtq_axes = np.array([rand_unit_vec(rng) for _ in range(3)])
        rw_kt = rng.uniform(0.5, 2.0, size=3)
        mtq_kt = rng.uniform(0.5, 2.0, size=3)

        tau_mtq = rng.uniform(-1e-3, 1e-3, size=3)
        tau_rw = rng.uniform(-1e-3, 1e-3, size=3)
        b_body = rng.uniform(-3e-5, 3e-5, size=3)

        old_u = old_mod.to_current_commands(
            np.concatenate([tau_mtq, tau_rw]),
            b_body,
            [_actuator(k, ax) for k, ax in zip(mtq_kt, mtq_axes, strict=True)],
            [_actuator(k, ax) for k, ax in zip(rw_kt, rw_axes, strict=True)],
        )
        new_u = to_current_commands(
            tau_rw,
            tau_mtq,
            b_body,
            allocation_matrix(rw_axes, rw_kt),
            allocation_matrix(mtq_axes, mtq_kt),
        )
        np.testing.assert_allclose(new_u, old_u, rtol=1e-10, atol=1e-12)


def test_warm_start_agree_at_dare_fixed_point() -> None:
    """Both warm-start solvers are consistent at the DARE solution (their shared fixed point).

    Despite using different algorithms (old = one Newton-Kleinman step; new = value iteration), both
    must leave the exact Riccati solution ``P*`` essentially unchanged and return the same gain
    ``K*``. We seed both with ``P*`` (from ``solve_discrete_are`` on a realistic reduced model) and
    check ``P`` and ``K`` against ``P*``/``K*`` and against each other.
    """
    old_mod = pytest.importorskip("flight_software.controllers")

    inertia = np.diag([8.0, 9.0, 10.0])
    omega_c = np.array([0.0, -1.0e-3, 0.0])
    b_field = np.array([1.5e-5, -2.0e-5, 3.0e-5])
    a_mat, b_mat = reduced_model(b_field, 0.5, omega_c, inertia)
    q = np.eye(6)
    r = np.eye(6) * 1e2

    p_star = scipy.linalg.solve_discrete_are(a_mat, b_mat, q, r)
    k_star = np.linalg.solve(r + b_mat.T @ p_star @ b_mat, b_mat.T @ p_star @ a_mat)

    old_k, old_p = old_mod.update_lqr_warm_start(a_mat, b_mat, q, r, p_star)
    new_k, new_p = _dlqr_warm_start(a_mat, b_mat, q, r, p_star)

    # Both warm-start methods leave the fixed point essentially unchanged.
    np.testing.assert_allclose(old_p, p_star, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(new_p, p_star, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(old_k, k_star, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(new_k, k_star, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(new_k, old_k, rtol=1e-6, atol=1e-6)


def test_warm_start_diverges_away_from_fixed_point() -> None:
    """DOCUMENTED DIFFERENCE: one warm-start call gives different gains off the fixed point.

    Old = single Newton-Kleinman step; new = up to 50 value-iteration sweeps. From a perturbed ``P``
    their single-call outputs disagree (they only reconverge to the shared DARE solution after enough
    iterations).
    """
    old_mod = pytest.importorskip("flight_software.controllers")

    inertia = np.diag([8.0, 9.0, 10.0])
    omega_c = np.array([0.0, -1.0e-3, 0.0])
    b_field = np.array([1.5e-5, -2.0e-5, 3.0e-5])
    a_mat, b_mat = reduced_model(b_field, 0.5, omega_c, inertia)
    q = np.eye(6)
    r = np.eye(6) * 1e2

    p0 = np.eye(6)  # far from the DARE solution
    old_k, _ = old_mod.update_lqr_warm_start(a_mat, b_mat, q, r, p0)
    new_k, _ = _dlqr_warm_start(a_mat, b_mat, q, r, p0)
    assert not np.allclose(old_k, new_k, rtol=1e-3, atol=1e-3)
