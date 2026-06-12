"""Differential tests for the rotational dynamics that became a class method during the port.

The legacy free function ``simulation.dynamics.attitude_dynamics`` was folded into
``RigidBodyDynamics.dynamics`` (the ``omega_dot`` block). We drive the new class with a
:class:`~rigid_body.effector.Wrench` (supplying the control+disturbance torque) and a tiny constant
momentum effector (supplying the reaction-wheel momentum ``h``) and compare the resolved angular
acceleration against the old function.

Also pins the one genuine behavioral difference found inside the otherwise verbatim-copied
``Quaternion``: ``kinematics(omega, scalar_first=True)`` lost its ``0.5`` factor in the old code.
"""

from typing import Any

import numpy as np
import pytest
from diffhelpers import rand_inertia, rand_quat_array

from rigid_body.effector import Effector, RigidBodyState, Wrench
from rigid_body.quaternion import Quaternion
from rigid_body.rigid_body import ANGULAR_VELOCITY, QUATERNION, RigidBodyDynamics


class _ConstMomentum(Effector):
    """Stateless effector carrying a fixed body-frame internal angular momentum ``h``."""

    n_inputs = 0
    n_states = 0

    def __init__(self, h: np.ndarray) -> None:
        """Store the constant body-frame angular momentum ``h`` this effector injects."""
        self.h = np.asarray(h, dtype=float)

    def calc_contributions(
        self,
        t: float,
        state: RigidBodyState,
        x_eff: np.ndarray,
        cmd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Contribute zero force/torque and the fixed momentum ``h`` (so it appears in ``J@w + h``)."""
        del t, state, x_eff, cmd
        return np.zeros(3), np.zeros(3), self.h

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "_ConstMomentum":
        """Not used in tests; only present to satisfy the abstract :class:`Effector` interface."""
        del config
        raise NotImplementedError


def _new_omega_dot(
    inertia: np.ndarray,
    omega: np.ndarray,
    torque: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Resolve ``omega_dot`` from the new ``RigidBodyDynamics`` for matched inputs."""
    body = RigidBodyDynamics(dt=0.1, mass=10.0, inertia=inertia, effectors=[Wrench(), _ConstMomentum(h)])
    x = np.zeros(13)
    x[QUATERNION] = np.array([0.0, 0.0, 0.0, 1.0])
    x[ANGULAR_VELOCITY] = omega
    u = np.concatenate([np.zeros(3), torque])  # Wrench command: [force, torque]
    return body.dynamics(0.0, x, u)[ANGULAR_VELOCITY]


def test_attitude_dynamics_matches_rigidbody_euler_term(rng: np.random.Generator) -> None:
    """Euler's equation matches with non-zero wheel momentum across random inertias/rates.

    For 20 random (inertia, omega, control torque, disturbance torque, wheel momentum) tuples,
    the old free function ``attitude_dynamics(omega, J, ctrl, dist, h)`` and the new class's
    resolved ``omega_dot`` (control+disturbance fed via a ``Wrench``, ``h`` via ``_ConstMomentum``)
    must agree -- confirming the ``J^-1 (tau - omega x (J omega + h))`` term survived the refactor.
    """
    old = pytest.importorskip("simulation.dynamics")

    for _ in range(20):
        inertia = rand_inertia(rng)
        omega = rng.uniform(-0.1, 0.1, size=3)
        ctrl = rng.uniform(-1e-3, 1e-3, size=3)
        dist = rng.uniform(-1e-4, 1e-4, size=3)
        h = rng.uniform(-0.05, 0.05, size=3)

        old_omega_dot = old.attitude_dynamics(omega, inertia, ctrl, dist, h)
        new_omega_dot = _new_omega_dot(inertia, omega, ctrl + dist, h)

        np.testing.assert_allclose(new_omega_dot, old_omega_dot, rtol=1e-10, atol=1e-12)


def test_attitude_dynamics_zero_momentum_default(rng: np.random.Generator) -> None:
    """The old ``h_int=None`` default (zero momentum) also matches."""
    old = pytest.importorskip("simulation.dynamics")
    inertia = rand_inertia(rng)
    omega = rng.uniform(-0.1, 0.1, size=3)
    torque = rng.uniform(-1e-3, 1e-3, size=3)

    old_omega_dot = old.attitude_dynamics(omega, inertia, torque, np.zeros(3))
    new_omega_dot = _new_omega_dot(inertia, omega, torque, np.zeros(3))
    np.testing.assert_allclose(new_omega_dot, old_omega_dot, rtol=1e-10, atol=1e-12)


def test_quaternion_kinematics_scalar_last_is_identical(rng: np.random.Generator) -> None:
    """The default ``scalar_first=False`` quaternion kinematics is a verbatim match."""
    old_utils = pytest.importorskip("utils")
    q = rand_quat_array(rng)
    omega = rng.standard_normal(3)
    old_q = old_utils.Quaternion.from_array(q)
    new_q = Quaternion.from_array(q)
    np.testing.assert_allclose(new_q.kinematics(omega), old_q.kinematics(omega), rtol=1e-12, atol=1e-14)


def test_quaternion_kinematics_scalar_first_differs(rng: np.random.Generator) -> None:
    """DOCUMENTED DIFFERENCE: old ``kinematics(..., scalar_first=True)`` dropped the 0.5 factor.

    Old ``utils.py`` returns ``roll(xi @ omega, 1)``; new ``quaternion.py`` returns
    ``0.5 * roll(xi @ omega, 1)``. So the old result is exactly twice the new one. The new value is
    the consistent one (it equals the scalar-last derivative, just reordered).

    Its a bug in the old implementation.
    """
    old_utils = pytest.importorskip("utils")
    q = rand_quat_array(rng)
    omega = rng.standard_normal(3)
    old_q = old_utils.Quaternion.from_array(q)
    new_q = Quaternion.from_array(q)

    old_sf = old_q.kinematics(omega, scalar_first=True)
    new_sf = new_q.kinematics(omega, scalar_first=True)

    assert not np.allclose(old_sf, new_sf), "expected the scalar-first branch to differ"
    np.testing.assert_allclose(old_sf, 2.0 * new_sf, rtol=1e-12, atol=1e-14)
    # The new scalar-first result is just the scalar-last derivative reordered.
    np.testing.assert_allclose(new_sf, np.roll(new_q.kinematics(omega), 1), rtol=1e-12, atol=1e-14)
