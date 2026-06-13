"""Differential test for ``gravity_gradient``, the one disturbance whose signature changed.

The other disturbance functions (``non_spherical_gravity_forces``, ``third_body_forces``,
``aerodynamic_drag``, ``solar_radiation_pressure``) are verbatim copies and are not retested.

Old ``gravity_gradient(r, R_BO, J_B)`` takes the ORC->body rotation and applies it to the ORC nadir
axis ``[0, 0, 1]``; new ``gravity_gradient(r, q_bi, J_B)`` takes the inertial->body rotation and
applies it to the inertial nadir ``-r/|r|``. Both reduce to the same body-frame nadir ``o_body`` and
the same torque ``3*mu/|r|^3 * (o_body x J o_body)``. We pick ``r`` and a body attitude, compute
``o_body`` the new way, then synthesize an ``R_BO`` that maps ``[0, 0, 1]`` onto the same ``o_body``
so the two functions receive equivalent inputs.
"""

import numpy as np
import pytest
from diffhelpers import leo_rv, rand_inertia, rand_quat_array
from scipy.spatial.transform import Rotation

from spacecraft.effector import EarthGravity, RigidBodyState
from spacecraft.quaternion import Quaternion


def test_gravity_gradient_matches_after_frame_transform(rng: np.random.Generator) -> None:
    """Gravity-gradient torque matches once the changed frame argument is reconciled.

    Old took the ORC->body rotation, new takes inertial->body; both ultimately need the same
    body-frame nadir ``o_body``. For 20 random (orbit, attitude, inertia) cases we compute
    ``o_body`` the new way, synthesize an old ``R_BO`` that maps the ORC nadir ``[0,0,1]`` onto that
    same ``o_body``, and check the two torque vectors are equal -- so the frame swap, not the physics,
    is all that changed.
    """
    old = pytest.importorskip("simulation.disturbances")
    old_utils = pytest.importorskip("utils")

    unused_arr = np.empty(3)

    for _ in range(20):
        r, _ = leo_rv(rng)
        inertia = rand_inertia(rng)
        q_bi = Quaternion.from_array(rand_quat_array(rng))

        state = RigidBodyState(r, unused_arr, q_bi, unused_arr)
        earth_grav_eff = EarthGravity()
        earth_grav_eff.bind(0, inertia)

        o_body = q_bi.apply(-r / np.linalg.norm(r))
        # Any rotation mapping the ORC nadir [0, 0, 1] onto o_body gives the matching R_BO.
        align_rot, _ = Rotation.align_vectors([o_body], [[0.0, 0.0, 1.0]])
        q_bo = old_utils.Quaternion.from_scipy(align_rot, canonical=False)

        _, new_tau, _ = earth_grav_eff.calc_contributions(0, state, unused_arr, unused_arr)
        old_tau = old.gravity_gradient(r, q_bo, inertia)
        np.testing.assert_allclose(new_tau, old_tau, rtol=1e-10, atol=1e-14)
