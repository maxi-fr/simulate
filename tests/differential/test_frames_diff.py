"""Differential tests for the reference-frame kinematics renamed/restructured during the port.

Old ``simulation.kinematics`` returned scipy ``Rotation`` objects and ``[roll, pitch, yaw]`` Euler
angles; new ``rigid_body.frames`` returns :class:`Quaternion` objects and ``[pitch, roll, yaw]``
(intrinsic ``Y-X-Z``). The correspondences, with the convention transforms made explicit:

* ``orc_to_eci`` (ORC->ECI matrix) is the inverse of ``orc_from_orbit`` (ECI->ORC quaternion).
* ``euler_ocr_to_sbc(roll, pitch, yaw)`` == ``quaternion_from_euler([pitch, roll, yaw], degrees=True)``.
* ``to_euler`` returns ``[roll, pitch, yaw]`` == ``euler_from_quaternion(q_bo)[[1, 0, 2]]``.
* ``orc_to_sbc`` == ``q_bi (x) orc_from_orbit(r, v).conjugate()``.
"""

import numpy as np
import pytest
from diffhelpers import leo_rv, rand_quat_array

from rigid_body.frames import euler_from_quaternion, orc_from_orbit, quaternion_from_euler
from rigid_body.quaternion import Quaternion


def test_orc_to_eci_is_inverse_of_orc_from_orbit(rng: np.random.Generator) -> None:
    """The orbital-frame builders are inverses: old ORC->ECI matrix == (new ECI->ORC matrix) transposed.

    The port flipped the returned rotation's direction (and type: scipy ``Rotation`` -> ``Quaternion``),
    so for 20 random LEO states we check the two describe the same frame by comparing one matrix to the
    transpose of the other.
    """
    old = pytest.importorskip("simulation.kinematics")
    for _ in range(20):
        r, v = leo_rv(rng)
        old_mat = old.orc_to_eci(r, v).as_matrix()  # ORC -> ECI
        new_mat = orc_from_orbit(r, v).to_rot_mat()  # ECI -> ORC
        np.testing.assert_allclose(old_mat, new_mat.T, rtol=1e-10, atol=1e-12)


def test_euler_to_rotation_matches(rng: np.random.Generator) -> None:
    """Euler-angle -> rotation agrees once the argument order is matched.

    Old ``euler_ocr_to_sbc(roll, pitch, yaw)`` and new ``quaternion_from_euler([pitch, roll, yaw])``
    both build an intrinsic ``Y-X-Z`` rotation but take their angles in a different order; feeding the
    reordered angles, the resulting rotation matrices must be identical.
    """
    old = pytest.importorskip("simulation.kinematics")
    for _ in range(20):
        roll, yaw = rng.uniform(-80.0, 80.0, size=2)
        pitch = rng.uniform(-170.0, 170.0)
        old_mat = old.euler_ocr_to_sbc(roll, pitch, yaw).as_matrix()
        new_mat = quaternion_from_euler([pitch, roll, yaw], degrees=True).to_rot_mat()
        np.testing.assert_allclose(new_mat, old_mat, rtol=1e-10, atol=1e-12)


def test_to_euler_matches_with_rollpitch_swap(rng: np.random.Generator) -> None:
    """Quaternion -> Euler agrees after accounting for the roll/pitch column swap.

    Old ``to_euler`` returns ``[roll, pitch, yaw]`` while new ``euler_from_quaternion`` returns
    ``[pitch, roll, yaw]``. We synthesize ``q_bi`` from a moderate ORC-relative attitude (bounded
    ``roll`` to dodge the ``Y-X-Z`` gimbal singularity), run both decompositions, and check the old
    angles equal the new ones with indices ``[1, 0, 2]`` (i.e. roll<->pitch swapped back).
    """
    old = pytest.importorskip("simulation.kinematics")
    old_utils = pytest.importorskip("utils")
    for _ in range(20):
        r, v = leo_rv(rng)
        # Build q_bi from a moderate ORC-relative attitude (|roll| < 80 deg avoids the YXZ singularity).
        roll, yaw = rng.uniform(-70.0, 70.0, size=2)
        pitch = rng.uniform(-150.0, 150.0)
        q_bo = quaternion_from_euler([pitch, roll, yaw], degrees=True)
        q_bi = q_bo * orc_from_orbit(r, v)
        q_arr = q_bi.to_array()

        old_angles = old.to_euler(old_utils.Quaternion.from_array(q_arr), r, v)  # [roll, pitch, yaw]
        new_angles = euler_from_quaternion(q_bo, degrees=True)  # [pitch, roll, yaw]
        np.testing.assert_allclose(old_angles, new_angles[[1, 0, 2]], rtol=1e-8, atol=1e-8)


def test_orc_to_sbc_matches(rng: np.random.Generator) -> None:
    """The ORC->body rotation matches: old ``orc_to_sbc`` == new ``q_bi (x) orc_from_orbit(r,v)^-1``.

    Old exposed this as a dedicated function; in the new code it is just a composition. For 20 random
    attitudes/orbits we compare rotation matrices (rather than quaternions) to sidestep the
    quaternion double-cover sign ambiguity.
    """
    old = pytest.importorskip("simulation.kinematics")
    old_utils = pytest.importorskip("utils")
    for _ in range(20):
        r, v = leo_rv(rng)
        q_arr = rand_quat_array(rng)

        old_rbo = old.orc_to_sbc(old_utils.Quaternion.from_array(q_arr), r, v)
        new_rbo = Quaternion.from_array(q_arr) * orc_from_orbit(r, v).conjugate()
        np.testing.assert_allclose(new_rbo.to_rot_mat(), old_rbo.to_rot_mat(), rtol=1e-10, atol=1e-12)
