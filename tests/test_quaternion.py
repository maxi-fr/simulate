from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rigid_body.quaternion import Quaternion


def test_init_and_properties() -> None:
    vec = np.array([0.1, 0.2, 0.3])
    scalar = 0.9
    q = Quaternion(vec, scalar)

    np.testing.assert_array_equal(q.vec, vec)
    assert q.scalar == scalar

    # Test immutability (frozen dataclass)
    with pytest.raises(FrozenInstanceError):
        q.scalar = 0.5  # ty:ignore[invalid-assignment]


def test_from_array() -> None:
    # Test scalar_first=False (default) -> [v1, v2, v3, s]
    arr = [1, 2, 3, 4]
    q = Quaternion.from_array(arr)
    np.testing.assert_array_equal(q.vec, np.array([1, 2, 3]))
    assert q.scalar == 4

    # Test scalar_first=True -> [s, v1, v2, v3]
    q2 = Quaternion.from_array(arr, scalar_first=True)
    np.testing.assert_array_equal(q2.vec, np.array([2, 3, 4]))
    assert q2.scalar == 1


def test_to_array() -> None:
    q = Quaternion(np.array([1.0, 2.0, 3.0]), 4.0)

    arr_sf = q.to_array(scalar_first=True)
    np.testing.assert_array_equal(arr_sf, np.array([4.0, 1.0, 2.0, 3.0]))

    arr_sl = q.to_array(scalar_first=False)
    np.testing.assert_array_equal(arr_sl, np.array([1.0, 2.0, 3.0, 4.0]))


def test_multiplication() -> None:
    q1 = Quaternion(np.array([0.0, 0.0, 0.0]), 1.0)
    q2 = Quaternion(np.array([0.5, 0.5, 0.5]), 0.5)

    # Identity mult
    res = q1 * q2
    np.testing.assert_array_equal(res.vec, q2.vec)
    assert res.scalar == q2.scalar

    res2 = q2 * q1
    np.testing.assert_array_equal(res2.vec, q2.vec)
    assert res2.scalar == q2.scalar

    # Check JPL convention property: i*j = -k (in standard Hamilton i*j=k)
    qi = Quaternion(np.array([1.0, 0.0, 0.0]), 0.0)
    qj = Quaternion(np.array([0.0, 1.0, 0.0]), 0.0)
    qk = qi * qj

    # Expected: -k = [0, 0, -1], w=0
    np.testing.assert_array_equal(qk.vec, np.array([0.0, 0.0, -1.0]))
    assert qk.scalar == 0.0


def test_conjugate() -> None:
    q = Quaternion(np.array([1.0, 2.0, 3.0]), 4.0)
    q_conj = q.conjugate()

    np.testing.assert_array_equal(q_conj.vec, -q.vec)
    assert q_conj.scalar == q.scalar


def test_apply() -> None:
    # Rotation 90 deg about Z
    val = np.sin(np.deg2rad(45))
    q = Quaternion(np.array([0.0, 0.0, val]), val)

    v = np.array([1.0, 0.0, 0.0])
    v_rot = q.apply(v)

    # JPL apply usually rotates frame A to B. v_B = q.apply(v_A)
    # 90 deg Z rotation of frame implies X_A -> -Y_B
    np.testing.assert_allclose(v_rot, np.array([0.0, -1.0, 0.0]), atol=1e-7)


def test_scipy_interop() -> None:
    v = np.array([1.0, 2.0, 3.0])

    # Create a random rotation
    r = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)

    # Convert to custom Quaternion
    q = Quaternion.from_scipy(r)

    # Compare matrices
    np.testing.assert_allclose(r.apply(v), q.apply(v), atol=1e-7)


def test_xi() -> None:
    q = Quaternion(np.array([1.0, 2.0, 3.0]), 4.0)
    xi = q.xi
    assert xi.shape == (4, 3)
