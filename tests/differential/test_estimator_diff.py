"""Differential tests for the attitude MEKF, renamed ``AttitudeEKF`` -> ``AttitudeMEKF``.

The math kernels were ported essentially verbatim, just reorganized: ``skew`` -> ``_skew``,
``_continuous_FG`` -> ``_continuous_fg`` (instance method -> staticmethod), the generic ``_update``,
the gyro ``predict`` step, and the merged ``update_sun``/``update_mag`` -> ``update_vector``. Each is
exercised on identical inputs with both filters started from the same state/covariances.
"""

from typing import Any

import numpy as np
import pytest
from diffhelpers import rand_quat_array, rand_unit_vec

from rigid_body.estimator import AttitudeMEKF, _skew


def _filters(q0: np.ndarray) -> tuple[Any, AttitudeMEKF]:
    """Build an old ``AttitudeEKF`` and a new ``AttitudeMEKF`` with identical configuration."""
    old_mod = pytest.importorskip("flight_software.estimators")
    p0 = np.diag([1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])
    qc = np.diag([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9])
    r_sun = np.eye(3) * 1e-3
    r_mag = np.eye(3) * 1e-2
    old = old_mod.AttitudeEKF(q0.copy(), p0.copy(), qc.copy(), r_sun.copy(), r_mag.copy())
    new = AttitudeMEKF(q0.copy(), p0.copy(), qc.copy(), r_sun.copy(), r_mag.copy())
    return old, new


def test_skew_matches(rng: np.random.Generator) -> None:
    """The skew-symmetric (cross-product) matrix builder is bit-identical (``skew`` -> ``_skew``)."""
    old_mod = pytest.importorskip("flight_software.estimators")
    for _ in range(20):
        v = rng.standard_normal(3)
        np.testing.assert_array_equal(_skew(v), old_mod.skew(v))


def test_continuous_fg_matches(rng: np.random.Generator) -> None:
    """The continuous error-state Jacobian/noise matrices ``(F, G)`` match for random ``omega``.

    Confirms the error-dynamics linearization survived the move from an instance method
    (``_continuous_FG``) to a staticmethod (``_continuous_fg``).
    """
    old, _ = _filters(rand_quat_array(rng))
    for _ in range(20):
        omega = rng.uniform(-0.1, 0.1, size=3)
        old_f, old_g = old._continuous_FG(omega)
        new_f, new_g = AttitudeMEKF._continuous_fg(omega)
        np.testing.assert_allclose(new_f, old_f, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(new_g, old_g, rtol=1e-12, atol=1e-14)


def test_generic_update_matches(rng: np.random.Generator) -> None:
    """The generic EKF correction ``_update`` leaves both filters in the same posterior state.

    Driving both with the same innovation/Jacobian/noise ``(z, z_pred, H, R)``, the updated
    quaternion, gyro bias and covariance must agree -- covering the Kalman gain, the multiplicative
    quaternion reset and the Joseph-form covariance update together.
    """
    old_mod = pytest.importorskip("flight_software.estimators")
    old, new = _filters(rand_quat_array(rng))
    z_meas = rand_unit_vec(rng)
    z_pred = rand_unit_vec(rng)
    h = np.hstack((old_mod.skew(z_pred), np.zeros((3, 3))))
    r_meas = np.eye(3) * 1e-2

    old._update(z_meas, z_pred, h, r_meas)
    new._update(z_meas, z_pred, h, r_meas)

    np.testing.assert_allclose(new.q, old.q, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(new.b, old.b, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(new.P, old.P, rtol=1e-10, atol=1e-12)


def test_predict_matches(rng: np.random.Generator) -> None:
    """The gyro-driven prediction agrees, despite the timekeeping API change.

    Old ``predict`` takes datetimes (and its first call only seeds the time reference); new
    ``predict`` takes an explicit ``dt``. We drive both over the same step and compare the propagated
    quaternion and covariance.
    """
    import datetime

    old, new = _filters(rand_quat_array(rng))
    omega = rng.uniform(-0.05, 0.05, size=3)
    dt = 0.1
    t0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)

    old.predict(t0, omega)  # first call only initializes the time reference
    old.predict(t0 + datetime.timedelta(seconds=dt), omega)
    new.predict(omega, dt)

    np.testing.assert_allclose(new.q, old.q, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(new.P, old.P, rtol=1e-10, atol=1e-12)


def test_update_vector_matches_old_wiring(rng: np.random.Generator) -> None:
    """The merged ``update_vector`` reproduces the old ``update_sun``/``update_mag`` wiring.

    Old had two near-identical line-of-sight updates; the new code unifies them into
    ``update_vector(ref_eci, body_meas, R)``. We run the new method, then by hand replicate the old
    recipe (predict ``q.apply(ref)``, normalize prediction and measurement, build ``H = [skew(pred), 0]``,
    call ``_update``) and confirm both reach the same posterior ``q``/``b``/``P``.
    """
    old_mod = pytest.importorskip("flight_software.estimators")
    old_utils = pytest.importorskip("utils")
    old, new = _filters(rand_quat_array(rng))

    ref_eci = rand_unit_vec(rng) * 3.0
    body_meas = rand_unit_vec(rng) * 0.9
    r_meas = np.eye(3) * 1e-2

    new.update_vector(ref_eci, body_meas, r_meas)

    pred = old_utils.Quaternion.from_array(old.q).apply(ref_eci)
    pred_n = pred / np.linalg.norm(pred)
    meas_n = body_meas / np.linalg.norm(body_meas)
    h = np.hstack((old_mod.skew(pred_n), np.zeros((3, 3))))
    old._update(meas_n, pred_n, h, r_meas)

    np.testing.assert_allclose(new.q, old.q, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(new.b, old.b, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(new.P, old.P, rtol=1e-10, atol=1e-12)
