import datetime

import numpy as np

from rigid_body.orbit_dynamics import SGP4
from rigid_body.quaternion import Quaternion
from rigid_body.reference import NadirPointingReference

_EPOCH = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
_MEAN_MOTION_REV_DAY = 15.2  # ~520 km circular orbit


def _iss_like_propagator() -> SGP4:
    return SGP4.from_elements(
        e=0.0,
        i=51.6,
        raan=0.0,
        arg_pe=0.0,
        M0=0.0,
        MM=_MEAN_MOTION_REV_DAY,
        t0=_EPOCH,
    )


def test_reference_shape_and_unit_quaternion() -> None:
    ref_gen = NadirPointingReference(dt=1.0, propagator=_iss_like_propagator(), epoch=_EPOCH)

    ref, _ = ref_gen.update(0.0)
    ref = np.asarray(ref)

    assert ref.shape == (7,)
    np.testing.assert_allclose(np.linalg.norm(ref[:4]), 1.0, atol=1e-9)


def test_reference_points_nadir_over_orbit() -> None:
    propagator = _iss_like_propagator()
    ref_gen = NadirPointingReference(dt=1.0, propagator=propagator, epoch=_EPOCH)

    for t in (0.0, 600.0, 1800.0, 3000.0):
        ref, _ = ref_gen.update(t)
        q = Quaternion.from_array(np.asarray(ref)[:4])

        r_eci, _ = propagator.propagate(_EPOCH + datetime.timedelta(seconds=t))
        nadir = -r_eci / np.linalg.norm(r_eci)

        # The desired body z axis points at nadir by construction of the ORC frame.
        np.testing.assert_allclose(q.apply(nadir), np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_reference_rate_matches_mean_motion() -> None:
    ref_gen = NadirPointingReference(dt=1.0, propagator=_iss_like_propagator(), epoch=_EPOCH)

    ref, _ = ref_gen.update(0.0)
    omega_des = np.asarray(ref)[4:]

    mean_motion = 2.0 * np.pi * _MEAN_MOTION_REV_DAY / 86400.0  # rad/s
    np.testing.assert_allclose(np.linalg.norm(omega_des), mean_motion, rtol=0.1)
