"""Environment-coupled measurement models for the rigid body.

These models read the orbit state (and, where needed, the environment evaluated at
`epoch + t`) and return truth measurements in the body frame, to be owned by a
:class:`~simulate.sensor.Sensor` that adds noise. They mirror the epoch and coordinate
handling used by the environmental effectors in :mod:`spacecraft.effector`.

A measurement model is a plain callable ``(t, x, u) -> y`` (see
:data:`simulate.measurement_model.MeasurementModel`). The dynamics-only measurements (rate
gyro, star tracker, reaction-wheel tachometer) live in :mod:`spacecraft.rigid_body`
(:func:`~spacecraft.rigid_body.rigid_body_rate`,
:func:`~spacecraft.rigid_body.rigid_body_attitude`,
:class:`~spacecraft.rigid_body.ReactionWheelTelemetry`); this module adds the ones that
need the environment models.
"""

import datetime

import numpy as np

from .environment import is_in_shadow, magnetic_field_vector, sun_position
from .frames import eci_to_geodedic
from .quaternion import Quaternion
from .rigid_body import POSITION, QUATERNION, VELOCITY


def _ensure_utc(epoch: datetime.datetime) -> datetime.datetime:
    """Return ``epoch`` as a timezone-aware UTC datetime (naive inputs are assumed UTC)."""
    return epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=datetime.UTC)


class MagneticFieldMeasurement:
    """Magnetometer truth: the IGRF magnetic field at the body, in the body frame [T].

    The inertial position is converted to geodetic coordinates and passed to
    :func:`environment.magnetic_field_vector` (which returns the field in the inertial frame
    at ``epoch + t``); the field is then rotated into the body frame. Pair with a
    :class:`~simulate.sensor.RandomWalkBiasSensor` to model magnetometer bias and noise.
    """

    def __init__(self, epoch: str | datetime.datetime) -> None:
        """Initialize with the simulation epoch (``t = 0``); ISO strings are parsed."""
        if isinstance(epoch, str):
            epoch = datetime.datetime.fromisoformat(epoch)
        self.epoch = _ensure_utc(epoch)

    def __call__(
        self,
        t: float,
        x: float | np.ndarray,
        _u: float | np.ndarray,
    ) -> np.ndarray:
        """Compute the body-frame IGRF magnetic field from position and attitude."""
        dt_utc = self.epoch + datetime.timedelta(seconds=t)
        r_eci = np.asarray(x[POSITION], dtype=float)  # ty:ignore[not-subscriptable]
        lat_deg, lon_deg, alt_m = eci_to_geodedic(r_eci)

        b_eci = magnetic_field_vector(dt_utc, float(lat_deg), float(lon_deg), float(alt_m))
        q_bi = Quaternion.from_array(x[QUATERNION])  # ty:ignore[not-subscriptable]
        return q_bi.apply(b_eci)


class SunDirectionMeasurement:
    """Sun-sensor truth: the unit sun direction in the body frame, zeroed in eclipse.

    The sun position comes from :func:`environment.sun_position` at ``epoch + t`` and the
    cylindrical :func:`environment.is_in_shadow` model decides eclipse. In sunlight the
    spacecraft-to-Sun unit vector is rotated into the body frame; in eclipse a zero vector is
    returned (the sun sensor is inactive). Pair with a :class:`~simulate.sensor.GaussianSensor`.
    """

    def __init__(self, epoch: str | datetime.datetime) -> None:
        """Initialize with the simulation epoch (``t = 0``); ISO strings are parsed."""
        if isinstance(epoch, str):
            epoch = datetime.datetime.fromisoformat(epoch)
        self.epoch = _ensure_utc(epoch)

    def __call__(
        self,
        t: float,
        x: float | np.ndarray,
        _u: float | np.ndarray,
    ) -> np.ndarray:
        """Compute the body-frame unit sun direction, or zeros when in eclipse."""
        dt_utc = self.epoch + datetime.timedelta(seconds=t)
        r_eci = np.asarray(x[POSITION], dtype=float)  # ty:ignore[not-subscriptable]
        sun_pos = sun_position(dt_utc)

        if is_in_shadow(r_eci, sun_pos):
            return np.zeros(3, dtype=float)

        sc_to_sun = sun_pos - r_eci
        sun_dir_eci = sc_to_sun / np.linalg.norm(sc_to_sun)
        q_bi = Quaternion.from_array(x[QUATERNION])  # ty:ignore[not-subscriptable]
        return q_bi.apply(sun_dir_eci)


class GpsMeasurement:
    """GPS truth: inertial position (and optionally velocity) sliced from the state.

    Returns ``[r(3), v(3)]`` when ``include_velocity`` is set (the default), otherwise the
    position ``r(3)`` alone. Pair with a :class:`~simulate.sensor.GaussianSensor`.
    """

    def __init__(self, *, include_velocity: bool = True) -> None:
        """Initialize with whether to also report velocity."""
        self.include_velocity = include_velocity

    def __call__(
        self,
        _t: float,
        x: float | np.ndarray,
        _u: float | np.ndarray,
    ) -> np.ndarray:
        """Select the inertial position (and optionally velocity) from the state."""
        if self.include_velocity:
            return np.concatenate([x[POSITION], x[VELOCITY]])  # ty:ignore[not-subscriptable]
        return x[POSITION]  # ty:ignore[not-subscriptable]
