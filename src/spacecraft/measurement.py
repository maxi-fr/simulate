"""Environment-coupled measurement models for the rigid body.

These models read the orbit state (and, where needed, the environment evaluated at
`epoch + t`) and return truth measurements in the body frame, to be owned by a
:class:`~simulate.sensor.Sensor` that adds noise. They mirror the epoch and coordinate
handling used by the environmental effectors in :mod:`spacecraft.effector`.

A measurement model is a plain callable ``(t, x, u) -> y`` (see
:data:`simulate.sensor.MeasurementModel`). The dynamics-only measurements (rate
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
from .signals import BASE_STATES, STATE


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
        r_eci = np.asarray(x[STATE.r], dtype=float)  # ty:ignore[not-subscriptable]
        lat_deg, lon_deg, alt_m = eci_to_geodedic(r_eci)

        b_eci = magnetic_field_vector(dt_utc, float(lat_deg), float(lon_deg), float(alt_m))
        q_bi = Quaternion.from_array(x[STATE.q])  # ty:ignore[not-subscriptable]
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
        r_eci = np.asarray(x[STATE.r], dtype=float)  # ty:ignore[not-subscriptable]
        sun_pos = sun_position(dt_utc)

        if is_in_shadow(r_eci, sun_pos):
            return np.zeros(3, dtype=float)

        sc_to_sun = sun_pos - r_eci
        sun_dir_eci = sc_to_sun / np.linalg.norm(sc_to_sun)
        q_bi = Quaternion.from_array(x[STATE.q])  # ty:ignore[not-subscriptable]
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
            return np.concatenate([x[STATE.r], x[STATE.v]])  # ty:ignore[not-subscriptable]
        return x[STATE.r]  # ty:ignore[not-subscriptable]


def rigid_body_attitude(_t: float, x: float | np.ndarray, _u: float | np.ndarray) -> np.ndarray:
    """Attitude measurement: the body->inertial unit quaternion ``q`` ``(4, 1)``.

    Pair with a :class:`~simulate.sensor.GaussianSensor` to model a star tracker. Note that
    additive noise on ``q`` yields a non-unit quaternion; consumers must renormalize.

    Returns
    -------
    np.ndarray
        The body->inertial unit quaternion ``q``, shape ``(4,)``.
    """
    return x[STATE.q]  # ty:ignore[not-subscriptable]


def rigid_body_rate(_t: float, x: float | np.ndarray, _u: float | np.ndarray) -> np.ndarray:
    """Angular-rate measurement: the body-frame angular velocity ``omega`` ``(3, 1)``.

    Pair with a :class:`~simulate.sensor.GaussianSensor` to model a rate gyro.

    Returns
    -------
    np.ndarray
        The body-frame angular velocity ``omega``, shape ``(3,)``.
    """
    return x[STATE.omega]  # ty:ignore[not-subscriptable]


class ReactionWheelTelemetry:
    """Effector telemetry: a single effector internal state (e.g. a wheel's momentum ``h_w``).

    ``index`` is the absolute position of the effector state in the rigid body state vector;
    effector states begin at :data:`BASE_STATES` in composition order, so the first effector's
    first state is ``BASE_STATES``.
    """

    def __init__(self, index: int = BASE_STATES) -> None:
        """Initialize with the absolute state-vector index of the effector state to report."""
        self.index = index

    def __call__(self, _t: float, x: float | np.ndarray, _u: float | np.ndarray) -> np.ndarray:
        """Select the effector internal state at ``index`` from the full rigid body state."""
        return x[self.index : self.index + 1]  # ty:ignore[not-subscriptable]
