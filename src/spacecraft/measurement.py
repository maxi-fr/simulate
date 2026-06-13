"""Environment-coupled measurement Outputs for the rigid body.

These Outputs read the orbit state (and, where needed, the environment evaluated at
`epoch + t`) and return truth measurements in the body frame, to be paired with a
:class:`~simulate.sensor.Sensor` that adds noise. They mirror the epoch and coordinate
handling used by the environmental effectors in :mod:`spacecraft.effector`.

The dynamics-only sensors (rate gyro, star tracker, reaction-wheel tachometer) already have
Outputs in :mod:`spacecraft.rigid_body` (:class:`~spacecraft.rigid_body.RigidBodyRateOutput`,
:class:`~spacecraft.rigid_body.RigidBodyAttitudeOutput`,
:class:`~spacecraft.rigid_body.ReactionWheelTelemetryOutput`); pair those directly with a
sensor. This module adds the Outputs that need the environment models.
"""

import datetime
from typing import Any, Self

import numpy as np

from simulate.component import NoLog
from simulate.output import Output

from .environment import is_in_shadow, magnetic_field_vector, sun_position
from .frames import eci_to_geodedic
from .quaternion import Quaternion
from .rigid_body import POSITION, QUATERNION, VELOCITY


def _ensure_utc(epoch: datetime.datetime) -> datetime.datetime:
    """Return ``epoch`` as a timezone-aware UTC datetime (naive inputs are assumed UTC)."""
    return epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=datetime.UTC)


class MagneticFieldOutput(Output[NoLog]):
    """Magnetometer truth: the IGRF magnetic field at the body, in the body frame [T].

    The inertial position is converted to geodetic coordinates and passed to
    :func:`environment.magnetic_field_vector` (which returns the field in the inertial frame
    at ``epoch + t``); the field is then rotated into the body frame. Pair with a
    :class:`~simulate.sensor.RandomWalkBiasSensor` to model magnetometer bias and noise.
    """

    def __init__(self, dt: float, epoch: datetime.datetime) -> None:
        """Initialize with the sample time and the simulation epoch (``t = 0``)."""
        super().__init__(dt)
        self.epoch = _ensure_utc(epoch)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), epoch=datetime.datetime.fromisoformat(config["epoch"]))

    def update(
        self,
        t: float,
        x: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, NoLog]:
        """Compute the body-frame IGRF magnetic field from position and attitude."""
        dt_utc = self.epoch + datetime.timedelta(seconds=t)
        r_eci = np.asarray(x[POSITION], dtype=float)  # ty:ignore[not-subscriptable]
        lat_deg, lon_deg, alt_m = eci_to_geodedic(r_eci)

        b_eci = magnetic_field_vector(dt_utc, float(lat_deg), float(lon_deg), float(alt_m))
        q_bi = Quaternion.from_array(x[QUATERNION])  # ty:ignore[not-subscriptable]
        return q_bi.apply(b_eci), NoLog()


class SunDirectionOutput(Output[NoLog]):
    """Sun-sensor truth: the unit sun direction in the body frame, zeroed in eclipse.

    The sun position comes from :func:`environment.sun_position` at ``epoch + t`` and the
    cylindrical :func:`environment.is_in_shadow` model decides eclipse. In sunlight the
    spacecraft-to-Sun unit vector is rotated into the body frame; in eclipse a zero vector is
    returned (the sun sensor is inactive). Pair with a :class:`~simulate.sensor.GaussianSensor`.
    """

    def __init__(self, dt: float, epoch: datetime.datetime) -> None:
        """Initialize with the sample time and the simulation epoch (``t = 0``)."""
        super().__init__(dt)
        self.epoch = _ensure_utc(epoch)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), epoch=datetime.datetime.fromisoformat(config["epoch"]))

    def update(
        self,
        t: float,
        x: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, NoLog]:
        """Compute the body-frame unit sun direction, or zeros when in eclipse."""
        dt_utc = self.epoch + datetime.timedelta(seconds=t)
        r_eci = np.asarray(x[POSITION], dtype=float)  # ty:ignore[not-subscriptable]
        sun_pos = sun_position(dt_utc)

        if is_in_shadow(r_eci, sun_pos):
            return np.zeros(3, dtype=float), NoLog()

        sc_to_sun = sun_pos - r_eci
        sun_dir_eci = sc_to_sun / np.linalg.norm(sc_to_sun)
        q_bi = Quaternion.from_array(x[QUATERNION])  # ty:ignore[not-subscriptable]
        return q_bi.apply(sun_dir_eci), NoLog()


class GpsOutput(Output[NoLog]):
    """GPS truth: inertial position (and optionally velocity) sliced from the state.

    Returns ``[r(3), v(3)]`` when ``include_velocity`` is set (the default), otherwise the
    position ``r(3)`` alone. Pair with a :class:`~simulate.sensor.GaussianSensor`.
    """

    def __init__(self, dt: float, *, include_velocity: bool = True) -> None:
        """Initialize with the sample time and whether to also report velocity."""
        super().__init__(dt)
        self.include_velocity = include_velocity

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), include_velocity=bool(config.get("include_velocity", True)))

    def update(
        self,
        t: float,  # noqa: ARG002
        x: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, NoLog]:
        """Select the inertial position (and optionally velocity) from the state."""
        if self.include_velocity:
            return np.concatenate([x[POSITION], x[VELOCITY]]), NoLog()  # ty:ignore[not-subscriptable]
        return x[POSITION], NoLog()  # ty:ignore[not-subscriptable]
