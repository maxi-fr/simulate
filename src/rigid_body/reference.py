"""Attitude reference generators for the rigid body."""

import datetime
from typing import Any, Self

import numpy as np

from simulate.component import NoLog
from simulate.reference import Reference

from .frames import orbital_rate, orc_from_orbit
from .orbit_dynamics import SGP4


def _ensure_utc(epoch: datetime.datetime) -> datetime.datetime:
    """Return ``epoch`` as a timezone-aware UTC datetime (naive inputs are assumed UTC)."""
    return epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=datetime.UTC)


# TODO: nadir pointing reference in the BO frame is much easier to define (no need for orbit propagator)
class NadirPointingReference(Reference[NoLog]):
    """Nadir-pointing attitude reference driven by an SGP4 orbit propagator.

    The desired attitude is deterministic in the orbit, so it is derived from an SGP4
    propagation rather than the feedback state. At each step the orbit is propagated to
    ``epoch + t`` and the desired attitude/rate are taken from the orbital reference (LVLH)
    frame: the desired body frame is the inertial->ORC rotation
    (:func:`~rigid_body.frames.orc_from_orbit`) and the feedforward body rate is the ORC
    frame's angular velocity (:func:`~rigid_body.frames.orbital_rate`). The emitted reference
    is the 7-vector ``[q_des(4), omega_des(3)]`` (quaternion scalar-last), matching the
    ``[q, omega]`` layout the attitude controllers consume.
    """

    def __init__(self, dt: float, propagator: SGP4, epoch: datetime.datetime) -> None:
        """Initialize with the sample time, an SGP4 propagator, and the epoch (``t = 0``)."""
        super().__init__(dt)
        self.propagator = propagator
        self.epoch = _ensure_utc(epoch)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary.

        The orbit is specified by ``tle`` (a ``[line1, line2]`` pair). ``epoch`` is the simulation start (ISO 8601); for the
        element form it also serves as the element epoch ``t0``.
        """
        epoch = datetime.datetime.fromisoformat(config["epoch"])

        tle1, tle2 = config["tle"]
        propagator = SGP4.from_tle(tle1, tle2)

        return cls(dt=float(config["dt"]), propagator=propagator, epoch=epoch)

    def update(self, t: float) -> tuple[float | np.ndarray, NoLog]:
        """Propagate the orbit to ``epoch + t`` and return ``[q_des(4), omega_des(3)]``."""
        dt_utc = self.epoch + datetime.timedelta(seconds=t)
        r_eci, v_eci = self.propagator.propagate(dt_utc)
        q_des = orc_from_orbit(r_eci, v_eci)
        omega_des = orbital_rate(r_eci, v_eci)
        ref = np.concatenate([q_des.to_array(), omega_des])
        return ref, NoLog()
