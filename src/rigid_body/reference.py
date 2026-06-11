"""Attitude reference generators for the rigid body."""

from typing import Any, Self

import numpy as np

from simulate.component import NoLog
from simulate.reference import Reference


class NadirPointingReference(Reference[NoLog]):
    """Nadir-pointing attitude reference.

    The desired attitude is a constant reference in the BO frame.
    The emitted reference is the 7-vector ``[q_des(4), omega_des(3)]``
    (quaternion scalar-last), where q_des is [0, 0, 0, 1] and omega_des is [0, 0, 0].
    """

    def __init__(self, dt: float) -> None:
        """Initialize with the sample time."""
        super().__init__(dt)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]))

    def update(self, _: float) -> tuple[float | np.ndarray, NoLog]:
        """Return constant nadir pointing reference ``[0, 0, 0, 1, 0, 0, 0]``."""
        ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        return ref, NoLog()
