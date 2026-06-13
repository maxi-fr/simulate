"""Attitude reference generators for the rigid body."""

from typing import Any, Self

import numpy as np

from simulate.component import NoLog
from simulate.reference import Reference


class OrbitReference(Reference[NoLog]):
    """Attitude reference, expressed **relative to the orbital (ORC) frame**.

    The emitted 7-vector is ``[q_bo(4), omega_bo(3)]`` (quaternion scalar-last): ``q_bo`` is the
    desired ORC->body rotation and ``omega_bo`` the desired body rate *relative to* the ORC frame of the reference.
    For nadir pointing the body coincides with the ORC frame, so this is constant ``[0, 0, 0, 1]``
    (identity) and ``[0, 0, 0]``.

    The reference is deliberately orbit-relative and constant: the inertial-frame attitude/rate
    targets are recovered inside the controller, which reconstructs the ORC frame and the orbital
    feedforward rate from the estimated orbit ``r, v`` in ``x_hat`` (see
    :func:`rigid_body.controller._attitude_error`). It therefore needs no orbit propagator of its own.
    """

    def __init__(self, dt: float) -> None:
        """Initialize with the sample time."""
        super().__init__(dt)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]))

    def update(self, t: float) -> tuple[float | np.ndarray, NoLog]:  # noqa: ARG002
        """Return the constant nadir reference ``[0, 0, 0, 1, 0, 0, 0]`` (ORC-relative)."""
        ref = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        return ref, NoLog()
