"""Example-specific signal layouts for the satellite simulation.

Defines layouts for estimating, referencing, control inputs, and model dynamics
for the 3 reaction wheel + 3 magnetorquer satellite configuration.
"""

import dataclasses

from spacecraft.rigid_body import BASE_STATES, STATE

__all__ = [
    "BASE_STATES",
    "CONTROL",
    "ESTIMATE",
    "MODEL",
    "REFERENCE",
    "STATE",
]


@dataclasses.dataclass(frozen=True)
class _EstimateLayout:
    """Estimator output ``x_hat`` (length 19): ``[r(3), v(3), q(4), omega(3), b_body(3), h_wheel(3)]``."""

    r: slice = STATE.r
    v: slice = STATE.v
    q: slice = STATE.q
    omega: slice = STATE.omega
    b_body: slice = slice(13, 16)
    h_wheel: slice = slice(16, 19)


@dataclasses.dataclass(frozen=True)
class _ReferenceLayout:
    """Reference ``ref`` (length 7): ``[q_des(4), omega_des(3)]`` (ORC-relative)."""

    q_des: slice = slice(0, 4)
    omega_des: slice = slice(4, 7)


@dataclasses.dataclass(frozen=True)
class _ControlLayout:
    """LQR control vector (length 6): ``[tau_mtq(3), tau_rw(3)]``."""

    tau_mtq: slice = slice(0, 3)
    tau_rw: slice = slice(3, 6)


@dataclasses.dataclass(frozen=True)
class _ModelLayout:
    """Controller model of the plant: state ``[q(4), omega(3), h_w(3)]`` and input ``[u_mag(3), u_rw(3)]``.

    This is the full 10-state model the model-based controllers integrate; the linearized
    error system it reduces to has 9 states (the unit quaternion contributes only 3 DOF).
    """

    q: slice = slice(0, 4)
    omega: slice = slice(4, 7)
    h_w: slice = slice(7, 10)
    u_mag: slice = slice(0, 3)
    u_rw: slice = slice(3, 6)


ESTIMATE = _EstimateLayout()
REFERENCE = _ReferenceLayout()
CONTROL = _ControlLayout()
MODEL = _ModelLayout()
