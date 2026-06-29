"""Full nonlinear state estimator: orbit Kalman filter + attitude MEKF + environment exposure.

The framework feeds the estimator the concatenated measurement vector ``y_mea`` (every sensor
channel, in the simulation's ``outputs``/``sensors`` order) and the control input ``u``, and
expects ``x_hat`` back (see :mod:`simulate.simulation`). This estimator:

* splits ``y_mea`` into named channels with :class:`MeasurementLayout` (Step 4.1),
* runs a linear :class:`OrbitKalmanFilter` over ``[r, v]`` driven by the GPS channel (Step 4.2),
* runs an :class:`AttitudeMEKF` (multiplicative EKF over attitude error + gyro bias) driven by
  the gyro, magnetometer, sun and optional star-tracker channels (Step 4.3),
* exposes the environment (magnetic field, sun direction, density, geodetic position) evaluated
  at the *estimated* orbit (Step 4.4),

and assembles ``x_hat = [r(3), v(3), q(4), omega(3), b_body(3), h_wheel(3)]`` plus a rich log
(Step 4.5). The first 13 entries are the orbit/attitude state; the trailing ``b_body`` (estimated
magnetic field in the body frame [T]) and ``h_wheel`` (estimated reaction-wheel angular momentum in
the body frame [N*m*s]) are exposed for the attitude controllers (magnetorquer allocation + wheel
momentum dumping), which only receive ``x_hat``. The gyro bias and the remaining environment
variables ride in the log.
"""

import dataclasses
import datetime
from typing import Any

import numpy as np

from simulate.integrator import rk4

from .orbit_dynamics import MU
from .quaternion import Quaternion

_EPS = 1e-12
_ERROR_STATE = 6  # MEKF error state: [delta_theta(3), delta_bias(3)]
_ORBIT_STATE = 6  # orbit state: [r(3), v(3)]


def _ensure_utc(epoch: datetime.datetime) -> datetime.datetime:
    """Return ``epoch`` as a timezone-aware UTC datetime (naive inputs are assumed UTC)."""
    return epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=datetime.UTC)


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric (cross-product) matrix of a 3-vector."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _as_matrix(value: Any, n: int) -> np.ndarray:  # noqa: ANN401
    """Coerce a config entry to an ``(n, n)`` matrix, expanding a length-n vector to a diagonal."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == n:
        return np.diag(arr)
    return arr.reshape(n, n)


@dataclasses.dataclass(frozen=True)
class MeasurementLayout:
    """Maps a concatenated measurement vector to named channels.

    ``channels`` is an ordered tuple of ``(name, dim)`` pairs mirroring the simulation's
    ``outputs``/``sensors`` ordering. Recognized names used by the estimator are ``"gps"``,
    ``"gyro"``, ``"magnetometer"``, ``"sun"`` and ``"star_tracker"``; any other channel
    (e.g. ``"tachometer"``) is parsed for layout bookkeeping but otherwise ignored.
    """

    channels: tuple[tuple[str, int], ...]

    def split(self, y_mea: np.ndarray) -> dict[str, np.ndarray]:
        """Slice ``y_mea`` into ``{name: vector}`` following the configured channel order."""
        out: dict[str, np.ndarray] = {}
        idx = 0
        for name, dim in self.channels:
            out[name] = y_mea[idx : idx + dim]
            idx += dim
        return out

    @property
    def size(self) -> int:
        """Total length of a measurement vector described by this layout."""
        return sum(dim for _, dim in self.channels)


class OrbitKalmanFilter:
    """Linear Kalman filter over ``[r, v]`` with two-body prediction and a GPS update.

    The mean is propagated with the nonlinear two-body dynamics (RK4); the covariance uses the
    first-order transition ``Phi = I + F dt`` with the two-body Jacobian. The GPS update is a
    standard Joseph-form linear update with measurement matrix ``H``.
    """

    def __init__(  # noqa: PLR0913
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """Initialize from the initial state/covariance and the GPS measurement model ``(H, R)``."""
        self.x = np.concatenate([r0, v0])
        self.P = P0
        self.Q = Q
        self.H = H
        self.R = R

    @staticmethod
    def _f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG004
        """Two-body state derivative ``[v, -mu r / |r|^3]``."""
        r = x[:3]
        v = x[3:]
        accel = -MU * r / np.linalg.norm(r) ** 3
        return np.concatenate([v, accel])

    def _jacobian(self, r: np.ndarray) -> np.ndarray:
        """Continuous-time state Jacobian ``F`` of the two-body dynamics at position ``r``."""
        rn = np.linalg.norm(r)
        gradient = -MU * (np.eye(3) / rn**3 - 3.0 * np.outer(r, r) / rn**5)
        F = np.zeros((_ORBIT_STATE, _ORBIT_STATE))
        F[:3, 3:] = np.eye(3)
        F[3:, :3] = gradient
        return F

    def predict(self, dt: float) -> None:
        """Propagate the mean (RK4 two-body) and covariance over ``dt``."""
        if dt <= 0.0:
            return
        Phi = np.eye(_ORBIT_STATE) + self._jacobian(self.x[:3]) * dt
        self.x = rk4(self._f, 0.0, dt, self.x, np.zeros(0))
        self.P = Phi @ self.P @ Phi.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: np.ndarray) -> None:
        """Joseph-form GPS measurement update."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        S = 0.5 * (S + S.T)
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self.x = self.x + K @ y
        joseph = np.eye(_ORBIT_STATE) - K @ self.H
        self.P = joseph @ self.P @ joseph.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


class AttitudeMEKF:
    """Multiplicative EKF over the 6-state attitude error ``[delta_theta(3), delta_bias(3)]``.

    The maintained estimate is the unit quaternion ``q`` (scalar-last, inertial->body) and the
    gyro bias ``b``; the error state is reset into ``q``/``b`` after every update. Ported from the
    legacy ``AttitudeEKF``: gyro propagation, normalized vector updates (sun/magnetometer) with
    ``H = [skew(pred), 0]``, an optional direct star-tracker attitude update, Joseph-form
    covariance and a left-multiplicative quaternion reset.
    """

    def __init__(  # noqa: PLR0913
        self,
        q0: np.ndarray,
        P0: np.ndarray,
        Qc: np.ndarray,
        R_sun: np.ndarray,
        R_mag: np.ndarray,
        R_star: np.ndarray | None = None,
        b0: np.ndarray | None = None,
    ) -> None:
        """Initialize the attitude/bias estimate and the process/measurement covariances."""
        self.q = q0 / np.linalg.norm(q0)
        self.b = np.zeros(3) if b0 is None else b0
        self.P = P0
        self.Qc = Qc
        self.R_sun = R_sun
        self.R_mag = R_mag
        self.R_star = np.eye(3) * 1e-4 if R_star is None else R_star

    @staticmethod
    def _continuous_fg(omega_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous error-state dynamics ``F`` and noise-input ``G`` matrices."""
        F = np.zeros((_ERROR_STATE, _ERROR_STATE))
        F[:3, :3] = -_skew(omega_eff)
        F[:3, 3:] = -np.eye(3)
        G = np.zeros((_ERROR_STATE, _ERROR_STATE))
        G[:3, :3] = -np.eye(3)
        G[3:, 3:] = np.eye(3)
        return F, G

    def predict(self, omega_meas: np.ndarray, dt: float) -> None:
        """Propagate the quaternion (bias-compensated gyro) and covariance over ``dt``."""
        if dt <= 0.0:
            return
        omega_eff = omega_meas - self.b
        dqdt = Quaternion.from_array(self.q).kinematics(omega_eff)
        self.q = self.q + dqdt * dt
        self.q /= np.linalg.norm(self.q)
        F, G = self._continuous_fg(omega_eff)
        Phi = np.eye(_ERROR_STATE) + F * dt
        Qd = G @ self.Qc @ G.T * dt
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)

    def _update(self, z_meas: np.ndarray, z_pred: np.ndarray, H: np.ndarray, R_meas: np.ndarray) -> None:
        """Apply a generic EKF correction with multiplicative quaternion reset (Joseph covariance)."""
        y = z_meas - z_pred
        S = H @ self.P @ H.T + R_meas
        S = 0.5 * (S + S.T) + _EPS * np.eye(3)
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        dx = K @ y
        self.b = self.b + dx[3:]
        dq = np.hstack((0.5 * dx[:3], [1.0]))
        dq /= np.linalg.norm(dq)
        self.q = (Quaternion.from_array(dq) * Quaternion.from_array(self.q)).to_array()
        self.q /= np.linalg.norm(self.q)
        joseph = np.eye(_ERROR_STATE) - K @ H
        self.P = joseph @ self.P @ joseph.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_vector(self, ref_eci: np.ndarray, body_meas: np.ndarray, R_meas: np.ndarray) -> None:
        """Apply a normalized line-of-sight update (sun sensor / magnetometer)."""
        pred = Quaternion.from_array(self.q).apply(ref_eci)
        n_meas = np.linalg.norm(body_meas)
        n_pred = np.linalg.norm(pred)
        if n_meas < _EPS or n_pred < _EPS:
            return
        H = np.hstack((_skew(pred / n_pred), np.zeros((3, 3))))
        self._update(body_meas / n_meas, pred / n_pred, H, R_meas)

    def update_attitude(self, q_meas: np.ndarray) -> None:
        """Direct star-tracker attitude update from a measured quaternion."""
        q_meas = q_meas / np.linalg.norm(q_meas)
        error = Quaternion.from_array(q_meas) * Quaternion.from_array(self.q).conjugate()
        delta_theta = 2.0 * error.vec * np.sign(error.scalar if error.scalar != 0.0 else 1.0)
        H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self._update(delta_theta, np.zeros(3), H, self.R_star)
