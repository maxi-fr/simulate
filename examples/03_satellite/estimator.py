# ruff: noqa: N806
"""Example-specific full state estimator for the satellite simulation.

Composes the orbit KF, attitude MEKF, and environment exposure into one estimator,
returning an estimate matching the ESTIMATE layout.
"""

import dataclasses
import datetime
from typing import Any, Self

import numpy as np

from simulate.estimator import Estimator
from spacecraft.environment import atmosphere_density_msis, is_in_shadow, magnetic_field_vector, sun_position
from spacecraft.estimator import AttitudeMEKF, MeasurementLayout, OrbitKalmanFilter
from spacecraft.frames import eci_attitude_from_lvlh, eci_to_geodedic
from spacecraft.orbit_dynamics import SGP4
from spacecraft.quaternion import Quaternion

_ERROR_STATE = 6  # MEKF error state: [delta_theta(3), delta_bias(3)]
_ORBIT_STATE = 6  # orbit state: [r(3), v(3)]


def _ensure_utc(epoch: datetime.datetime) -> datetime.datetime:
    """Return ``epoch`` as a timezone-aware UTC datetime (naive inputs are assumed UTC)."""
    return epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=datetime.UTC)


def _as_matrix(value: Any, n: int) -> np.ndarray:  # noqa: ANN401
    """Coerce a config entry to an ``(n, n)`` matrix, expanding a length-n vector to a diagonal."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == n:
        return np.diag(arr)
    return arr.reshape(n, n)


@dataclasses.dataclass
class FullStateEstimatorLog:
    """Internal log: gyro bias plus the environment exposed at the estimated orbit."""

    gyro_bias: np.ndarray
    b_field_body: np.ndarray
    sun_dir_body: np.ndarray
    density: float
    geodetic: np.ndarray
    orbit_cov_trace: float
    wheel_momentum: np.ndarray


class FullStateEstimator(Estimator[FullStateEstimatorLog]):
    """Composes the orbit KF, attitude MEKF and environment exposure into one estimator."""

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        epoch: datetime.datetime,
        layout: MeasurementLayout,
        orbit: OrbitKalmanFilter,
        attitude: AttitudeMEKF,
        rw_axes: np.ndarray | None = None,
        rw_inertia: np.ndarray | None = None,
        tach_channel: str = "tachometer",
    ) -> None:
        """Initialize with the sample time, epoch, channel layout and the two sub-filters.

        ``rw_axes`` (N, 3) and ``rw_inertia`` (N,) describe the reaction-wheel array used to turn
        the tachometer channel (relative wheel speeds) into the body-frame wheel angular momentum
        exposed in ``x_hat``; when omitted the exposed momentum is zero.
        """
        super().__init__(dt)
        self.epoch = _ensure_utc(epoch)
        self.layout = layout
        self.orbit = orbit
        self.attitude = attitude
        self.tach_channel = tach_channel
        if rw_axes is None:
            self.rw_axes: np.ndarray | None = None
            self.rw_inertia: np.ndarray | None = None
        else:
            axes = np.asarray(rw_axes, dtype=float)
            self.rw_axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
            self.rw_inertia = np.asarray(rw_inertia, dtype=float)
        self._t_prev: float | None = None
        self._last_channels: dict[str, np.ndarray] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary.

        The initial orbit/attitude guess is resolved from the shared ``initial_state`` block (a TLE
        plus an ORC-relative attitude, the same anchor the dynamics use) so the estimate starts
        consistent with the truth; the ``orbit``/``attitude`` blocks then carry only the filter
        covariances. The epoch used for environment exposure also comes from ``initial_state``.

        Returns
        -------
        Self
            The estimator configured from ``config``.
        """
        init = config["initial_state"]
        epoch = datetime.datetime.fromisoformat(init["epoch"])
        r0, v0 = SGP4.from_tle(*init["tle"]).propagate(epoch)
        att = init.get("attitude_lvlh", init.get("attitude_orc"))
        omega_bo = init.get("angular_velocity_lvlh", init.get("angular_velocity_orc"))
        q_bi, _ = eci_attitude_from_lvlh(
            r0,
            v0,
            roll=att["roll"],
            pitch=att["pitch"],
            yaw=att["yaw"],
            omega_bo=omega_bo,
        )

        layout = MeasurementLayout(channels=tuple((name, int(dim)) for name, dim in config["channels"]))
        gps_dim = dict(layout.channels).get("gps", 3)

        orbit_cfg = config["orbit"]
        if gps_dim == _ORBIT_STATE:
            H = np.eye(_ORBIT_STATE)
        else:
            H = np.zeros((3, _ORBIT_STATE))
            H[:, :3] = np.eye(3)
        orbit = OrbitKalmanFilter(
            r0=np.asarray(r0, dtype=float),
            v0=np.asarray(v0, dtype=float),
            P0=_as_matrix(orbit_cfg["P0"], _ORBIT_STATE),
            Q=_as_matrix(orbit_cfg["Q"], _ORBIT_STATE),
            H=H,
            R=_as_matrix(orbit_cfg["R"], gps_dim),
        )

        att_cfg = config["attitude"]
        attitude = AttitudeMEKF(
            q0=q_bi.to_array(),
            P0=_as_matrix(att_cfg["P0"], _ERROR_STATE),
            Qc=_as_matrix(att_cfg["Qc"], _ERROR_STATE),
            R_sun=_as_matrix(att_cfg["R_sun"], 3),
            R_mag=_as_matrix(att_cfg["R_mag"], 3),
            R_star=_as_matrix(att_cfg["R_star"], 3) if "R_star" in att_cfg else None,
            b0=np.asarray(att_cfg["b0"], dtype=float) if "b0" in att_cfg else None,
        )
        wheels_cfg = config.get("wheels")
        if wheels_cfg is None:
            rw_axes = rw_inertia = None
            tach_channel = "tachometer"
        else:
            rw_axes = np.asarray(wheels_cfg["axes"], dtype=float)
            rw_inertia = np.asarray(wheels_cfg["inertia"], dtype=float)
            tach_channel = str(wheels_cfg.get("tach_channel", "tachometer"))

        return cls(
            dt=float(config["dt"]),
            epoch=epoch,
            layout=layout,
            orbit=orbit,
            attitude=attitude,
            rw_axes=rw_axes,
            rw_inertia=rw_inertia,
            tach_channel=tach_channel,
        )

    def update(
        self,
        t: float,
        y_mea: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, FullStateEstimatorLog]:
        """Run both sub-filters on the split measurements and assemble ``[r, v, q, omega]``."""
        # The simulation seeds the first measurement from scalar zeros before any Output has produced
        # real truth, so y_mea is undersized on the warm-up step; skip the updates (predict only) and
        # emit the current best estimate until a full-width measurement arrives.
        y = np.atleast_1d(y_mea)
        channels = self.layout.split(y) if y.size == self.layout.size else {}
        dt = 0.0 if self._t_prev is None else t - self._t_prev
        self._t_prev = t
        dt_utc = self.epoch + datetime.timedelta(seconds=t)

        # Slow sensors are zero-order-hold-held by the simulation between samples, so a held value
        # is byte-identical to the one already fused; only fuse a channel when it is a fresh sample
        # (re-fusing a stale measurement would spuriously pin the estimate and shrink the covariance).
        def fresh(name: str) -> bool:
            if name not in channels:
                return False
            prev = self._last_channels.get(name)
            return prev is None or not np.array_equal(prev, channels[name])

        # Orbit Kalman filter.
        self.orbit.predict(dt)
        if fresh("gps"):
            self.orbit.update(channels["gps"])
        r_est = self.orbit.x[:3]
        v_est = self.orbit.x[3:]

        # Attitude MEKF (the gyro drives the prediction at the base rate, so it is always fresh).
        omega_meas = channels.get("gyro", np.zeros(3))
        self.attitude.predict(omega_meas, dt)
        if fresh("magnetometer"):
            dt_naive = dt_utc.replace(tzinfo=None)
            lat, lon, alt = eci_to_geodedic(r_est)
            b_ref = magnetic_field_vector(dt_naive, float(lat), float(lon), float(alt))
            self.attitude.update_vector(b_ref, channels["magnetometer"], self.attitude.R_mag)
        if fresh("sun"):
            self.attitude.update_vector(sun_position(dt_utc) - r_est, channels["sun"], self.attitude.R_sun)
        if fresh("star_tracker"):
            self.attitude.update_attitude(channels["star_tracker"])

        self._last_channels = {name: np.asarray(val).copy() for name, val in channels.items()}

        q_est = Quaternion.from_array(self.attitude.q)
        omega_est = omega_meas - self.attitude.b

        h_wheel = self._wheel_momentum(channels, omega_est)
        log = self._expose_environment(r_est, q_est, dt_utc, h_wheel)
        x_hat = np.concatenate([r_est, v_est, q_est.to_array(), omega_est, log.b_field_body, h_wheel])
        return x_hat, log

    def _wheel_momentum(self, channels: dict[str, np.ndarray], omega_est: np.ndarray) -> np.ndarray:
        """Body-frame reaction-wheel angular momentum from the tachometer channel (zero if absent).

        The tachometer reports relative wheel speeds ``omega_rel``; the stored momentum mirrors the
        :class:`~spacecraft.effector.ReactionWheelArray` contribution
        ``axes^T @ (J_w * (omega_rel + axes @ omega_body))``.

        Returns
        -------
        np.ndarray
            Body-frame reaction-wheel angular momentum, shape ``(3,)`` (zeros if no tachometer).
        """
        if self.rw_axes is None or self.rw_inertia is None or self.tach_channel not in channels:
            return np.zeros(3)
        omega_rel = channels[self.tach_channel]
        omega_abs = omega_rel + self.rw_axes @ omega_est
        return self.rw_axes.T @ (self.rw_inertia * omega_abs)

    def _expose_environment(
        self,
        r_est: np.ndarray,
        q_est: Quaternion,
        dt_utc: datetime.datetime,
        wheel_momentum: np.ndarray,
    ) -> FullStateEstimatorLog:
        """Evaluate the environment at the estimated orbit and pack it into the log."""
        lat, lon, alt = eci_to_geodedic(r_est)
        dt_naive = dt_utc.replace(tzinfo=None)
        b_field_body = q_est.apply(magnetic_field_vector(dt_naive, float(lat), float(lon), float(alt)))

        sun_eci = sun_position(dt_utc)
        if is_in_shadow(r_est, sun_eci):
            sun_dir_body = np.zeros(3)
        else:
            sc_to_sun = sun_eci - r_est
            sun_dir_body = q_est.apply(sc_to_sun / np.linalg.norm(sc_to_sun))

        density = atmosphere_density_msis(dt_utc, float(lat), float(lon), float(alt))
        return FullStateEstimatorLog(
            gyro_bias=self.attitude.b.copy(),
            b_field_body=b_field_body,
            sun_dir_body=sun_dir_body,
            density=float(density),
            geodetic=np.array([float(lat), float(lon), float(alt)]),
            orbit_cov_trace=float(np.trace(self.orbit.P)),
            wheel_momentum=wheel_momentum,
        )
