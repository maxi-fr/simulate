import datetime
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import simulation.environment as env
from simulation.kinematics import eci_to_geodedic
from utils import Logger, Quaternion


def skew(v: np.ndarray) -> np.ndarray:
    """
    Computes the skew-symmetric matrix of a 3D vector.

    Parameters
    ----------
    v : np.ndarray
        Input 3D vector.

    Returns
    -------
    np.ndarray
        3x3 skew-symmetric matrix.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class AttitudeEstimator(ABC):
    """
    Abstract base class for attitude estimators.
    """

    def __init__(self) -> None:
        """
        Initializes the AttitudeEstimator.
        """
        self.logger: Logger | None = None
        self.header: list[str] = []

    def init_log(self, log_folder: str) -> None:
        """
        Initializes the logger.

        Parameters
        ----------
        log_folder : str
            Path to the log folder.
        """
        if self.header:
            self.logger = Logger(os.path.join(log_folder, "estimation.csv"), self.header)

    def close_log(self) -> None:
        """
        Closes the logger to ensure data is written to disk.
        """
        if self.logger is not None:
            self.logger.close()

    @abstractmethod
    def predict(self, t: datetime.datetime, omega_meas: np.ndarray) -> None:
        """
        Performs the prediction step of the estimator.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        omega_meas : np.ndarray
            Measured angular velocity [rad/s].
        """

    @abstractmethod
    def update_sun(self, t: datetime.datetime, sun_body_meas: np.ndarray) -> None:
        """
        Performs the measurement update using the sun sensor.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        sun_body_meas : np.ndarray
            Measured sun vector in the body frame.
        """

    @abstractmethod
    def update_mag(self, t: datetime.datetime, B_body_meas: np.ndarray, r_ECI: np.ndarray) -> None:
        """
        Performs the measurement update using the magnetometer.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        B_body_meas : np.ndarray
            Measured magnetic field vector in the body frame.
        r_ECI : np.ndarray
            Satellite position in ECI frame [m].
        """

    @abstractmethod
    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the estimated attitude state.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (q_est, omega_est) where q_est is [x, y, z, w] and omega_est is [x, y, z]
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Converts the estimator configuration to a dictionary.

        Returns
        -------
        dict
            Dictionary configuration.
        """


class AttitudeEKF(AttitudeEstimator):
    """
    Multiplicative Extended Kalman Filter (MEKF) for attitude estimation.
    """

    # Filtering the orientation of the satellite + base drift of gyro
    def __init__(
        self,
        q0: np.ndarray,
        P0: np.ndarray | list[float],
        Qc: np.ndarray | list[float],
        R_sun: np.ndarray | list[float],
        R_mag: np.ndarray | list[float],
        b0: np.ndarray | list[float] | None = None,
    ) -> None:
        """
        Initializes the AttitudeEKF.

        Parameters
        ----------
        q0 : np.ndarray
            Initial quaternion guess [x, y, z, w].
        P0 : Union[np.ndarray, List[float]]
            Initial covariance matrix (6x6).
        Qc : Union[np.ndarray, List[float]]
            Process noise spectral density (6x6).
        R_sun : Union[np.ndarray, List[float]]
            Sun sensor measurement covariance (3x3).
        R_mag : Union[np.ndarray, List[float]]
            Magnetometer measurement covariance (3x3).
        b0 : Optional[Union[np.ndarray, List[float]]], optional
            Initial bias guess, by default None.
        """
        super().__init__()

        # making sure it's a 4-element vector
        q0 = np.asarray(q0, dtype=float).reshape(4)
        # quaternion normalization (to represent rotation)
        self.q = q0 / np.linalg.norm(q0)

        if b0 is None:
            self.b = np.zeros(3)
        else:
            self.b = np.asarray(b0, dtype=float).reshape(3)  # gyro bias

        self.last_omega_meas = np.zeros(3)

        # Helper to convert list/array to array
        def to_array(x: np.ndarray | list[float], shape: tuple[int, ...]) -> np.ndarray:
            """
            Helper to convert input to numpy array of specific shape.

            Parameters
            ----------
            x : np.ndarray | list
                Input array or list.
            shape : Tuple[int, ...]
                Target shape.

            Returns
            -------
            np.ndarray
                Reshaped array.
            """
            a = np.asarray(x, dtype=float)
            if a.ndim == 1 and len(shape) == 2 and shape[0] == shape[1] and len(a) == shape[0]:
                return np.diag(a)
            return a.reshape(shape)

        self.P = to_array(P0, (6, 6))
        self.Qc = to_array(Qc, (6, 6))
        self.R_sun = to_array(R_sun, (3, 3))
        self.R_mag = to_array(R_mag, (3, 3))

        # saving the time of last prediction
        self.t_last: datetime.datetime | None = None

        self.header = [
            "t",
            "q_est_x",
            "q_est_y",
            "q_est_z",
            "q_est_w",
            "b_est_x",
            "b_est_y",
            "b_est_z",
            "omega_est_x",
            "omega_est_y",
            "omega_est_z",
            "P_diag_0",
            "P_diag_1",
            "P_diag_2",
            "P_diag_3",
            "P_diag_4",
            "P_diag_5",
        ]

    # prediction step
    def predict(self, t: datetime.datetime, omega_meas: np.ndarray) -> None:
        """
        Propagates the state estimate using gyro measurements.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        omega_meas : np.ndarray
            Measured angular velocity [rad/s].
        """
        # Time update using gyro measurement omega_meas (rad/s)
        self.last_omega_meas = np.asarray(omega_meas, dtype=float).reshape(3)

        if self.t_last is None:
            # initialise time reference but not propagating
            self.t_last = t
            return
        # calculating the discrete time step
        dt = (t - self.t_last).total_seconds()

        if dt <= 0.0:
            # non-positive time step: ignore to avoid singular behaviour
            self.t_last = t
            return
        self.t_last = t

        # bias-compensated angular rate
        omega_eff = self.last_omega_meas - self.b

        # propagate quaternion (nominal attitude)
        dqdt = Quaternion.from_array(self.q).kinematics(omega_eff)
        self.q = self.q + dqdt * dt
        self.q /= np.linalg.norm(self.q)

        # getting the continuous F and G matrices for our error model (omega_eff)
        F, G = self._continuous_FG(omega_eff)
        Phi = np.eye(6) + F * dt  # state transition approx
        Qd = G @ self.Qc @ G.T * dt  # discrete process noise (Euler approx)
        # EKF porpagation
        self.P = Phi @ self.P @ Phi.T + Qd
        # enforce symmetry numerically
        self.P = 0.5 * (self.P + self.P.T)

    # Construct continuous-time F and G for the 6-state error model
    def _continuous_FG(self, omega_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs continuous-time Jacobian matrices F and G.

        Parameters
        ----------
        omega_eff : np.ndarray
            Effective angular velocity (bias corrected).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            F (6x6) and G (6x6) matrices.
        """
        omega_eff = np.asarray(omega_eff, dtype=float).reshape(3)
        # input being effective angle velocity
        F11 = -skew(omega_eff)
        F12 = -np.eye(3)
        F = np.block(
            [
                [F11, F12],
                [np.zeros((3, 3)), np.zeros((3, 3))],
            ]
        )

        G = np.block(
            [
                [-np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.eye(3)],
            ]
        )

        return F, G

    # GENERIC MEASUREMENT UPDATE
    def _update(
        self,
        z_meas: np.ndarray,
        z_pred: np.ndarray,
        H: np.ndarray,
        R_meas: np.ndarray,
    ) -> None:
        """
        Generic EKF measurement update step.

        Parameters
        ----------
        z_meas : np.ndarray
            Measurement vector.
        z_pred : np.ndarray
            Predicted measurement vector.
        H : np.ndarray
            Measurement Jacobian matrix.
        R_meas : np.ndarray
            Measurement noise covariance.
        """
        # Generic EKF measurement update for a 3D vector measurement
        z_meas = np.asarray(z_meas, dtype=float).reshape(3)
        z_pred = np.asarray(z_pred, dtype=float).reshape(3)
        H = np.asarray(H, dtype=float).reshape(3, 6)
        R_meas = np.asarray(R_meas, dtype=float).reshape(3, 3)
        # innovation
        y = z_meas - z_pred
        # innovation covariance
        S = H @ self.P @ H.T + R_meas
        # Kalman Gain
        S = 0.5 * (S + S.T)  # symetry (numerics)
        S = S + 1e-12 * np.eye(3)  # jitter just in case
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        # error-state update
        dx = K @ y
        dtheta = dx[:3]
        db = dx[3:]
        # update bias
        self.b = self.b + db
        # update quaternion using small-angle rotation
        dq_vec = 0.5 * dtheta
        dq = np.hstack((dq_vec, [1.0]))
        dq /= np.linalg.norm(dq)
        # converting existing quaternion and correction into rotation
        q_rot = Quaternion.from_array(self.q)
        dq_rot = Quaternion.from_array(dq)
        self.q = (dq_rot * q_rot).to_array()
        self.q /= np.linalg.norm(self.q)
        # Joseph-form covariance update for better numerical stability
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # SUN SENSOR UPDATE
    def update_sun(self, t: datetime.datetime, sun_body_meas: np.ndarray) -> None:
        """
        Updates the state estimate with sun sensor measurement.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        sun_body_meas : np.ndarray
            Measured sun vector in body frame.
        """
        sun_body_meas = np.asarray(sun_body_meas, dtype=float).reshape(3)
        r_sun_I = env.sun_position(t)
        s_I = r_sun_I / np.linalg.norm(r_sun_I)
        C_BI = Quaternion.from_array(self.q)
        s_B_pred = C_BI.apply(s_I)
        n_meas = np.linalg.norm(sun_body_meas)
        n_pred = np.linalg.norm(s_B_pred)
        if n_meas < 1e-12 or n_pred < 1e-12:
            return
        sun_body_meas = sun_body_meas / n_meas
        s_B_pred = s_B_pred / n_pred
        H = np.hstack((skew(s_B_pred), np.zeros((3, 3))))
        self._update(sun_body_meas, s_B_pred, H, self.R_sun)

    # update from the magnometer
    def update_mag(self, t: datetime.datetime, B_body_meas: np.ndarray, r_ECI: np.ndarray) -> None:
        """
        Updates the state estimate with magnetometer measurement.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        B_body_meas : np.ndarray
            Measured magnetic field in body frame.
        r_ECI : np.ndarray
            Satellite position in ECI.
        """
        B_body_meas = np.asarray(B_body_meas, dtype=float).reshape(3)
        r_ECI = np.asarray(r_ECI, dtype=float).reshape(3)
        lat, lon, alt = eci_to_geodedic(r_ECI)
        B_I = env.magnetic_field_vector(t, lat, lon, alt)
        C_BI = Quaternion.from_array(self.q)
        B_B_pred = C_BI.apply(B_I)
        n_meas = np.linalg.norm(B_body_meas)
        n_pred = np.linalg.norm(B_B_pred)
        if n_meas < 1e-12 or n_pred < 1e-12:
            return
        B_body_meas = B_body_meas / n_meas
        B_B_pred = B_B_pred / n_pred
        H = np.hstack((skew(B_B_pred), np.zeros((3, 3))))
        self._update(B_body_meas, B_B_pred, H, self.R_mag)

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the estimated attitude state.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Estimated quaternion [x,y,z,w] and angular velocity [rad/s].
        """
        omega_est = self.last_omega_meas - self.b

        if self.logger is not None and self.t_last is not None:
            self.logger.log(
                [self.t_last, *self.q.tolist(), *self.b.tolist(), *omega_est.tolist(), *np.diag(self.P).tolist()]
            )

        return self.q.copy(), omega_est

    def to_dict(self) -> dict[str, Any]:
        """
        Converts configuration to dictionary.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return {
            "P0": np.diag(self.P).tolist(),
            "Qc": np.diag(self.Qc).tolist(),
            "R_sun": np.diag(self.R_sun).tolist(),
            "R_mag": np.diag(self.R_mag).tolist(),
        }
