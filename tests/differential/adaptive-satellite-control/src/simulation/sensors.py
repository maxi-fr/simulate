# TODO
import datetime

# TODO: Refactor: only need 1 generic gaussian random walk bias + gaussian noise class. bias noise = 0 for sensor like SunSensor
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from utils import Logger


class Sensor(ABC):
    """
    Abstract base class for sensors.
    """

    def __init__(self, frequency: float) -> None:
        """
        Initializes the Sensor.

        Parameters
        ----------
        frequency : float
            Sampling frequency [Hz].
        """
        self.rng = np.random.default_rng()
        self.period = datetime.timedelta(seconds=1.0 / frequency)
        self.last_measurement: datetime.datetime | None = None
        self.logger: Logger | None = None
        self.header: list[str] = []

    def init_log(self, log_folder: str, name: str) -> None:
        """
        Initializes the logger.

        Parameters
        ----------
        log_folder : str
            Path to the folder where the log file will be saved.
        name : str
            Name of the logger file.
        """
        if self.header:
            self.logger = Logger(os.path.join(log_folder, f"{name}.csv"), self.header)

    def close_log(self) -> None:
        """
        Closes the logger to ensure data is written to disk.
        """
        if self.logger is not None:
            self.logger.close()

    @abstractmethod
    def measure(self, t: datetime.datetime, state_dict: dict[str, Any], env_dict: dict[str, Any]) -> None:
        """
        Updates the internal state of the sensor based on the current truth data.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        state_dict : Dict[str, Any]
            Dictionary of the current state variables.
        env_dict : Dict[str, Any]
            Dictionary containing environment variables.
        """

    @abstractmethod
    def read(self, t: datetime.datetime) -> tuple[Any, bool]:
        """
        Reads the sensor value.

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        Tuple[Any, bool]
            The measured value and a boolean indicating if a new measurement is available.
        """

    def to_dict(self) -> dict[str, Any]:
        """
        Converts sensor parameters to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sensor parameters.
        """
        return {"frequency": 1.0 / self.period.total_seconds()}


class Gyroscope(Sensor):
    """
    Gyroscope sensor model.
    """

    def __init__(self, frequency: float, sigma_sq: float, bias_sigma_sq: float) -> None:
        """
        Initializes the Gyroscope.

        Parameters
        ----------
        frequency : float
            Sampling frequency [Hz].
        sigma_sq : float
            Variance of the measurement noise.
        bias_sigma_sq : float
            Variance of the bias random walk.
        """
        super().__init__(frequency)
        self.omega = np.zeros(3)
        self.bias = np.zeros(3)
        self.bias_sigma_sq = bias_sigma_sq
        self.sigma_sq = sigma_sq
        self.header = ["t", "omega_x", "omega_y", "omega_z"]

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the Gyroscope configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameters.
        """
        data = super().to_dict()
        data["sigma_sq"] = self.sigma_sq
        data["bias_sigma_sq"] = self.bias_sigma_sq
        return data

    def measure(self, t: datetime.datetime, state_dict: dict[str, Any], env_dict: dict[str, Any]) -> None:
        """
        Updates the gyroscope measurement.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        state_dict : Dict[str, Any]
            Dictionary of the current state variables.
        env_dict : Dict[str, Any]
            Dictionary containing environment variables.
        """
        omega = state_dict["omega"]
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.UTC)

        if self.last_measurement is None:
            self.last_measurement = t
            self.omega = omega.copy()  # alebo omega + maly šum podľa period, nie podľa dt_s=nekonečno
            if self.logger is not None:
                self.logger.log([t, *self.omega.tolist()])
            return
        dt = t - self.last_measurement
        if dt >= self.period:
            self.last_measurement = t
            dt_s = dt.total_seconds()

            bias = self.bias + np.sqrt(self.bias_sigma_sq) * np.sqrt(dt_s) * self.rng.normal(0, 1, 3)
            self.omega = (
                omega
                + 0.5 * (bias + self.bias)
                + ((self.sigma_sq / dt_s + 1 / 12 * self.bias_sigma_sq * dt_s) ** 0.5) * self.rng.normal(0, 1, 3)
            )
            self.bias = bias
            if self.logger is not None:
                self.logger.log([t, *self.omega.tolist()])

    def read(self, t: datetime.datetime) -> tuple[np.ndarray, bool]:
        """
        Returns last measured value for the gyroscope.

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        Tuple[np.ndarray, bool]
            Angular velocity [rad/s] and validity flag.
        """
        return self.omega, self.last_measurement == t


class Magnetometer(Sensor):
    """
    Magnetometer sensor model.
    """

    def __init__(self, frequency: float, sigma_sq: float, const_bias: np.ndarray) -> None:
        """
        Initializes the Magnetometer.

        Parameters
        ----------
        frequency : float
            Sampling frequency [Hz].
        sigma_sq : float
            Variance of the measurement noise.
        const_bias : np.ndarray
            Constant bias vector [T].

        """
        super().__init__(frequency)

        self.sigma_sq = sigma_sq
        self.const_bias = const_bias
        self.B = np.zeros(3)
        self.header = ["t", "B_x", "B_y", "B_z"]

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the Magnetometer configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameters.
        """
        data = super().to_dict()
        data["sigma_sq"] = self.sigma_sq
        data["const_bias"] = self.const_bias.tolist() if isinstance(self.const_bias, np.ndarray) else self.const_bias
        return data

    def measure(self, t: datetime.datetime, state_dict: dict[str, Any], env_dict: dict[str, Any]) -> None:
        """
        Updates the magnetometer measurement.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        state_dict : Dict[str, Any]
            Dictionary of the current state variables.
        env_dict : Dict[str, Any]
            Dictionary containing environment variables. Expects 'B_body'.
        """
        B_body = env_dict["B_body"]
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.UTC)
        if self.last_measurement is None:
            self.last_measurement = t
            self.B = self.const_bias + B_body + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, 3)
            if self.logger is not None:
                self.logger.log([t, *self.B.tolist()])
            return
        dt = t - self.last_measurement

        if dt >= self.period:
            self.last_measurement = t
            self.B = self.const_bias + B_body + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, 3)
            if self.logger is not None:
                self.logger.log([t, *self.B.tolist()])

    def read(self, t: datetime.datetime) -> tuple[np.ndarray, bool]:
        """
        Returns last measured value for the magnetometer.

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        Tuple[np.ndarray, bool]
            Magnetic field vector [T] and validity flag.
        """
        return self.B, self.last_measurement == t


class SunSensor(Sensor):
    """
    Sun sensor model.
    """

    def __init__(self, frequency: float, sigma_sq: float) -> None:
        """
        Initializes the SunSensor.

        Parameters
        ----------
        frequency : float
            Sampling frequency [Hz].
        sigma_sq : float
            Variance of the measurement noise.
        """
        super().__init__(frequency)

        self.sigma_sq = sigma_sq
        self.sun_pos = np.zeros(3)
        self.header = ["t", "sun_x", "sun_y", "sun_z"]

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the SunSensor configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameters.
        """
        data = super().to_dict()
        data["sigma_sq"] = self.sigma_sq
        return data

    def measure(self, t: datetime.datetime, state_dict: dict[str, Any], env_dict: dict[str, Any]) -> None:
        """
        Updates the sun sensor measurement.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        state_dict : Dict[str, Any]
            Dictionary of the current state variables.
        env_dict : Dict[str, Any]
            Dictionary containing environment variables. Expects 's_B' and 'in_shadow'.
        """
        if env_dict["in_shadow"]:
            return

        sun_pos = env_dict["s_B"]

        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.UTC)

        if self.last_measurement is None:
            self.last_measurement = t
            self.sun_pos = sun_pos + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, 3)
            if self.logger is not None:
                self.logger.log([t, *self.sun_pos.tolist()])
            return

        dt = t - self.last_measurement

        if dt >= self.period:
            self.last_measurement = t
            self.sun_pos = sun_pos + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, 3)
            if self.logger is not None:
                self.logger.log([t, *self.sun_pos.tolist()])

    def read(self, t: datetime.datetime) -> tuple[np.ndarray, bool]:
        """
        Reads the sun position.

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        Tuple[np.ndarray, bool]
            Sun position vector and validity flag.
        """
        return self.sun_pos, self.last_measurement == t


class GPS(Sensor):
    """
    GPS sensor model.
    """

    def __init__(self, frequency: float, sigma_sq: float) -> None:
        """
        Initializes the GPS.

        Parameters
        ----------
        frequency : float
            Sampling frequency [Hz].
        sigma_sq : float
            Variance of the measurement noise.
        """
        super().__init__(frequency)

        self.sigma_sq = sigma_sq
        self.sat_pos = np.zeros(3)
        self.header = ["t", "sat_pos_x", "sat_pos_y", "sat_pos_z"]

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the GPS configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameters.
        """
        data = super().to_dict()
        data["sigma_sq"] = self.sigma_sq
        return data

    def measure(self, t: datetime.datetime, state_dict: dict[str, Any], env_dict: dict[str, Any]) -> None:
        """
        Updates the GPS measurement.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        state_dict : Dict[str, Any]
            Dictionary of the current state variables.
        env_dict : Dict[str, Any]
            Dictionary containing environment variables.
        """
        sat_pos = state_dict["r_eci"]
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.UTC)

        if self.last_measurement is None:
            self.last_measurement = t
            self.sat_pos = sat_pos + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, 3)
            if self.logger is not None:
                self.logger.log([t, *self.sat_pos.tolist()])
            return

        dt = t - self.last_measurement

        if dt >= self.period:
            self.last_measurement = t
            self.sat_pos = sat_pos + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, 3)
            if self.logger is not None:
                self.logger.log([t, *self.sat_pos.tolist()])

    def read(self, t: datetime.datetime) -> tuple[np.ndarray, bool]:
        """
        Reads the GPS position.

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        Tuple[np.ndarray, bool]
             Position vector [m] and validity flag.
        """
        return self.sat_pos, self.last_measurement == t


class RW_tachometer(Sensor):
    """
    Reaction Wheel tachometer model.
    """

    def __init__(self, frequency: float, sigma_sq: float) -> None:
        """
        Initializes the tachometer.

        Parameters
        ----------
        frequency : float
            Sampling frequency [Hz].
        sigma_sq : float
            Variance of the measurement noise.
        """
        super().__init__(frequency)

        self.sigma_sq = sigma_sq
        self.omega = np.array([])
        # Assume up to 6 reaction wheels for the header, though we log whatever is present.
        self.header = ["t", "omega_1", "omega_2", "omega_3", "omega_4", "omega_5", "omega_6"]

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the RW_tachometer configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameters.
        """
        data = super().to_dict()
        data["sigma_sq"] = self.sigma_sq
        return data

    def measure(self, t: datetime.datetime, state_dict: dict[str, Any], env_dict: dict[str, Any]) -> None:
        """
        Updates the tachometer measurement for all wheels.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        state_dict : Dict[str, Any]
            Dictionary of the current state variables.
        env_dict : Dict[str, Any]
            Dictionary containing environment variables.
        """
        # Reaction wheel states are stored under state_dict["Actuators"]["ReactionWheel"]
        # Each RW state is an array, we extract the first element (speed) for each wheel.
        rw_states = state_dict["Actuators"]["ReactionWheel"]
        omega_rw = np.array([state[0] for state in rw_states if len(state) > 0])

        if len(omega_rw) == 0:
            return

        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.UTC)

        if self.last_measurement is None:
            self.last_measurement = t
            self.omega = omega_rw + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, len(omega_rw))
            if self.logger is not None:
                log_data = self.omega.tolist()
                log_data += [0.0] * (6 - len(log_data)) if len(log_data) < 6 else []
                self.logger.log([t] + log_data[:6])
            return

        dt = t - self.last_measurement
        if dt >= self.period:
            self.last_measurement = t
            self.omega = omega_rw + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, len(omega_rw))
            if self.logger is not None:
                log_data = self.omega.tolist()
                log_data += [0.0] * (6 - len(log_data)) if len(log_data) < 6 else []
                self.logger.log([t] + log_data[:6])

    def read(self, t: datetime.datetime) -> tuple[np.ndarray, bool]:
        """
        Reads the wheel speeds.

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        Tuple[np.ndarray, bool]
            Angular velocities [rad/s] and validity flag.
        """
        return self.omega, self.last_measurement == t
