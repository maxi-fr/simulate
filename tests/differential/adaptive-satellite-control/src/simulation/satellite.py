import datetime
from typing import Any

import numpy as np

from . import actuators as act_module
from . import sensors as sen_module
from .actuators import Actuator
from .dynamics import orbit_dynamics
from .sensors import Sensor
from .surface import Surface

"""
The idea of these classes is that they hold the parameters and provide wrappers for the dynamics functions
but they do not contain the state. The state is managed externally.
(TODO: maybe doesnt have to be a wrapper, they can just implement the dynamics)
"""


class Spacecraft:
    """
    A class representing a satellite, holding its physical parameters, sensors, and actuators.

    This class acts as a container for the satellite's configuration and provides wrappers
    for dynamics functions using the satellite's physical properties.
    """

    def __init__(
        self, m: float, J_B: np.ndarray, surfaces: list[Surface], actuators: list[Actuator], sensors: list[Sensor]
    ) -> None:
        """
        Initialize the Spacecraft object.

        Parameters
        ----------
        m : float
            Mass of the satellite [kg].
        J_B : np.ndarray
            Inertia tensor of the satellite in the body frame [kg*m^2].
        surfaces : List[Surface]
            List of surface elements defining the satellite geometry.
        actuators : List[Actuator]
            List of all satellite actuators.
        sensors : List[Sensor]
            List of all satellite sensors.
        """
        self.m = m

        self.J_B = J_B

        self.sensors = sensors

        self.actuators = actuators

        self.surfaces = surfaces

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Spacecraft":
        """
        Creates a Spacecraft instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing spacecraft configuration.

        Returns
        -------
        Spacecraft
            The created Spacecraft instance.
        """
        m = data["Mass"]
        J_B = np.array(data["Inertia"])

        surfaces = [Surface.from_dict(n, s) for n, s in data["Surfaces"].items()]

        actuators_list = []
        if "Actuators" in data:
            for class_name, act_configs in data["Actuators"].items():
                if hasattr(act_module, class_name):
                    act_cls = getattr(act_module, class_name)
                    if not isinstance(act_configs, list):
                        act_configs = [act_configs]
                    for config in act_configs:
                        actuators_list.append(act_cls(**config))

        sensors_list = []
        if "Sensors" in data:
            for class_name, sen_configs in data["Sensors"].items():
                if hasattr(sen_module, class_name):
                    sen_cls = getattr(sen_module, class_name)
                    if not isinstance(sen_configs, list):
                        sen_configs = [sen_configs]
                    for config in sen_configs:
                        config_copy = config.copy()
                        sensors_list.append(sen_cls(**config_copy))

        return cls(m, J_B, surfaces, actuators_list, sensors_list)

    def init_log(self, log_folder: str) -> None:
        """
        Initializes the loggers for all sensors and actuators.

        Parameters
        ----------
        log_folder : str
            Path to the log folder.
        """
        for sen in self.sensors:
            sen.init_log(log_folder, sen.__class__.__name__)
        for idx, act in enumerate(self.actuators):
            act.init_log(log_folder, f"{act.__class__.__name__}_{idx}")

    def close_log(self) -> None:
        """
        Closes all loggers.
        """
        for sen in self.sensors:
            sen.close_log()
        for act in self.actuators:
            act.close_log()

    def log_actuators(self, t: datetime.datetime, x: np.ndarray, u: np.ndarray, env_dict: dict[str, Any]) -> None:
        """
        Logs the state and inputs of all actuators.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        x : np.ndarray
            The full state vector.
        u : np.ndarray
            The full input vector.
        env_dict : Dict[str, Any]
            Dictionary containing environment variables.
        """
        x_actuators = []
        start_x = 13
        for act in self.actuators:
            end_x = start_x + act.n_states
            x_actuators.append(x[start_x:end_x])
            start_x = end_x

        u_split = []
        start_u = 0
        for act in self.actuators:
            end_u = start_u + act.n_inputs
            u_split.append(u[start_u:end_u])
            start_u = end_u

        for i, act in enumerate(self.actuators):
            act.log(t, x_actuators[i], u_split[i], env_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the Spacecraft instance to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing spacecraft configuration.
        """
        data = {
            "Mass": self.m,
            "Inertia": self.J_B.tolist(),
            "Surfaces": {s.name: s.to_dict() for s in self.surfaces},
            "Sensors": {},
            "Actuators": {},
        }

        if self.sensors:
            for sen in self.sensors:
                class_name = sen.__class__.__name__
                if class_name not in data["Sensors"]:
                    data["Sensors"][class_name] = []
                data["Sensors"][class_name].append(sen.to_dict())

            for class_name, sen_configs in data["Sensors"].items():
                if len(sen_configs) == 1:
                    data["Sensors"][class_name] = sen_configs[0]

        if self.actuators:
            for act in self.actuators:
                class_name = act.__class__.__name__
                if class_name not in data["Actuators"]:
                    data["Actuators"][class_name] = []
                data["Actuators"][class_name].append(act.to_dict())

        return data

    def measure_sensors(self, t: "datetime.datetime", state_dict: dict[str, Any], env_dict: dict[str, Any]) -> None:
        """
        Updates the internal state of all sensors based on the current truth data.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        state_dict : Dict[str, Any]
            Dictionary containing views of the current state.
        env_dict : Dict[str, Any]
            Dictionary containing environment variables.
        """
        for sen in self.sensors:
            sen.measure(t, state_dict, env_dict)

    def state_to_dict(self, x: np.ndarray) -> dict[str, Any]:
        """
        Converts a flat state array to a structured dictionary of state views.

        Parameters
        ----------
        x : np.ndarray
            The flat state array [r(3), v(3), q(4), w(3), ...actuator states...].

        Returns
        -------
        Dict[str, Any]
            A dictionary containing sliced views of the state vector.
        """
        state_dict = {"r_eci": x[0:3], "v_eci": x[3:6], "q_BI": x[6:10], "omega": x[10:13], "Actuators": {}}

        start = 13
        for act in self.actuators:
            class_name = act.__class__.__name__
            if class_name not in state_dict["Actuators"]:  # type: ignore[operator]
                state_dict["Actuators"][class_name] = []  # type: ignore[index]

            end = start + act.n_states
            # Append the state slice for this specific actuator
            state_dict["Actuators"][class_name].append(x[start:end])  # type: ignore[index]
            start = end

        return state_dict

    def read_sensors(self, t: datetime.datetime) -> dict[str, Any]:
        """
        Reads the values from all sensors.

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing measured values from all sensors. The structure is:
            {"SensorClassName": [(measured_value, new_measurement_flag), ...]}
        """
        readings: dict[str, Any] = {}
        for sen in self.sensors:
            class_name = sen.__class__.__name__
            if class_name not in readings:
                readings[class_name] = []
            readings[class_name].append(sen.read(t))

        return readings

    def orbit_dynamics(self, r_eci: np.ndarray, ext_force: np.ndarray) -> np.ndarray:
        """
        Computes the orbital dynamics (acceleration) of the spacecraft.

        Parameters
        ----------
        r_eci : np.ndarray
            Position vector in the ECI frame [m].
        ext_force : np.ndarray
            External force vector in the ECI frame [N].

        Returns
        -------
        np.ndarray
            Acceleration vector in the ECI frame [m/s^2].
        """
        return orbit_dynamics(self.m, r_eci, ext_force)
