import os
from typing import Any

import numpy as np
from utils import Logger


class Actuator:
    """
    Base class for satellite actuators.
    """

    def __init__(self) -> None:
        self.n_states = 0
        self.n_inputs = 0
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

    def log(self, t, x_act, u, env) -> None:  # type: ignore[no-untyped-def]
        """
        Logs the actuator state.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        x_act : Any
            State of the actuator.
        u : Any
            Control input.
        env : dict
            Environment variables.
        """

    def to_dict(self) -> dict[str, Any]:
        return {}

    def __call__(self, idx, x_act, x, u, env) -> Any:  # type: ignore[no-untyped-def]
        """
        (example for just one actuator)
        |J      lhs_21||omega_dot| = |rhs_1|
        |lhs_12 lhs_22||x_act_dot| = |rhs_2|

        Parameters
        ----------
        idx : int
            Index of the actuator.
        x_act : np.ndarray
            State of the actuator.
        x : np.ndarray
            State of the satellite.
        u : np.ndarray
            Control input.
        env : dict
            Environment variables.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        n = self.n_states

        lhs_12 = np.zeros((3, n), dtype=float)
        lhs_21 = np.zeros((n, 3), dtype=float)
        lhs_22 = np.zeros((n, n), dtype=float)
        rhs_1 = np.zeros(3, dtype=float)
        rhs_2 = np.zeros(n, dtype=float)

        return lhs_12, lhs_21, lhs_22, rhs_1, rhs_2


class ReactionWheel(Actuator):
    """
    Model of a reaction wheel actuator.
    """

    def __init__(
        self,
        max_torque: float,
        max_rpm: float,
        inertia: float,
        axis: np.ndarray,
        max_current: float = 1.0,
        T_current: float = 0.1,
    ) -> None:
        super().__init__()
        """
        Initializes the ReactionWheel.

        Parameters
        ----------
        max_torque : float
            Maximum torque the wheel can produce [N*m].
        max_rpm : float
            Maximum speed in RPM.
        inertia : float
            Moment of inertia of the rotor about its spin axis [kg*m^2].
        axis : np.ndarray
            Rotation axis vector in the body frame.
        max_current : float, optional
            Maximum current limit [A], by default 1.0.
        T_current : float, optional
            Time constant for current dynamics [s], by default 0.1.

        Raises
        ------
        ValueError
            If the axis vector is zero.
        """
        self.max_torque = float(max_torque)
        self.max_rpm = float(max_rpm)
        self.inertia = float(inertia)

        axis = np.asarray(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            msg = "ReactionWheel axis must be non-zero!"
            raise ValueError(msg)
        self.axis = axis / norm

        self.max_current = float(max_current)
        self.T_current = float(T_current)

        self.K_t = self.max_torque / self.max_current

        self.max_omega = 2.0 * np.pi * self.max_rpm / 60.0

        self.n_states = 2
        self.n_inputs = 1

        self.header = ["t", "omega_w", "i", "u_cmd"]

    def log(self, t, x_act, u, env) -> None:  # type: ignore[no-untyped-def]
        """
        Logs the reaction wheel state.
        """
        if self.logger is not None:
            omega_w = x_act[0]
            x_act_val = x_act[1][0] if isinstance(x_act[1], np.ndarray) and x_act[1].size == 1 else x_act[1]
            i = float(np.clip(x_act_val, -self.max_current, self.max_current))
            u_val = u[0] if isinstance(u, np.ndarray) and u.size == 1 else u
            self.logger.log([t, omega_w, i, u_val])

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the ReactionWheel configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameters.
        """
        return {
            "max_torque": self.max_torque,
            "max_rpm": self.max_rpm,
            "inertia": self.inertia,
            "max_current": self.max_current,
            "T_current": self.T_current,
            "axis": self.axis.tolist(),
        }

    def __call__(self, idx, x_act, x, u, env) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
        """
        Calculates the contribution of the reaction wheel to the satellite dynamics.

        See README.md for details
        """
        omega = x[10:13]
        omega_w = x_act[idx][0]
        u_val = u[idx][0] if isinstance(u[idx], np.ndarray) and u[idx].size == 1 else u[idx]
        i_cmd = float(np.clip(u_val, -self.max_current, self.max_current))
        x_act_val = (
            x_act[idx][1][0] if isinstance(x_act[idx][1], np.ndarray) and x_act[idx][1].size == 1 else x_act[idx][1]
        )
        i = float(np.clip(x_act_val, -self.max_current, self.max_current))

        tau_m = self.K_t * i
        tau_m = np.clip(tau_m, -self.max_torque, self.max_torque)
        if omega_w >= self.max_omega:
            tau_m = min(tau_m, 0.0)
            tau_m = min(tau_m, 0.0)
        elif omega_w <= -self.max_omega:
            tau_m = max(tau_m, 0.0)

        lhs_12 = np.vstack([self.inertia * self.axis, np.zeros(3)]).T
        lhs_21 = np.vstack([self.axis, np.zeros(3)])
        lhs_22 = np.eye(2)

        rhs_1 = -np.cross(omega, self.inertia * omega_w * self.axis)

        di_dt = (i_cmd - i) / self.T_current

        if i >= self.max_current:
            di_dt = min(di_dt, 0.0)
        elif i <= -self.max_current:
            di_dt = max(di_dt, 0.0)

        rhs_2 = np.array([tau_m / self.inertia, di_dt])

        return lhs_12, lhs_21, lhs_22, rhs_1, rhs_2


class Magnetorquer(Actuator):
    """
    Model of a magnetorquer actuator.
    """

    def __init__(self, max_moment: float, axis: np.ndarray, max_current: float = 1.0, T_current: float = 0.1) -> None:
        super().__init__()
        """
        Initializes the Magnetorquer.

        Parameters
        ----------
        max_moment : float
            Maximum magnetic moment [A*m^2].
        axis : np.ndarray
            Axis vector in the body frame.
        max_current : float, optional
            Maximum current limit [A], by default 1.0.
        T_current : float, optional
            Time constant for current dynamics [s], by default 0.1.

        Raises
        ------
        ValueError
            If the axis vector is zero.
        """
        self.max_moment = float(max_moment)
        axis = np.asarray(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            msg = "Magnetorquer axis must be non-zero!"
            raise ValueError(msg)
        self.axis = axis / norm

        self.max_current = float(max_current)
        self.T_current = float(T_current)
        self.K_t = self.max_moment / self.max_current

        self.n_states = 1
        self.n_inputs = 1

        self.header = ["t", "i", "u_cmd"]

    def log(self, t, x_act, u, env) -> None:  # type: ignore
        """
        Logs the magnetorquer state.
        """
        if self.logger is not None:
            x_act_val = x_act[0] if isinstance(x_act, np.ndarray) and x_act.size == 1 else x_act
            i = float(np.clip(x_act_val, -self.max_current, self.max_current))
            u_val = u[0] if isinstance(u, np.ndarray) and u.size == 1 else u
            self.logger.log([t, i, u_val])

    def torque(self, i: float, B_body: np.ndarray) -> np.ndarray:
        m_vec = (self.K_t * i) * self.axis
        return np.cross(m_vec, B_body)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the Magnetorquer configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameters.
        """
        return {
            "max_moment": self.max_moment,
            "axis": self.axis.tolist(),
            "max_current": self.max_current,
            "T_current": self.T_current,
        }

    def __call__(
        self,
        idx: int,
        x_act: list,  # type: ignore[type-arg]
        x: np.ndarray,
        u: np.ndarray,
        env: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the contribution of the magnetorquer to the satellite dynamics.

        """
        x_act_val = x_act[idx][0] if isinstance(x_act[idx], np.ndarray) and x_act[idx].size == 1 else x_act[idx]
        i = float(np.clip(x_act_val, -self.max_current, self.max_current))
        u_val = u[idx][0] if isinstance(u[idx], np.ndarray) and u[idx].size == 1 else u[idx]
        i_cmd = float(np.clip(u_val, -self.max_current, self.max_current))

        B_body = env["B_body"]

        # Torque = m x B = (K_t * i * axis) x B
        m_vec = (self.K_t * i) * self.axis
        tau_mag = np.cross(m_vec, B_body)

        lhs_12 = np.zeros((3, 1))
        lhs_21 = np.zeros((1, 3))
        lhs_22 = np.array([[1.0]])

        rhs_1 = tau_mag

        di_dt = (i_cmd - i) / self.T_current
        if i >= self.max_current:
            di_dt = min(di_dt, 0.0)
        elif i <= -self.max_current:
            di_dt = max(di_dt, 0.0)

        rhs_2 = np.array([di_dt])

        return lhs_12, lhs_21, lhs_22, rhs_1, rhs_2
