import datetime
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any

import casadi as ca
import control as ct
import numpy as np
import scipy
import scipy.linalg
from simulation.actuators import Magnetorquer, ReactionWheel
from simulation.dynamics import SGP4
from simulation.environment import magnetic_field_vector
from simulation.kinematics import eci_to_geodedic, orc_to_eci, orc_to_sbc
from simulation.satellite import Spacecraft
from utils import Logger, Quaternion

from flight_software.controller_models import (
    build_reduced_system_dynamics,
    integrator,
    quaternion_rotation,
    satellite_dynamics,
)


class Controller(ABC):
    """
    Abstract base class for controllers.
    """

    def __init__(self) -> None:
        """
        Initializes the controller.

        """
        self.logger: Logger | None = None
        self.header: list[str] = []

    def init_satellite_model(self, sat: Spacecraft) -> None:
        """
        Initializes the satellite model within the controller.

        Parameters
        ----------
        sat : Spacecraft
            The spacecraft object containing parameters and configuration.
        """
        self.mtqs: list[Magnetorquer] = [mtq for mtq in sat.actuators if isinstance(mtq, Magnetorquer)]
        self.rws: list[ReactionWheel] = [rw for rw in sat.actuators if isinstance(rw, ReactionWheel)]

        self.i_sat = np.concatenate(([mtq.max_current for mtq in self.mtqs], [rw.max_current for rw in self.rws]))
        self.omega_w_sat = np.array([rw.max_omega for rw in self.rws])

        self.Lambda_m = np.vstack([mtq.axis for mtq in self.mtqs]).T
        self.Lambda_w = np.vstack([rw.axis for rw in self.rws]).T

        self.K_m = np.array([mtq.K_t for mtq in self.mtqs])
        self.K_w = np.array([rw.K_t for rw in self.rws])
        self.J_w = np.array([rw.inertia for rw in self.rws])

        self.J_w_lambda_w_inv = np.diag(np.reciprocal(self.J_w)) @ np.linalg.inv(self.Lambda_w)
        self.K_w_Lambda = np.diag(self.K_w) @ self.Lambda_w
        self.K_m_Lambda = np.diag(self.K_m) @ self.Lambda_m

        J_rw = sum([self.rws[i].inertia * np.outer(self.rws[i].axis, self.rws[i].axis) for i in range(len(self.rws))])
        self.J_hat = sat.J_B - J_rw

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Converts the controller to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the controller.
        """
        raise NotImplementedError

    def calc_nadir_state_error(self, state_est: np.ndarray, orbit_state: np.ndarray) -> np.ndarray:
        """
        Calculates the state error relative to the nadir frame.

        Parameters
        ----------
        state_est : np.ndarray
            Current state estimation.
        orbit_state : np.ndarray
            Current orbit state (position and velocity).

        Returns
        -------
        np.ndarray
            Error state vector.
        """
        r_eci, v_eci = orbit_state[:3], orbit_state[3:6]
        q_BI = Quaternion.from_array(state_est[:4])
        omega_BI = state_est[4:7]
        h_w = state_est[7:]

        q_BO = orc_to_sbc(q_BI, r_eci, v_eci)

        q_err_full = q_BO.to_array()

        omega_0 = np.linalg.norm(v_eci) / np.linalg.norm(r_eci)
        omega_c = np.array([0.0, -omega_0, 0.0])

        omega_err = omega_BI - q_BO.apply(omega_c)

        h_w_err = h_w + self.J_hat @ omega_c

        return np.concatenate((q_err_full[:3] * np.sign(q_err_full[3]), omega_err, h_w_err))

    @abstractmethod
    def calc_input_cmds(
        self, t: datetime.datetime, att_state: np.ndarray, orbit_state: np.ndarray, B_eci: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the control input commands.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        att_state : np.ndarray
            Attitude state vector.
        orbit_state : np.ndarray
            Orbit state vector.
        B_eci : np.ndarray
            Magnetic field vector in the ECI frame.

        Returns
        -------
        np.ndarray
            Control input vector.
        """
        raise NotImplementedError

    def notify_actuator_failure(self, actuator_type: str, index: int) -> None:
        """
        Notifies the controller that an actuator has failed.

        Parameters
        ----------
        actuator_type : str
            Type of actuator ('rw' for reaction wheel, 'mtq' for magnetorquer).
        index : int
            Index of the failed actuator.
        """


class ZeroInputs(Controller):
    """
    A controller that always outputs zero for all control inputs.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Initialize the ZeroInputs controller.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        super().__init__()

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the controller to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the controller.
        """
        return {}

    def calc_input_cmds(
        self, t: datetime.datetime, att_state: np.ndarray, orbit_state: np.ndarray, B_eci: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the control input commands.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        att_state : np.ndarray
            Attitude state vector.
        orbit_state : np.ndarray
            Orbit state vector.
        B_eci : np.ndarray
            Magnetic field vector in the ECI frame.

        Returns
        -------
        np.ndarray
            Control input vector (all zeros).
        """
        return np.zeros(6)


class PI(Controller):
    """
    Proportional-Integral controller with anti-windup.
    """

    def __init__(
        self,
        K_q: np.ndarray,
        K_omega: np.ndarray,
        K_w: np.ndarray,
        K_q_int: np.ndarray,
        operating_point: tuple[np.ndarray, np.ndarray],
        dt: float,
        m: float | None = None,
    ) -> None:
        """
        Initializes the PI controller.

        Parameters
        ----------
        K_q : np.ndarray
            Proportional gain matrix for attitude error.
        K_omega : np.ndarray
            Derivative gain matrix for angular velocity error.
        K_w : np.ndarray
            Gain matrix for wheel momentum error.
        K_q_int : np.ndarray
            Integral gain matrix for attitude error.
        operating_point : Tuple[np.ndarray, np.ndarray]
            The operating point (x_star, u_star).
        dt : float
            Time step.
        m : float, optional
            Anti-windup gain.
        """
        super().__init__()
        self.x_star, self.u_star = np.asarray(operating_point[0]), np.asarray(operating_point[1])
        self.q_err_int = np.zeros(3)

        self.K_q = np.asarray(K_q)
        self.K_omega = np.asarray(K_omega)
        self.K_wheel = np.asarray(K_w)
        self.K_q_int = np.asarray(K_q_int)
        self.K = np.hstack((self.K_q, self.K_omega, self.K_wheel))

        self.dt = dt

        self.m = m

        self.header = [
            "t",
            "q_err_x",
            "q_err_y",
            "q_err_z",
            "omega_err_x",
            "omega_err_y",
            "omega_err_z",
            "h_w_x",
            "h_w_y",
            "h_w_z",
            "q_err_int_x",
            "q_err_int_y",
            "q_err_int_z",
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the controller to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the controller.
        """
        return {
            "K_q": self.K_q.tolist(),
            "K_omega": self.K_omega.tolist(),
            "K_wheel": self.K_wheel.tolist(),
            "K_q_int": self.K_q_int.tolist(),
            "operating_point": [self.x_star.tolist(), self.u_star.tolist()],
            "m": self.m,
        }

    def calc_input_cmds(
        self, t: datetime.datetime, att_state: np.ndarray, orbit_state: np.ndarray, B_eci: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the control input.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        att_state : np.ndarray
            Attitude state vector.
        orbit_state : np.ndarray
            Orbit state vector.
        B_eci : np.ndarray
            Magnetic field vector in the ECI frame.

        Returns
        -------
        np.ndarray
            Control input vector.
        """
        state_err = self.calc_nadir_state_error(att_state, orbit_state)
        q_err = state_err[:3]

        self.q_err_int += q_err * self.dt

        u = -(self.K @ state_err + self.K_q_int @ self.q_err_int)

        if self.logger is not None:
            self.logger.log(list(chain.from_iterable(([t], state_err, self.q_err_int))))

        B_body = orc_to_sbc(Quaternion.from_array(att_state[:4]), orbit_state[:3], orbit_state[3:6]).apply(B_eci)

        return to_current_commands(u + self.u_star, B_body, self.mtqs, self.rws)


def update_lqr_warm_start(
    A: np.ndarray, B_current: np.ndarray, Q: np.ndarray, R: np.ndarray, P_prev: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Updates the LQR gain using a single Newton-Kleinman iteration (Warm Start).

    Parameters
    ----------
    A : np.ndarray
        State matrix (constant).
    B_current : np.ndarray
        Current time-varying input matrix B(t).
    Q : np.ndarray
        State cost matrix.
    R : np.ndarray
        Input cost matrix.
    P_prev : np.ndarray
        The solution P from the previous time step.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        K_new: The updated control gain.
        P_new: The updated Riccati solution.
    """
    #    K_0 = (R + B^T P B)^-1 B^T P A
    R_total = R + B_current.T @ P_prev @ B_current
    K_0 = np.linalg.solve(
        R_total, B_current.T @ P_prev @ A
    )  # TODO: might be unnecessary because we already have lat K matrix

    A_cl = A - B_current @ K_0

    S = Q + K_0.T @ R @ K_0

    P_new = scipy.linalg.solve_discrete_lyapunov(A_cl.T, S)

    R_total_new = R + B_current.T @ P_new @ B_current
    K_new = np.linalg.solve(R_total_new, B_current.T @ P_new @ A)

    return K_new, P_new


class AdaptiveLQR(PI):
    """
    Adaptive Linear Quadratic Regulator (LQR) controller.
    """

    def __init__(self, Q: np.ndarray, R: np.ndarray, dt: float, m: float | None = None, k_i: float = 0.0) -> None:
        """
        Initializes the AdaptiveLQR controller.

        Parameters
        ----------
        Q : np.ndarray
            State cost matrix.
        R : np.ndarray
            Input cost matrix.
        dt : float
            Time step.
        m : float, optional
            Anti-windup gain.
        k_i : float, optional
            Integral gain for attitude error, by default 0.0.
        """
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

        self.dt = dt
        self.m = m

        u_star = np.zeros(6)
        x_star = np.zeros(10)

        K = np.zeros((6, 9))
        K_q_int = k_i * np.vstack((np.zeros((3, 3)), np.eye(3)))

        super().__init__(K[:, :3], K[:, 3:6], K[:, 6:9], K_q_int, (x_star, u_star), dt, m)
        self.P: np.ndarray | None = None

    def init_satellite_model(self, sat: Spacecraft) -> None:
        """
        Initializes the satellite model within the controller.

        Parameters
        ----------
        sat : Spacecraft
            The spacecraft object containing parameters and configuration.
        """
        super().init_satellite_model(sat)

        _, self.A_func, self.B_func = build_reduced_system_dynamics(self.dt, self.J_hat)

    def calc_input_cmds(
        self, t: datetime.datetime, att_state: np.ndarray, orbit_state: np.ndarray, B_eci: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the control input commands using an adaptive LQR approach.

        Parameters
        ----------
        t : datetime.datetime
            Current time.
        att_state : np.ndarray
            Attitude state vector.
        orbit_state : np.ndarray
            Orbit state vector.
        B_eci : np.ndarray
            Magnetic field vector in the ECI frame.

        Returns
        -------
        np.ndarray
            Control input vector.
        """
        r_eci, v_eci = orbit_state[:3], orbit_state[3:6]

        omega_ref = np.array([0, -np.linalg.norm(v_eci) / np.linalg.norm(r_eci), 0])

        R_IO = orc_to_eci(r_eci, v_eci)
        R_OI = R_IO.inv()
        q_ref = Quaternion.from_scipy(R_OI, canonical=False).to_array()

        self.x_star[:4] = q_ref
        self.x_star[4:7] = omega_ref

        A = self.A_func(self.x_star, self.u_star, B_eci)
        B = self.B_func(self.x_star, self.u_star, B_eci)

        if self.P is None:
            self.K, self.P, _eig_vals = ct.dlqr(A, B, self.Q, self.R)
        else:
            self.K, self.P = update_lqr_warm_start(A, B, self.Q, self.R, self.P)

        return super().calc_input_cmds(t, att_state, orbit_state, B_eci)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the controller to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the controller.
        """
        return {
            "Q": self.Q.tolist(),
            "R": self.R.tolist(),
            "m": self.m,
            "K_q_int": self.K_q_int.tolist(),
            "dt": self.dt,
        }


class ClassicalQuatFeedback(PI):
    """
    Classical quaternion feedback controller.
    """

    def __init__(self, k_p: float, k_d: float, k_i: float, k_m: float, dt: float, m: float | None = None) -> None:
        """
        Initializes the ClassicalQuatFeedback controller.

        RW for attitude stabilization and Magnetorquers for momentum dumping.

        x = [q_err, omega_err, h_w]
        u = [u_mag, u_rw]

        Parameters
        ----------
        k_p : float
            Proportional gain.
        k_d : float
            Derivative gain.
        k_i : float
            Integral gain.
        k_m : float
            Momentum dumping gain.
        dt : float
            Time step.
        m : float, optional
            Anti-windup gain. Defaults to None.
        """
        K_q = np.vstack((np.zeros((3, 3)), np.eye(3) * k_p))
        K_omega = np.vstack((np.zeros((3, 3)), np.eye(3) * k_d))
        K_wheel = np.vstack((np.eye(3) * k_m, np.zeros((3, 3))))
        K_q_int = np.vstack((np.zeros((3, 3)), np.eye(3) * k_i))

        super().__init__(K_q, K_omega, K_wheel, K_q_int, (np.zeros(9), np.zeros(6)), dt, m)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the controller to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the controller.
        """
        return {
            "k_p": self.K_q[0, 0],
            "k_d": self.K_omega[0, 0],
            "k_m": self.K_wheel[0, 0],
            "k_i": self.K_q_int[0, 0],
            "m": self.m,
            "dt": self.dt,
        }


class AvanziniLinear(ClassicalQuatFeedback):
    """
    Avanzini linear controller for attitude stabilization.
    """

    def __init__(
        self,
        damping_ratio: float,
        h_sat: float,
        J: np.ndarray,
        tle1: str,
        tle2: str,
        t0: str,
        k_i: float,
        dt: float,
        m: float | None,
    ) -> None:
        """
        Initializes the AvanziniLinear controller.

        Parameters
        ----------
        damping_ratio : float
            Damping ratio.
        h_sat : float
            Satellite altitude.
        J : np.ndarray
            Inertia matrix.
        tle1 : str
            TLE line 1.
        tle2 : str
            TLE line 2.
        t0 : str
            Initial epoch (ISO format).
        k_i : float
            Integral gain.
        dt : float
            Time step.
        m : float, optional
            Anti-windup gain.
        """
        self.h_sat = h_sat
        self.J = np.asarray(J)
        self.tle1 = tle1
        self.tle2 = tle2
        self.t0 = t0
        self.damping_ratio = damping_ratio

        J_max = np.max(np.linalg.eigvals(J))
        orbit_model = SGP4.from_tle(tle1, tle2)
        t0_ = datetime.datetime.fromisoformat(t0)
        if t0_.tzinfo is None:
            t0_ = t0_.replace(tzinfo=datetime.UTC)

        r_ECI, v_ECI = orbit_model.propagate(t0_)

        omega_0 = abs(v_ECI.mean(axis=0) / np.linalg.norm(r_ECI.mean(axis=0)))

        k_p = 2 * (h_sat * np.e / np.pi) ** 2 / J_max

        omega_n = np.sqrt(k_p / (2 * J_max))
        k_d = max(omega_0 * J_max, 2 * J_max * damping_ratio * omega_n)

        k_m = 2 * omega_0 * (1 + np.sin(orbit_model.satrec.inclo))

        super().__init__(k_p, k_d, k_i, k_m, dt, m)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the controller to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the controller.
        """
        return {
            "damping_ratio": self.damping_ratio,
            "h_sat": self.h_sat,
            "J": self.J.tolist(),
            "tle1": self.tle1,
            "tle2": self.tle2,
            "t0": self.t0,
            "k_i": self.K_q_int[0, 0],
            "m": self.m,
            "dt": self.dt,
        }


def to_current_commands(u: np.ndarray, B: np.ndarray, mtqs: list[Magnetorquer], rws: list[ReactionWheel]) -> np.ndarray:
    """
    Calculates the current commands for magnetorquers and reaction wheels based on desired torques.

    Parameters
    ----------
    u : np.ndarray
        Desired control torques (concatenation of magnetorquer torques and reaction wheel torques).
    B : np.ndarray, shape (3,)
        Magnetic field vector in the body frame [T].
    mtqs : List[Magnetorquer]
        List of Magnetorquer objects.
    rws : List[ReactionWheel]
        List of ReactionWheel objects.

    Returns
    -------
    np.ndarray
        Current commands (concatenation of magnetorquer currents and reaction wheel currents).
    """
    u_mag = u[: len(mtqs)]
    u_rw = u[len(mtqs) :]

    Alpha = np.array([rw.K_t * rw.axis for rw in rws]).T
    rw_i_cmd = np.linalg.solve(Alpha, -u_rw)

    B_norm = np.linalg.norm(B)
    b = B / B_norm
    m_cmd = np.cross(b, u_mag / B_norm)

    Alpha = np.array([m.K_t * m.axis for m in mtqs]).T

    mag_i_cmd = np.linalg.solve(Alpha, m_cmd)

    return np.concatenate((mag_i_cmd, rw_i_cmd))
