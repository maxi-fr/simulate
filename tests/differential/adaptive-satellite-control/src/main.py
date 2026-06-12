import argparse
import datetime
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import simulation.disturbances as dis
import simulation.environment as env
from flight_software import controllers, estimators
from flight_software.controllers import Controller
from flight_software.estimators import AttitudeEstimator
from scipy.spatial.transform import Rotation as R
from simulation.dynamics import SGP4
from simulation.kinematics import eci_to_geodedic, euler_ocr_to_sbc, orc_to_eci, orc_to_sbc
from simulation.satellite import Spacecraft
from tqdm import tqdm
from utils import Logger, PiecewiseConstant, Quaternion, floor_time_to_minute, floor_time_to_second, string_to_timedelta


class Simulation:
    """
    Simulation environment for satellite attitude and orbit dynamics.
    """

    def __init__(
        self,
        sat: Spacecraft,
        controller: Controller,
        estimator: AttitudeEstimator,
        tle: tuple[str, str],
        initial_attitude_BO: R,
        initial_ang_vel_B: np.ndarray,
        dt: datetime.timedelta,
        t0: datetime.datetime,
        tf: datetime.datetime,
        enable_log: bool = True,
        enable_disturbance_torques: bool = True,
        enable_disturbance_forces: bool = True,
        log_folder: str | None = None,
    ) -> None:
        """
        Initializes the Simulation.

        Parameters
        ----------
        sat : Spacecraft
            The spacecraft model.
        controller : Controller
            The attitude controller.
        estimator : AttitudeEstimator
            The attitude estimator.
        tle : tuple[str, str]
            Two-line element set (TLE) for initial orbit.
        initial_attitude_BO : scipy.spatial.transform.Rotation
            Initial attitude rotation from Orbital Reference Frame (ORC) to Body Frame (SBC).
        initial_ang_vel_B : np.ndarray
            Initial angular velocity in the body frame [rad/s].
        dt : datetime.timedelta
            Simulation time step.
        t0 : datetime.datetime
            Start time.
        tf : datetime.datetime
            End time.
        enable_log : bool, optional
            Whether to enable logging, by default True.
        enable_disturbance_torques : bool, optional
            Whether to enable disturbance torques, by default True.
        enable_disturbance_forces : bool, optional
            Whether to enable disturbance forces, by default True.
        log_folder : str | None, optional
            Folder to save logs, by default None.
        """
        self.controller = controller
        self.att_estimator = estimator

        self.t0 = t0
        self.dt = dt
        self.tf = tf

        self.sat = sat
        self.inital_state = np.zeros(22)

        orbit_model = SGP4.from_tle(*tle)
        self.initial_tle = tle
        initial_r_ECI, initial_v_ECI = orbit_model.propagate(t0)
        self.inital_state[:3] = initial_r_ECI
        self.inital_state[3:6] = initial_v_ECI

        self.inital_state[6:10] = Quaternion.from_scipy(
            initial_attitude_BO * orc_to_eci(self.inital_state[0:3], self.inital_state[3:6]).inv(), canonical=False
        ).to_array()

        self.enable_disturbance_torques = enable_disturbance_torques
        self.enable_disturbance_forces = enable_disturbance_forces

        if initial_ang_vel_B is not None:
            self.inital_state[10:13] = initial_ang_vel_B

        self.enable_log = enable_log
        if self.enable_log:
            if log_folder is None:
                self.log_folder = "Simulation_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            else:
                self.log_folder = log_folder

            if not Path(self.log_folder).exists():
                Path(self.log_folder).mkdir(parents=True)

            with open(os.path.join(self.log_folder, ".gitignore"), "w") as f:
                f.write("*")

            self.to_json(os.path.join(self.log_folder, "config.json"))

            self.state_logger = Logger(
                os.path.join(self.log_folder, "state.csv"),
                [
                    "t",
                    "r_eci_x",
                    "r_eci_y",
                    "r_eci_z",
                    "v_eci_x",
                    "v_eci_y",
                    "v_eci_z",
                    "q_BI_x",
                    "q_BI_y",
                    "q_BI_z",
                    "q_BI_w",
                    "omega_x",
                    "omega_y",
                    "omega_z",
                    "omega_rw_1",
                    "omega_rw_2",
                    "omega_rw_3",
                    "i_mag_1",
                    "i_mag_2",
                    "i_mag_3",
                    "i_rw_1",
                    "i_rw_2",
                    "i_rw_3",
                    "h_rw_x",
                    "h_rw_y",
                    "h_rw_z",
                ],
            )
            self.env_logger = Logger(
                os.path.join(self.log_folder, "environment.csv"),
                [
                    "t",
                    "rho",
                    "B_eci_x",
                    "B_eci_y",
                    "B_eci_z",
                    "sun_pos_x",
                    "sun_pos_y",
                    "sun_pos_z",
                    "in_shadow",
                    "moon_pos_x",
                    "moon_pos_y",
                    "moon_pos_z",
                ],
            )
            if self.att_estimator is not None:
                self.att_estimator.init_log(self.log_folder)
            self.sat.init_log(self.log_folder)

        self.controller.init_satellite_model(sat)
        # TODO: in simulation config should be able to define a satellite model to controller. If not defined the true model is used

        self.sun_position = PiecewiseConstant(fn=env.sun_position, time_bucket_fn=floor_time_to_minute)
        self.moon_position = PiecewiseConstant(fn=env.moon_position, time_bucket_fn=floor_time_to_minute)

        self.atmosphere_density = PiecewiseConstant(fn=env.atmosphere_density_msis, time_bucket_fn=floor_time_to_second)
        self.magnetic_field = PiecewiseConstant(fn=env.magnetic_field_vector, time_bucket_fn=floor_time_to_second)

    @classmethod
    def from_json(
        cls,
        file_path: str,
        enable_log: bool = True,
        controller: Controller | None = None,
        estimator: AttitudeEstimator | None = None,
        log_folder: str | None = None,
    ) -> "Simulation":
        """
        Creates a Simulation instance from a JSON configuration file.

        Parameters
        ----------
        file_path : str
            Path to the JSON configuration file.
        enable_log : bool, optional
            Whether to enable logging, by default True.
        controller : Optional[Controller], optional
            Controller instance to override config, by default None.
        estimator : Optional[AttitudeEstimator], optional
            Estimator instance to override config, by default None.
        log_folder : Optional[str], optional
            Folder to save logs, by default None.

        Returns
        -------
        Simulation
            Initialized Simulation object.
        """
        with open(file_path) as f:
            data = json.load(f)

        if controller is None:
            cont: Controller = getattr(controllers, data["Controller"]["name"])(**data["Controller"]["params"])
        else:
            cont = controller

        data_sim = data["Simulation"]
        t0 = datetime.datetime.fromisoformat(data_sim["Start"])
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=datetime.UTC)

        dt = string_to_timedelta(data_sim["Stepsize"])

        dur = string_to_timedelta(data_sim["Duration"])

        tf = t0 + dur

        enable_disturbance_torques = data_sim.get("DisturbanceTorques", True)
        enable_disturbance_forces = data_sim.get("DisturbanceForces", True)

        data_init_state = data["InitialState"]

        tle1 = data_init_state["TLE"]["Line 1"]
        tle2 = data_init_state["TLE"]["Line 2"]
        orbit_model = SGP4.from_tle(tle1, tle2)
        r_ECI, v_ECI = orbit_model.propagate(t0)

        roll = data_init_state["Attitude"]["Roll (deg)"]
        pitch = data_init_state["Attitude"]["Pitch (deg)"]
        yaw = data_init_state["Attitude"]["Yaw (deg)"]

        R_BO = euler_ocr_to_sbc(roll, pitch, yaw)

        ang_vel_B_BO = np.deg2rad(data_init_state["AngularVelocity (wrt ORC in SBC) (deg/s)"])

        orbit_ang_vel = np.linalg.norm(v_ECI) / np.linalg.norm(r_ECI)
        init_ang_vel_B_BI = ang_vel_B_BO + R_BO.apply(np.array((0, -orbit_ang_vel, 0)))

        est = None
        if estimator is None:
            q0 = Quaternion.from_scipy(R_BO * orc_to_eci(r_ECI, v_ECI).inv(), canonical=False).to_array()

            est_data = data.get("Estimator", {})
            est_name = est_data.get("name", "AttitudeEKF")
            est_params = est_data.get("params", {})

            # Default params for EKF if missing
            if est_name == "AttitudeEKF" and not est_params:
                est_params = {
                    "P0": np.eye(6),
                    "Qc": np.eye(6) * 1e-6,
                    "R_sun": np.eye(3) * 1e-4,
                    "R_mag": np.eye(3) * 1e-12,
                }

            est = getattr(estimators, est_name)(q0=q0, **est_params)
        else:
            est = estimator

        sc_params = data["SpacecraftParams"]

        return cls(
            Spacecraft.from_dict(sc_params),
            cont,
            est,
            (tle1, tle2),
            R_BO,
            init_ang_vel_B_BI,
            dt,
            t0,
            tf,
            enable_log,
            enable_disturbance_torques,
            enable_disturbance_forces,
            log_folder,
        )

    def to_json(self, file_path: str) -> None:
        """
        Saves the simulation configuration to a JSON file.

        Parameters
        ----------
        file_path : str
            Path to the output JSON file.
        """
        data = {}

        data["Simulation"] = {
            "Start": self.t0.isoformat(),
            "Stepsize": str(self.dt),
            "Duration": str(self.tf - self.t0),
            "DisturbanceTorques": self.enable_disturbance_torques,
            "DisturbanceForces": self.enable_disturbance_forces,
        }

        data["Controller"] = {"name": self.controller.__class__.__name__, "params": self.controller.to_dict()}

        r_eci = self.inital_state[0:3]
        v_eci = self.inital_state[3:6]
        q_BI = Quaternion.from_array(self.inital_state[6:10])
        omega_BI_B = self.inital_state[10:13]

        R_BO = orc_to_sbc(q_BI, r_eci, v_eci)

        pitch, roll, yaw = R_BO.to_scipy().as_euler("XYZ", degrees=True)

        orbit_ang_vel = np.linalg.norm(v_eci) / np.linalg.norm(r_eci)
        omega_OI_B = R_BO.apply(np.array([0, -orbit_ang_vel, 0]))
        ang_vel_B_BO = omega_BI_B - omega_OI_B

        data["InitialState"] = {
            "TLE": {"Line 1": self.initial_tle[0], "Line 2": self.initial_tle[1]},
            "Attitude": {"Roll (deg)": float(roll), "Pitch (deg)": float(pitch), "Yaw (deg)": float(yaw)},
            "AngularVelocity (wrt ORC in SBC) (deg/s)": np.rad2deg(ang_vel_B_BO).tolist(),
        }

        data["SpacecraftParams"] = self.sat.to_dict()

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def run(self) -> None:
        """
        Runs the simulation loop.
        """
        state = self.inital_state
        t = self.t0

        with tqdm(total=(self.tf - self.t0).total_seconds() / 60, desc="Simulation time", unit="sim min") as pbar:
            while t < self.tf:
                r_eci = state[0:3]
                state[3:6]
                q_BI = Quaternion.from_array(state[6:10])
                state[10:13]

                lat, lon, alt = eci_to_geodedic(r_eci)

                rho: float = self.atmosphere_density(t, lat, lon, alt)
                B_eci: np.ndarray = self.magnetic_field(t, lat, lon, alt)
                B_body = q_BI.apply(B_eci)

                sun_pos: np.ndarray = self.sun_position(t)
                in_shadow = env.is_in_shadow(r_eci, sun_pos)
                moon_pos: np.ndarray = self.moon_position(t)
                s_I = sun_pos / np.linalg.norm(sun_pos)
                s_B = q_BI.apply(s_I)

                env_dict = {
                    "rho": rho,
                    "B_body": B_body,
                    "sun_pos": sun_pos,
                    "s_B": s_B,
                    "in_shadow": in_shadow,
                    "moon_pos": moon_pos,
                }

                state_dict = self.sat.state_to_dict(state)

                self.sat.measure_sensors(t, state_dict, env_dict)

                # Flight software (FSW)
                sensor_data = self.sat.read_sensors(t)

                sun_mea, new_sun_mea = sensor_data["SunSensor"][0]
                mag_mea, new_mag_mea = sensor_data["Magnetometer"][0]
                gps_mea, new_gps_mea = sensor_data["GPS"][0]
                gyro_mea, new_gyro_mea = sensor_data["Gyroscope"][0]
                omega_rw_mea, new_omega_rw_mea = sensor_data["RW_tachometer"][0]
                omega_rw_mea, new_omega_rw_mea = sensor_data["RW_tachometer"][0]

                # self.orbital_estimator.predict(t)

                # TODO: There should be one estimator class which handles the estimation of everything.

                self.att_estimator.predict(t, gyro_mea)

                if in_shadow:
                    new_sun_mea = False
                    sun_mea = np.array([np.nan, np.nan, np.nan])
                if new_sun_mea:
                    self.att_estimator.update_sun(t, sun_mea)
                if new_mag_mea:
                    self.att_estimator.update_mag(t, mag_mea, state[:3])

                q_est, omega_est = self.att_estimator.get_state()

                # if new_gps_mea:
                #     self.orbital_estimator.update_gps(t, gps_mea)
                r_eci_est, v_eci_est = state[:3], state[3:6]  # self.orbital_estimator.get_state()

                np.array(omega_rw_mea, dtype=float)

                h_w = np.zeros(3)
                # for i, rw in enumerate(self.sat.rws):
                #     omega_parallel_body = float(np.dot(rw.axis, omega_est))
                #     h_w += rw.inertia * (omega_rw_est[i] + omega_parallel_body) * rw.axis

                state_est = np.concatenate((q_est, omega_est, h_w))
                orbit_state_est = np.concatenate((r_eci_est, v_eci_est))

                u = self.controller.calc_input_cmds(
                    t, state_est, orbit_state_est, B_eci
                )  # TODO: refactor how state and environment get passed to controllers

                if self.enable_log:
                    h_rw = h_w

                    self.state_logger.log([t, *list(state), *list(h_rw)])
                    self.env_logger.log([t, rho, *list(B_eci), *list(sun_pos), in_shadow, *list(moon_pos)])

                    self.sat.log_actuators(t, state, u, env_dict)

                next_state = rk4_step(self.world_dynamics, state, u, t, self.dt)

                next_state[6:10] /= np.linalg.norm(next_state[6:10])

                t += self.dt
                state = next_state

                pbar.update(self.dt.total_seconds() / 60)

    def world_dynamics(self, x: np.ndarray, u: np.ndarray, t: datetime.datetime) -> np.ndarray:
        """
        Helper function to wrap dynamics for whole system for integration.

        Parameters
        ----------
        x : np.ndarray, shape (n_x,)
            All variable states during integration
        u : np.ndarray, shape (n_u,)
            All actuator commands during integration. They are constant during integration.
        t : datetime.datetime
            Current simulation time.

        Returns
        -------
        np.ndarray, shape (22,)
            State derivative dx/dt at time t.
        """
        r_eci = x[0:3]
        v_eci = x[3:6]
        q_BI = Quaternion.from_array(x[6:10])
        omega = x[10:13]

        x_actuators = []
        start = 13
        for act in self.sat.actuators:
            end = start + act.n_states

            x_actuators.append(x[start:end])

            start = end

        u_split = []
        start = 0
        for act in self.sat.actuators:
            end = start + act.n_inputs

            u_split.append(u[start:end])

            start = end

        q_BO = orc_to_sbc(q_BI, r_eci, v_eci)

        lat, lon, alt = eci_to_geodedic(r_eci)

        rho: float = self.atmosphere_density(t, lat, lon, alt)
        B_eci: np.ndarray = self.magnetic_field(t, lat, lon, alt)
        B_body = q_BI.apply(B_eci)

        sun_pos: np.ndarray = self.sun_position(t)
        in_shadow = env.is_in_shadow(r_eci, sun_pos)
        moon_pos: np.ndarray = self.moon_position(t)
        s_I = sun_pos / np.linalg.norm(sun_pos)
        s_B = q_BI.apply(s_I)

        env_dict = {
            "rho": rho,
            "B_body": B_body,
            "sun_pos": sun_pos,
            "s_B": s_B,
            "in_shadow": in_shadow,
            "moon_pos": moon_pos,
            "d_v": np.zeros(3),
            "R_BO": q_BO,
        }

        F_grav = dis.non_spherical_gravity_forces(r_eci, self.sat.m)
        F_third = dis.third_body_forces(r_eci, self.sat.m, sun_pos, moon_pos)
        tau_gg = dis.gravity_gradient(r_eci, q_BO, self.sat.J_B)
        F_aero, tau_aero = dis.aerodynamic_drag(r_eci, v_eci, q_BI, self.sat.surfaces, rho)
        F_SRP, tau_SRP = dis.solar_radiation_pressure(r_eci, sun_pos, in_shadow, q_BI, self.sat.surfaces)

        if not self.enable_disturbance_torques:
            tau_gg = np.zeros(3)
            tau_aero = np.zeros(3)
            tau_SRP = np.zeros(3)

        if not self.enable_disturbance_forces:
            F_grav = np.zeros(3)
            F_third = np.zeros(3)
            F_aero = np.zeros(3)
            F_SRP = np.zeros(3)

        L = np.zeros((x.size - 10, x.size - 10))  # 3 (omega) + n_actuator_states
        L[:3, :3] = self.sat.J_B

        rhs = np.zeros(x.size - 10)

        cross_term = np.cross(omega, self.sat.J_B @ omega)
        rhs[:3] = tau_gg + tau_aero + tau_SRP - cross_term

        start = 3
        for i, act in enumerate(self.sat.actuators):
            lhs_12, lhs_21, lhs_22, rhs_1, rhs_2 = act(i, x_actuators, x, u_split, env_dict)

            end = start + act.n_states

            L[:3, start:end] = lhs_12
            L[start:end, :3] = lhs_21
            L[start:end, start:end] = lhs_22

            rhs[:3] += rhs_1
            rhs[start:end] = rhs_2

            start = end

        d_r = v_eci
        d_v = self.sat.orbit_dynamics(r_eci, q_BI.conjugate().apply(F_aero + F_SRP) + F_third + F_grav)
        d_q = q_BI.kinematics(omega)

        return np.concatenate((d_r, d_v, d_q, np.linalg.solve(L, rhs)))


def rk4_step(
    f: Callable[[np.ndarray, np.ndarray, datetime.datetime], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    t: datetime.datetime,
    dt: datetime.timedelta,
) -> np.ndarray:
    """
    Classic 4th-order Runge-Kutta integrator.

    Parameters
    ----------
    f : Callable[[np.ndarray, np.ndarray, datetime.datetime], np.ndarray]
        Function f(x, u, t) -> dx/dt that computes the state derivative.
    x : np.ndarray
        Current state.
    u : np.ndarray
        Current input
    t : datetime.datetime
        Current simulation time.
    dt : datetime.timedelta
        Time step.

    Returns
    -------
    np.ndarray
        State after one time step.
    """
    dt_float = dt.total_seconds()

    k1 = f(x, u, t)
    k2 = f(x + 0.5 * dt_float * k1, u, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt_float * k2, u, t + 0.5 * dt)
    k4 = f(x + dt_float * k3, u, t + dt)

    x_next: np.ndarray = x + (dt_float / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x_next


if __name__ == "__main__":
    default_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "simulation_config.json")

    parser = argparse.ArgumentParser(description="Run a single satellite simulation.")
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default=default_file_path,
        help="Path to the simulation configuration JSON file.",
    )
    parser.add_argument("--disable-log", action="store_true", help="Disable simulation logging.")

    args = parser.parse_args()

    enable_log = not args.disable_log

    sim = Simulation.from_json(args.config_file, enable_log=enable_log)
    sim.run()
