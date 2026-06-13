# mypy: ignore-errors

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from simulate.config import load_config
    from simulate.simulation import Simulation
    from spacecraft.frames import euler_from_quaternion, orbital_rate, orc_from_orbit
    from spacecraft.quaternion import Quaternion

    return (
        Path,
        Quaternion,
        Simulation,
        euler_from_quaternion,
        load_config,
        mo,
        np,
        orbital_rate,
        orc_from_orbit,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Nadir Pointing in LEO

    An end-to-end ADCS simulation: a 3U-class CubeSat acquires and holds **nadir pointing**
    in low Earth orbit using **reaction wheels** (with magnetorquer momentum dumping), under
    real disturbances (central + gravity-gradient gravity, third body, solar radiation
    pressure, aerodynamic drag).

    The full satellite is built from [`03_nadir_pointing.yaml`](03_nadir_pointing.yaml) via
    `Simulation.from_yaml`. The loop closes:

    * a **`FullStateEstimator`** — orbit Kalman filter (GPS) + attitude MEKF (gyro / magnetometer
      / sun) — produces `x_hat = [r, v, q, omega, b_body, h_wheel]` and exposes environment
      variables,
    * a **`OrbitReference`** emits the desired attitude *relative to the orbital frame*
      (identity = nadir),
    * a **`QuaternionFeedbackController`** reconstructs the orbital frame from `x_hat` and drives
      the wheels.

    The body starts ~7.5° off nadir; we watch it converge and hold.
    """)
    return


@app.cell
def _(mo):
    controller_select = mo.ui.dropdown(
        options=["Quaternion Feedback", "Adaptive LQR"],
        value="Quaternion Feedback",
        label="Controller Type",
    )
    controller_select
    return (controller_select,)


@app.cell
def _(Path, Simulation, controller_select, load_config, np):
    # One orbit (~95 min) is the config default; a few minutes already shows acquisition + hold, and
    # keeps the notebook responsive (the estimator/disturbances evaluate IGRF/ephemeris/MSIS each step).
    config_path = Path(__file__).resolve().parent / "03_nadir_pointing.yaml"
    config = load_config(config_path)

    _sim_config = dict(config)
    if controller_select.value == "Adaptive LQR":
        _tle = [
            "1 25544U 98067A   24001.50000000  .00000000  00000-0  00000-0 2    07",
            "2 25544 097.6000 010.0000 0001000 000.0000 000.0000 15.25000000000009",
        ]
        _sim_config["controller"] = {
            "class_path": "spacecraft.controller.AdaptiveLQR",
            "dt": 0.2,
            "Q": np.diag([5, 5, 5, 2, 2, 2, 700, 700, 700]).tolist(),
            "R": (1e7 * np.diag([70, 70, 70, 7, 7, 7])).tolist(),
            "inertia": config["dynamics"]["inertia"],
            "tle": _tle,
            "epoch": "2024-01-01T12:00:00",
            "reaction_wheels": {"axes": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], "torque_constant": [0.01, 0.01, 0.01]},
            "magnetorquers": {"axes": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], "dipole_constant": [0.3, 0.3, 0.2]},
        }

    sim = Simulation.from_config(_sim_config)
    sim.t_end = 200.0
    sim.run()
    return config, sim


@app.cell
def _(Quaternion, euler_from_quaternion, np, orbital_rate, orc_from_orbit):
    def extract(sim_obj) -> dict[str, np.ndarray]:
        """Pull the universal/component logs of a run into plain numpy arrays for plotting."""
        logs = sim_obj.logger.universal_logs
        t = np.array([row["t"] for row in logs])
        x = np.array([np.asarray(row["x"]) for row in logs])
        x_hat = np.array([np.asarray(row["x_hat"]) for row in logs])
        u = np.array([np.asarray(row["u"]) for row in logs])

        # Pointing error as the body-vs-ORC (nadir) attitude, in Euler angles [deg].
        euler_err = np.zeros((len(t), 3))
        rate_ff = np.zeros((len(t), 3))
        for k, row in enumerate(x):
            q_oi = orc_from_orbit(row[0:3], row[3:6])
            q_err = Quaternion.from_array(row[6:10]).error_to(q_oi)  # desired q_bo = identity
            euler_err[k] = np.degrees(euler_from_quaternion(q_err))
            rate_ff[k] = orbital_rate(row[0:3], row[3:6])
        return {"t": t, "x": x, "x_hat": x_hat, "u": u, "euler_err": euler_err, "rate_ff": rate_ff}

    return (extract,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Pointing error and body rates

    The pointing error (body attitude relative to the orbital frame, in Euler angles) decays
    from the initial offset to near zero. The body rate tracks the orbital feedforward rate
    (the satellite rotates once per orbit to keep the same face down).
    """)
    return


@app.cell
def _(extract, plt, sim):
    d = extract(sim)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for _j, _lbl in enumerate(("pitch (Y)", "roll (X)", "yaw (Z)")):
        axes[0].plot(d["t"], d["euler_err"][:, _j], label=_lbl)
    axes[0].set_ylabel("pointing error (deg)")
    axes[0].axhline(0.0, color="k", lw=0.5)
    axes[0].legend()
    axes[0].grid(visible=True)
    axes[0].set_title("Nadir pointing error (body relative to ORC)")

    for _j, _lbl in enumerate(("$\\omega_x$", "$\\omega_y$", "$\\omega_z$")):
        axes[1].plot(d["t"], d["x"][:, 10 + _j], label=f"{_lbl} body")
        axes[1].plot(d["t"], d["rate_ff"][:, _j], "--", lw=1, label=f"{_lbl} feedforward")
    axes[1].set_ylabel("body rate (rad/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(visible=True)

    fig.tight_layout()
    plt.gca()
    return (d,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Actuators: wheel speeds and control currents

    The reaction-wheel relative speeds absorb the slewing momentum; the magnetorquers
    (`[i_mtq, i_rw]` command layout) provide slow momentum dumping.
    """)
    return


@app.cell
def _(d, plt):
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for _j in range(3):
        axes2[0].plot(d["t"], d["x"][:, 19 + _j], label=f"wheel {_j} $\\omega_{{rel}}$")
    axes2[0].set_ylabel("wheel speed (rad/s)")
    axes2[0].legend()
    axes2[0].grid(visible=True)
    axes2[0].set_title("Reaction-wheel relative speeds")

    for _j in range(3):
        axes2[1].plot(d["t"], d["u"][:, 3 + _j], label=f"$i_{{rw,{_j}}}$")
    for _j in range(3):
        axes2[1].plot(d["t"], d["u"][:, _j], "--", lw=1, label=f"$i_{{mtq,{_j}}}$")
    axes2[1].set_ylabel("current command (A)")
    axes2[1].set_xlabel("time (s)")
    axes2[1].legend(ncol=3, fontsize=8)
    axes2[1].grid(visible=True)

    fig2.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Estimator vs. truth

    The orbit Kalman filter smooths the noisy GPS to the (one-step-lagged) truth, and the
    attitude MEKF tracks the true attitude to a fraction of a degree.
    """)
    return


@app.cell
def _(Quaternion, d, np, plt, sim):
    elog = sim.logger.component_logs["estimator"]
    bias = np.array([np.asarray(row["gyro_bias"]) for row in elog])

    pos_err = np.linalg.norm(d["x_hat"][:, 0:3] - d["x"][:, 0:3], axis=1)
    att_err = np.array(
        [
            np.degrees(
                2.0
                * np.arctan2(
                    np.linalg.norm(Quaternion.from_array(xt[6:10]).error_to(Quaternion.from_array(xh[6:10])).vec),
                    1.0,
                )
            )
            for xt, xh in zip(d["x"], d["x_hat"], strict=True)
        ]
    )

    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes3[0].plot(d["t"], pos_err, "b-")
    axes3[0].set_ylabel("orbit error (m)")
    axes3[0].grid(visible=True)
    axes3[0].set_title("Estimator error vs. truth")

    axes3[1].plot(d["t"], att_err, "r-")
    axes3[1].set_ylabel("attitude error (deg)")
    axes3[1].grid(visible=True)

    for _j, _lbl in enumerate(("x", "y", "z")):
        axes3[2].plot(d["t"], bias[:, _j], label=f"bias $_{_lbl}$")
    axes3[2].set_ylabel("gyro bias (rad/s)")
    axes3[2].set_xlabel("time (s)")
    axes3[2].legend()
    axes3[2].grid(visible=True)

    fig3.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## LQR variant

    The same plant and estimator, driven by the **`AdaptiveLQR`** instead of quaternion feedback. It acquires nadir from the
    same initial offset.
    """)
    return


@app.cell
def _(Simulation, config, controller_select, d, extract, np, plt):
    if controller_select.value == "Adaptive LQR":
        _qf_sim = Simulation.from_config(config)
        _qf_sim.t_end = 200.0
        _qf_sim.run()
        _d_qf = extract(_qf_sim)

        _d_lqr_plot = d
        _d_qf_plot = _d_qf
    else:
        _tle = [
            "1 25544U 98067A   24001.50000000  .00000000  00000-0  00000-0 2    07",
            "2 25544 097.6000 010.0000 0001000 000.0000 000.0000 15.25000000000009",
        ]
        _lqr_config = dict(config)
        _lqr_config["controller"] = {
            "class_path": "spacecraft.controller.AdaptiveLQR",
            "dt": 0.2,
            "Q": np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).tolist(),
            "R": np.eye(6).tolist(),
            "inertia": config["dynamics"]["inertia"],
            "tle": _tle,
            "epoch": "2024-01-01T12:00:00",
            "reaction_wheels": {"axes": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], "torque_constant": [0.01, 0.01, 0.01]},
            "magnetorquers": {"axes": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], "dipole_constant": [0.3, 0.3, 0.2]},
        }
        _lqr_sim = Simulation.from_config(_lqr_config)
        _lqr_sim.t_end = 200.0
        _lqr_sim.run()
        _d_lqr = extract(_lqr_sim)

        _d_lqr_plot = _d_lqr
        _d_qf_plot = d

    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(_d_qf_plot["t"], np.linalg.norm(_d_qf_plot["euler_err"], axis=1), label="quaternion feedback")
    ax4.plot(_d_lqr_plot["t"], np.linalg.norm(_d_lqr_plot["euler_err"], axis=1), label="LQR")
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("pointing error norm (deg)")
    ax4.set_title("Controller comparison")
    ax4.legend()
    ax4.grid(visible=True)
    fig4.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
