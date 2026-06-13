# mypy: ignore-errors

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from simulate.sensor import GaussianSensor
    from spacecraft.effector import EarthGravity, ReactionWheelArray, Wrench
    from spacecraft.rigid_body import (
        ReactionWheelTelemetry,
        RigidBodyDynamics,
        rigid_body_attitude,
        rigid_body_rate,
    )

    return (
        EarthGravity,
        GaussianSensor,
        ReactionWheelArray,
        ReactionWheelTelemetry,
        RigidBodyDynamics,
        Wrench,
        mo,
        np,
        pl,
        plt,
        rigid_body_attitude,
        rigid_body_rate,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Rigid Body Dynamics with Modular Actuators

    This notebook demonstrates the pre-built `RigidBodyDynamics` component and the modular
    `Effector` abstraction. We compose two actuators into one body:

    1. A **`Wrench`** applying an inertial-frame thrust (translation), and
    2. A **`ReactionWheelArray`** with 3 orthogonal reaction wheels.

    We step the dynamics open-loop and verify that spinning up the wheel rotates the body
    while the **total angular momentum** $H = J\omega + h$ is conserved.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Build the rigid body

    State layout: $x = [\,r(3)\;|\;v(3)\;|\;q(4)\;|\;\omega(3)\;|\;h_w\,]$, with $q$ a
    scalar-last inertial$\to$body unit quaternion. Attitude integration uses
    `QuaternionRK4`, which renormalizes the quaternion after each step.
    """)
    return


@app.cell
def _(ReactionWheelArray, RigidBodyDynamics, Wrench, np):
    dt = 0.01
    inertia = np.diag([1.0, 2.0, 3.0])
    rw_array = ReactionWheelArray(
        axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        inertia=0.05,
        torque_constant=0.08,
        time_constant=0.04,
        max_current=2.5,
    )
    dynamics = RigidBodyDynamics(
        dt=dt,
        mass=5.0,
        inertia=inertia,
        effectors=[Wrench(), rw_array],
    )
    return dt, dynamics, inertia


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Step the simulation open-loop

    Command layout is `[Fx, Fy, Fz, tau_x, tau_y, tau_z, i_cmd_x, i_cmd_y, i_cmd_z]`. We apply a constant
    body-x thrust and a constant z-wheel current command for 5 s.
    """)
    return


@app.cell
def _(dt, dynamics, inertia, np):
    t_end = 5.0
    n_steps = int(t_end / dt)

    fx = 2.0  # body-frame thrust (N)
    i_cmd_z = 1.25  # z-wheel commanded current (A)
    cmd = np.array([fx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, i_cmd_z])

    rows = []
    for _k in range(n_steps):
        # Record the current state (x_0 on the first iteration) before advancing.
        omega = dynamics.x[10:13]
        # currents: x[13:16], omega_rel: x[16:19]
        h_wheels = 0.05 * (dynamics.x[16:19] + omega)
        total_h = inertia @ omega + h_wheels
        h_w_z = h_wheels[2]
        rows.append(
            {
                "t": _k * dt,
                "x": dynamics.x[0],
                "qw": dynamics.x[9],
                "qz": dynamics.x[8],
                "wz": omega[2],
                "h_w": h_w_z,
                "H_norm": float(np.linalg.norm(total_h)),
            }
        )
        dynamics.evaluate(_k * dt, cmd)
    return (rows,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Visualize

    The body counter-rotates as the wheel spins up ($\omega_z < 0$, $h_w > 0$), the body
    translates under thrust, and $\|H\|$ stays at zero — confirming momentum exchange with
    no external torque.
    """)
    return


@app.cell
def _(pl, plt, rows):
    data = pl.DataFrame(rows)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    axes[0].plot(data["t"], data["x"], "b-", label="position x (m)")
    axes[0].set_ylabel("x (m)")
    axes[0].legend()
    axes[0].grid(visible=True)

    axes[1].plot(data["t"], data["wz"], "r-", label="body $\\omega_z$ (rad/s)")
    axes[1].set_ylabel("rad/s")
    axes[1].legend()
    axes[1].grid(visible=True)

    axes[2].plot(data["t"], data["h_w"], "g-", label="wheel momentum $h_w$")
    axes[2].set_ylabel("N*m*s")
    axes[2].legend()
    axes[2].grid(visible=True)

    axes[3].plot(data["t"], data["H_norm"], "k-", label="$\\|H\\|$ total angular momentum")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("N*m*s")
    axes[3].legend()
    axes[3].grid(visible=True)

    fig.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. An environmental effector: gravity-gradient torque

    Environmental effects are just **command-free effectors** (`n_inputs = 0`) — autonomous
    functions of the body state rather than the control input. We release a gravity-gradient
    stabilized body with a small attitude tilt and watch it librate. The effector receives
    the body's inertia automatically via `bind`, so there is no commanded input.
    """)
    return


@app.cell
def _(EarthGravity, RigidBodyDynamics, np):
    dt_gg = 1.0
    body = RigidBodyDynamics(
        dt=dt_gg,
        mass=500.0,
        inertia=np.diag([100.0, 200.0, 300.0]),
        effectors=[EarthGravity(mu=3.986e14)],
    )
    body.x[0:3] = np.array([7.0e6, 0.0, 0.0])  # LEO radius along inertial x
    half = np.deg2rad(10.0)  # initial tilt about body z
    body.x[6:10] = np.array([0.0, 0.0, np.sin(half), np.cos(half)])

    gg_rows = []
    for _k in range(6000):
        gg_rows.append({"t": _k * dt_gg, "wz": body.x[12], "qz": body.x[8]})
        body.evaluate(_k * dt_gg, np.zeros(0))
    return (gg_rows,)


@app.cell
def _(gg_rows, pl, plt):
    gg_data = pl.DataFrame(gg_rows)
    gg_fig, gg_axes = plt.subplots(2, 1, figsize=(12, 6))
    gg_axes[0].plot(gg_data["t"], gg_data["qz"], "b-", label="$q_z$ (attitude)")
    gg_axes[0].set_ylabel("$q_z$")
    gg_axes[0].legend()
    gg_axes[0].grid(visible=True)
    gg_axes[1].plot(gg_data["t"], gg_data["wz"], "r-", label="body $\\omega_z$ (rad/s)")
    gg_axes[1].set_xlabel("Time (s)")
    gg_axes[1].set_ylabel("rad/s")
    gg_axes[1].legend()
    gg_axes[1].grid(visible=True)
    gg_fig.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Measuring the body: per-part measurement models + Sensors

    Measurement is modular. Each **`Sensor`** owns a **measurement model** — a callable
    `(t, x, u) -> y` that transforms the state into a true sub-measurement — and adds noise at
    *its own* sampling rate. Here a slow **star tracker** reads the attitude $q$,
    a fast **gyro** reads the body rate $\omega$, and a **wheel tachometer** reads the
    reaction-wheel momentum $h_w$ — each at a different rate.

    (Note: additive noise on $q$ yields a non-unit quaternion; a real pipeline renormalizes.)
    """)
    return


@app.cell
def _(
    GaussianSensor,
    ReactionWheelArray,
    ReactionWheelTelemetry,
    RigidBodyDynamics,
    Wrench,
    np,
    rigid_body_attitude,
    rigid_body_rate,
):
    dt_m = 0.01
    rw_array_m = ReactionWheelArray(
        axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        inertia=0.05,
        torque_constant=0.08,
        time_constant=0.04,
        max_current=2.5,
    )
    body_m = RigidBodyDynamics(
        dt=dt_m,
        mass=5.0,
        inertia=np.diag([1.0, 2.0, 3.0]),
        effectors=[Wrench(), rw_array_m],
    )

    # Each sensor owns a measurement model (truth) and adds noise, sampling at its own rate.
    gyro_sen = GaussianSensor(dt=dt_m, measurement=rigid_body_rate, std_dev=2e-3)
    track_sen = GaussianSensor(dt=10 * dt_m, measurement=rigid_body_attitude, std_dev=2e-3)
    tach_sen = GaussianSensor(dt=5 * dt_m, measurement=ReactionWheelTelemetry(index=18), std_dev=1e-2)

    cmd_m = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.25])
    meas_rows = []
    for _k in range(500):
        t_m = _k * dt_m
        # Measure the current state (x_0 on the first iteration) before advancing the plant.
        x_m = body_m.x

        wz_mea, gyro_log = gyro_sen.evaluate(t_m, x_m, cmd_m)
        hw_mea, tach_log = tach_sen.evaluate(t_m, x_m, cmd_m)
        q_mea, track_log = track_sen.evaluate(t_m, x_m, cmd_m)
        wz_true, hw_true, q_true = gyro_log.truth, tach_log.truth, track_log.truth

        meas_rows.append(
            {
                "t": t_m,
                "wz_true": np.ravel(wz_true)[2],
                "wz_mea": np.ravel(wz_mea)[2],
                "hw_true": float(np.ravel(hw_true)[0]),
                "hw_mea": float(np.ravel(hw_mea)[0]),
                "qz_true": np.ravel(q_true)[3],
                "qz_mea": np.ravel(q_mea)[3],
            }
        )

        body_m.evaluate(t_m, cmd_m)
    return (meas_rows,)


@app.cell
def _(meas_rows, pl, plt):
    meas_data = pl.DataFrame(meas_rows)
    meas_fig, meas_axes = plt.subplots(3, 1, figsize=(12, 9))

    meas_axes[0].plot(meas_data["t"], meas_data["wz_true"], "b-", label="gyro $\\omega_z$ truth")
    meas_axes[0].plot(meas_data["t"], meas_data["wz_mea"], "r.", alpha=0.3, label="gyro measured")
    meas_axes[0].set_ylabel("rad/s")
    meas_axes[0].legend()
    meas_axes[0].grid(visible=True)

    meas_axes[1].plot(meas_data["t"], meas_data["hw_true"], "g-", label="wheel relative speed $\\omega_{z, rel}$ truth")
    meas_axes[1].plot(meas_data["t"], meas_data["hw_mea"], "r.", alpha=0.3, label="tachometer measured (5x slower)")
    meas_axes[1].set_ylabel("rad/s")
    meas_axes[1].legend()
    meas_axes[1].grid(visible=True)

    meas_axes[2].plot(meas_data["t"], meas_data["qz_true"], "b-", label="star tracker $q_z$ truth")
    meas_axes[2].plot(meas_data["t"], meas_data["qz_mea"], "r.", alpha=0.3, label="measured (10x slower)")
    meas_axes[2].set_xlabel("Time (s)")
    meas_axes[2].set_ylabel("$q_z$")
    meas_axes[2].legend()
    meas_axes[2].grid(visible=True)

    meas_fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
