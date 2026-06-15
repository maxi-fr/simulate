# mypy: ignore-errors

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    # Add workspace root to sys.path to allow importing 'examples' package
    workspace_root = str(Path(__file__).resolve().parents[2])
    if workspace_root not in sys.path:
        sys.path.append(workspace_root)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from simulate.simulation import Simulation
    from spacecraft.signals import STATE

    return Path, STATE, Simulation, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Quadrocopter Dynamics with Modular Actuators

    This notebook demonstrates a quadrocopter simulation using the pre-built `RigidBodyDynamics` component
    and the modular `Effector` abstraction. We compose two effectors onto our rigid body:

    1. A **`FlatGravity`** effector applying constant gravitational acceleration in the inertial frame, and
    2. A **`Quadrocopter`** effector modeling 4 rotors that produce body-frame thrust and drag torque.

    We can simulate open-loop pitch control, aerodynamic drag damping, and sensor measurements.
    """)
    return


@app.cell
def _(mo):
    sim_select = mo.ui.dropdown(
        options=["Open Loop Pitch", "Aerodynamic Drag", "Sensors"],
        value="Open Loop Pitch",
        label="Simulation Case",
    )
    sim_select
    return (sim_select,)


@app.cell
def _(Path, Simulation, sim_select):
    # Map selection to configuration file
    filename_map = {
        "Open Loop Pitch": "open_loop.yaml",
        "Aerodynamic Drag": "drag.yaml",
        "Sensors": "sensors.yaml",
    }
    config_path = Path(__file__).resolve().parent / filename_map[sim_select.value]
    sim = Simulation.from_yaml(config_path)
    sim.run()
    return (sim,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization
    """)
    return


@app.cell
def _(STATE, np, plt, sim, sim_select):
    if sim_select.value == "Open Loop Pitch":
        logs = sim.logger.universal_logs
        t = np.array([row["t"] for row in logs])
        x = np.array([np.asarray(row["x"]) for row in logs])

        fig, axes = plt.subplots(3, 1, figsize=(12, 9))

        axes[0].plot(t, x[:, STATE.r][:, 0], "b-", label="position x (m)")
        axes[0].plot(t, x[:, STATE.r][:, 2], "g-", label="position z (m)")
        axes[0].set_ylabel("Position (m)")
        axes[0].legend()
        axes[0].grid(visible=True)
        axes[0].set_title("Open Loop Pitch Maneuver")

        axes[1].plot(t, x[:, STATE.omega][:, 1], "r-", label="pitch rate $\\omega_y$ (rad/s)")
        axes[1].plot(t, x[:, STATE.omega][:, 2], "k-", label="yaw rate $\\omega_z$ (rad/s)")
        axes[1].set_ylabel("Angular rate (rad/s)")
        axes[1].legend()
        axes[1].grid(visible=True)

        axes[2].plot(t, x[:, STATE.q][:, 3], "b-", label="$q_w$ (scalar)")
        axes[2].plot(t, x[:, STATE.q][:, 2], "g-", label="$q_z$ (yaw component)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Quaternion")
        axes[2].legend()
        axes[2].grid(visible=True)

        fig.tight_layout()
        plt.gca()

    elif sim_select.value == "Aerodynamic Drag":
        logs = sim.logger.universal_logs
        t = np.array([row["t"] for row in logs])
        x = np.array([np.asarray(row["x"]) for row in logs])

        drag_fig, drag_axes = plt.subplots(2, 1, figsize=(12, 6))

        drag_axes[0].plot(t, x[:, STATE.v][:, 0], "b-", label="velocity x (m/s)")
        drag_axes[0].set_ylabel("v_x (m/s)")
        drag_axes[0].legend()
        drag_axes[0].grid(visible=True)
        drag_axes[0].set_title("Aerodynamic Drag Damping")

        drag_axes[1].plot(t, x[:, STATE.omega][:, 1], "r-", label="pitch rate $\\omega_y$ (rad/s)")
        drag_axes[1].set_xlabel("Time (s)")
        drag_axes[1].set_ylabel("rad/s")
        drag_axes[1].legend()
        drag_axes[1].grid(visible=True)

        drag_fig.tight_layout()
        plt.gca()

    elif sim_select.value == "Sensors":
        uni_logs = sim.logger.universal_logs
        t = np.array([row["t"] for row in uni_logs])
        x = np.array([np.asarray(row["x"]) for row in uni_logs])
        y_mea = np.array([np.asarray(row["y_mea"]) for row in uni_logs])

        meas_fig, meas_axes = plt.subplots(3, 1, figsize=(12, 9))

        meas_axes[0].plot(t, x[:, STATE.omega][:, 2], "b-", label="gyro $\\omega_z$ truth")
        meas_axes[0].plot(t, y_mea[:, 2], "r.", alpha=0.3, label="gyro measured")
        meas_axes[0].set_ylabel("rad/s")
        meas_axes[0].legend()
        meas_axes[0].grid(visible=True)
        meas_axes[0].set_title("Sensor Measurements vs. Truth")

        meas_axes[1].plot(t, x[:, STATE.r][:, 0], "g-", label="position x truth")
        meas_axes[1].plot(t, y_mea[:, 3], "r.", alpha=0.3, label="GPS measured (5x slower)")
        meas_axes[1].set_ylabel("m")
        meas_axes[1].legend()
        meas_axes[1].grid(visible=True)

        meas_axes[2].plot(t, x[:, STATE.q][:, 3], "b-", label="star tracker $q_w$ truth")
        meas_axes[2].plot(t, y_mea[:, 9], "r.", alpha=0.3, label="measured (10x slower)")
        meas_axes[2].set_xlabel("Time (s)")
        meas_axes[2].set_ylabel("$q_w$")
        meas_axes[2].legend()
        meas_axes[2].grid(visible=True)

        meas_fig.tight_layout()
        plt.gca()
    return


if __name__ == "__main__":
    app.run()
