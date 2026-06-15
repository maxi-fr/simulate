# mypy: ignore-errors

import marimo

__generated_with = "0.23.6"
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
    import polars as pl

    from simulate.simulation import Simulation

    return Simulation, mo, pl, plt


@app.cell
def _(mo):
    mo.md(r"""
    # DC Motor Speed Control

    This notebook demonstrates how to use the simulation framework to model and control a DC Motor. We will:
    1. Implement a custom continuous-time `DCMotorDynamics` and `DCMotorOutput`.
    2. Configure a simulation with a PID controller, Gaussian sensor noise, and a step reference.
    3. Run the simulation and visualize the results.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The DC Motor Plant

    The DC motor is modeled by the following differential equations:
    - $L \frac{di}{dt} = -Ri - K_e \omega + V$
    - $J \frac{d\omega}{dt} = K_t i - b \omega$

    Where:
    - $x = [\omega, i]^T$ is the state vector (speed and armature current).
    - $u = V$ is the control input (voltage).
    - $y = \omega$ is the measured output.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Tuning Controls
    """)
    return


@app.cell
def _(mo):
    kp = mo.ui.slider(start=0.0, stop=1.0, step=0.1, value=1.0, label="kp")
    ki = mo.ui.slider(start=0.0, stop=2.0, step=0.1, value=2.0, label="ki")
    kd = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.0, label="kd")
    step_value = mo.ui.slider(start=0.0, stop=200.0, step=1.0, value=100.0, label="step_value (rad/s)")
    start_time = mo.ui.slider(start=0.0, stop=2.0, step=0.05, value=0.5, label="start_time (s)")
    mo.vstack([kp, ki, kd, step_value, start_time])
    return kd, ki, kp, start_time, step_value


@app.cell
def _(kd, ki, kp, start_time, step_value):
    config = {
        "t_end": 2.0,
        "dynamics": {
            "class_path": "dc_motor.DCMotorDynamics",
            "dt": 0.001,
            "R": 1.0,
            "L": 0.01,
            "Ke": 0.05,
            "Kt": 0.05,
            "J": 0.001,
            "b": 0.001,
            "integrator": "simulate.integrator.rk4",
        },
        "reference": {
            "class_path": "simulate.reference.StepReference",
            "dt": 0.001,
            "step_value": float(step_value.value),
            "start_time": float(start_time.value),
        },
        "sensors": {
            "class_path": "simulate.sensor.GaussianSensor",
            "dt": 0.001,
            "std_dev": 0.1,
            "measurement": {"class_path": "dc_motor.dc_motor_measurement"},
        },
        "estimator": {
            "class_path": "simulate.estimator.IdentityEstimator",
            "dt": 0.001,
        },
        "controller": {
            "class_path": "simulate.controller.PIDController",
            "dt": 0.001,
            "kp": [[float(kp.value)]],
            "ki": [[float(ki.value)]],
            "kd": [[float(kd.value)]],
        },
    }
    return (config,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Running the Simulation
    """)
    return


@app.cell
def _(Simulation, config):
    sim = Simulation.from_config(config)
    sim.run()
    return (sim,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Visualizing Results
    """)
    return


@app.cell
def _(pl, plt, sim):
    data = pl.DataFrame(sim.logger.universal_logs)
    dynamics_data = pl.DataFrame(sim.logger.component_logs["dynamics"])
    # The single channel logs the true speed (sensor_0 `truth`); the universal log carries the
    # noisy measurement (`y_mea`).
    sensor_data = pl.DataFrame(sim.logger.component_logs["sensor_0"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(data["t"], data["ref"], "k--", label="Reference (rad/s)")
    axes[0].plot(sensor_data["t"], sensor_data["truth"], "b-", label="Actual Speed (rad/s)")
    axes[0].plot(data["t"], data["y_mea"], "r.", alpha=0.1, label="Measured Speed (rad/s)")
    axes[0].set_title("DC Motor Speed Control")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].legend()
    axes[0].grid(visible=True)

    axes[1].plot(data["t"], data["u"], "m-", label="Control Input (V)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].legend()
    axes[1].grid(visible=True)

    axes[2].plot(data["t"], dynamics_data["current"], "g-", label="Armature Current (A)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Current (A)")
    axes[2].legend()
    axes[2].grid(visible=True)

    fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
