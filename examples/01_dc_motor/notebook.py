# mypy: ignore-errors

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    # Add workspace root to sys.path to allow importing 'examples' package
    workspace_root = str(Path(__file__).resolve().parents[2])
    if workspace_root not in sys.path:
        sys.path.append(workspace_root)

    from simulate.simulation import Simulation

    return Path, Simulation, mo, np, plt, yaml


@app.cell
def _(mo):
    mo.md(r"""
    # DC Motor Speed Control

    This notebook demonstrates how to use the simulation framework to model and control a DC Motor. We will:
    1. Implement a custom continuous-time `DCMotorDynamics` whose only measurement is the speed $\omega$.
    2. Reconstruct the full state $[\omega, i]$ with a model-based `LuenbergerObserver`.
    3. Control the speed with a `PIController`, using the observed current as the damping (derivative) term.
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
    - $y = \omega$ is the **only** measured output — the current $i$ is *not* measured.

    Because only $\omega$ is measured, a `LuenbergerObserver` reconstructs the full state from the motor
    model. The observed current carries $\dot{\omega}$, so feeding it back through a column of the
    proportional gain provides damping in place of a derivative term — without differentiating the noisy
    measurement.
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
    kp = mo.ui.slider(start=0.0, stop=2.0, step=0.1, value=1.0, label="kp (speed)")
    kp_i = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.2, label="kp_i (current / damping)")
    ki = mo.ui.slider(start=0.0, stop=5.0, step=0.1, value=2.0, label="ki (speed integral)")
    step_value = mo.ui.slider(start=0.0, stop=200.0, step=1.0, value=100.0, label="step_value (rad/s)")
    start_time = mo.ui.slider(start=0.0, stop=2.0, step=0.05, value=0.5, label="start_time (s)")
    mo.vstack([kp, kp_i, ki, step_value, start_time])
    return ki, kp, kp_i, start_time, step_value


@app.cell
def _(Path, ki, kp, kp_i, start_time, step_value, yaml):
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Overwrite slider-dependent values
    config["reference"]["step_value"] = [float(step_value.value), 0.0]
    config["reference"]["start_time"] = float(start_time.value)
    config["controller"]["kp"] = [[float(kp.value), float(kp_i.value)]]
    config["controller"]["ki"] = [[float(ki.value), 0.0]]
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

    The bottom panel shows the observer reconstructing the **unmeasured** armature current from the speed
    measurement alone — this estimate is what supplies the controller's damping term.
    """)
    return


@app.cell
def _(np, plt, sim):
    logs = sim.logger.core_logs
    t = np.array([row["t"] for row in logs])
    x = np.array([np.asarray(row["x"]) for row in logs])  # true [omega, i]
    x_hat = np.array([np.asarray(row["x_hat"]) for row in logs])  # observed [omega, i]
    u = np.array([np.asarray(row["u"]) for row in logs])
    ref = np.array([np.asarray(row["ref"]) for row in logs])
    y_mea = np.array([np.atleast_1d(row["y_mea"]) for row in logs])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(t, ref[:, 0], "k--", label="Reference (rad/s)")
    axes[0].plot(t, x[:, 0], "b-", label="Actual Speed (rad/s)")
    axes[0].plot(t, y_mea[:, 0], "r.", alpha=0.1, label="Measured Speed (rad/s)")
    axes[0].plot(t, x_hat[:, 0], "c-", lw=1, label="Observed Speed (rad/s)")
    axes[0].set_title("DC Motor Speed Control")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].legend()
    axes[0].grid(visible=True)

    axes[1].plot(t, u[:, 0], "m-", label="Control Input (V)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].legend()
    axes[1].grid(visible=True)

    axes[2].plot(t, x[:, 1], "g-", label="Actual Current (A)")
    axes[2].plot(t, x_hat[:, 1], "y--", label="Observed Current (A)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Current (A)")
    axes[2].legend()
    axes[2].grid(visible=True)

    fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
