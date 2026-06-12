# Adaptive Satellite Control Simulation

A modular simulation environment for satellite attitude control, designed to test and analyze various control strategies (e.g., MPC, iLQR) under realistic environmental disturbances.

## Features

*   **Modular Architecture:** Easily swap controllers, actuators, and sensor models.
*   **High-Fidelity Environment:** Includes models for:
    *   Magnetic field (IGRF)
    *   Atmospheric density (NRLMSISE-00)
    *   Solar radiation pressure
    *   Gravity gradient
    *   Aerodynamic drag
*   **Actuator & Sensor Models:** Reaction wheels, magnetorquers, sun sensors, magnetometers, GPS, and gyroscopes with noise and bias simulation.
*   **Logging & Analysis:** Comprehensive state logging with Jupyter Notebook support for post-simulation analysis.

## Prerequisites

*   Python 3.12 (Recommended)
*   `pip`

## Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **IGRF Coefficients Setup (Required):**
    The `pyIGRF` library requires the `igrf14coeffs.txt` file to be present in its installation directory. A helper script is provided to automate this process. Run the following command from the root of the repository:

    ```bash
    python setup_igrf.py
    ```

## Usage

### Configuration
The simulation is configured via `src/simulation_config.json`. You can modify this file to change:
*   **Initial State:** TLE (Two-Line Element), attitude, angular velocity.
*   **Simulation Parameters:** Start time, duration, step size.
*   **Disturbances:** Enable/disable torque and force disturbances.
*   **Satellite components:** Active sensors, actuators, estimators and controllers. And the satellites body properties


### Running the Simulation
To start the simulation, run:

```bash
python src/simulation.py
```

The simulation will log progress to the console and save data to a timestamped folder (e.g., `Simulation_YYYY-MM-DD_HH-MM-SS/`).

### Analysis
To visualize and analyze the results, use the provided Jupyter Notebook:

```bash
jupyter notebook src/analysis.ipynb
```

Update the notebook to point to your specific simulation log folder if necessary.

### For Developers: Extending the Simulation (Work in Progress)
To customize the simulation beyond what is possible through the JSON configuration file, such as introducing new types of Sensors, Actuators, Estimators or Controllers, you will need to extend the codebase. Any new components must be implemented by following the abstraction patterns and interfaces defined by their respective base classes.

To make the simulation highly modular, the physics engine reframes the coupled rotational dynamics of the satellite and its internal components as a generalized, implicit non-linear system. Instead of hardcoding sequential equations of motion for each type of actuator, we build a block matrix equation.

For example, consider a satellite with a single reaction wheel. The wheel has internal electrical dynamics dictating how its current responds to commands (where $T_{\text{cur}}$ is the time constant):

$$
\frac{di}{dt} = \frac{i_{\text{cmd}} - i}{T_{\text{cur}}}
$$

Its rotational dynamics are coupled with the spacecraft body. The acceleration of the wheel relative to the body $\dot{\Omega}_\mathrm{w}$
and the acceleration of the body itself $\dot{\boldsymbol{\omega}}$ projected onto the wheel's axis
$\hat{\mathbf{a}}_\mathrm{w}$ equal the generated torque:

$$\hat{\mathbf{a}}^\top_\mathrm{w} \dot{\boldsymbol{\omega}} + \dot{\Omega}_\mathrm{w} = \frac{\tau_{\mathrm{w}}}{J_{\text{w}}}$$

with $\tau_{\mathrm{w}} = K_\mathrm{w}i_\mathrm{w}$.

Meanwhile, the spacecraft's overall attitude dynamics must account for the external torques and the gyroscopic effects of the spinning wheel:

$$\mathbf{J}\dot{\boldsymbol{\omega}} + J_{\text{w}}\dot{\Omega}_\mathrm{w}\hat{\mathbf{a}}_{\text{w}} = \boldsymbol{\tau}_{\text{ext}} - \boldsymbol{\omega} \times \left(\mathbf{J}\boldsymbol{\omega} + J_{\text{w}}\left(\Omega_{\text{w}}+\hat{\mathbf{a}}^{\top}_{\text{w}}\boldsymbol{\omega}\right)\hat{\mathbf{a}}_{\text{w}} \right)$$

Instead of manually substituting the wheel equations into the body equations to isolate the angular acceleration $\dot{\boldsymbol{\omega}}$, the engine sets up the implicit ODE:

$$L \cdot \begin{bmatrix} \dot{\boldsymbol{\omega}} \\ \dot{x}_\text{act} \end{bmatrix} = \text{rhs}$$

Which, for our example, looks like:

$$
\begin{bmatrix}
\mathbf{J} & J_\mathrm{w}\hat{\mathbf{a}}_\mathrm{w} & \mathbf{0} \\
\hat{\mathbf{a}}^\top_\mathrm{w} & 1 & 0 \\
\mathbf{0} & 0 & 1
\end{bmatrix}
\begin{bmatrix}
\dot{\boldsymbol{\omega}} \\
\dot{\Omega}_\mathrm{w} \\
\frac{di}{dt}
\end{bmatrix}=\begin{bmatrix}
\boldsymbol{\tau}_{\text{ext}} - \boldsymbol{\omega} \times \left(\mathbf{J}\boldsymbol{\omega} + J_{\text{w}}\left(\Omega_{\text{w}}+\hat{\mathbf{a}}^{\top}_{\text{w}}\boldsymbol{\omega}\right)\hat{\mathbf{a}}_{\text{w}} \right) \\
\frac{\tau_{\mathrm{w}}}{J_{\text{w}}}\\
\frac{i_{\text{cmd}} - i}{T_{\text{cur}}}
\end{bmatrix}
$$

This linear system is solved to be able to integrate the state. Any new components (e.g., Magnetorquers) simply append their own rows and columns to this matrix via their `Actuator` interface.

## Project Structure

*   `src/simulation.py`: Main simulation loop and entry point.
*   `src/simulation_config.json`: Configuration file.
*   `src/controllers.py` & `src/controller_models.py`: Implementation of control algorithms (MPC, PI, etc.).
*   `src/dynamics.py` & `src/kinematics.py`: Core physics, orbital mechanics (SGP4), and attitude dynamics.
*   `src/environment.py`: Environmental models (Magnetic field, Sun/Moon position, Atmosphere).
*   `src/actuators.py` & `src/sensors.py`: Hardware models.
