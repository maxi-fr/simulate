from pathlib import Path
from typing import Any

import numpy as np

from simulate.config import (
    ControllerConfig,
    EstimatorConfig,
    PlantConfig,
    SensorConfig,
    SimulationConfig,
)
from simulate.controller import Controller
from simulate.estimator import Estimator
from simulate.logger import Logger, UniversalLog
from simulate.plant import Plant
from simulate.sensor import Sensor


class Simulation[P: PlantConfig, S: SensorConfig, E: EstimatorConfig, C: ControllerConfig]:
    """Central orchestrator for the simulation loop."""

    def __init__(
        self,
        config: SimulationConfig[P, S, E, C],
        plant: Plant[Any],
        sensor: Sensor[Any],
        estimator: Estimator[Any],
        controller: Controller[Any],
    ) -> None:
        """Initialize the simulation with the given configuration and instantiated components."""
        self.config = config
        self.plant = plant
        self.sensor = sensor
        self.estimator = estimator
        self.controller = controller
        self.logger = Logger()

        # The base tick is dictated by the plant's update period
        self.dt = self.config.plant.dt
        self.t_end = self.config.t_end

    def generate_reference(self, t: float) -> np.ndarray:
        """
        Generate the reference signal for the current time.

        In a full implementation, this might be a separate component.
        For now, we provide a simple step response scalar wrapped in a 1D array.
        """
        val = 1.0 if t >= 0.5 else 0.0  # noqa: PLR2004
        return np.array([val])

    def run(self) -> None:
        """Run the simulation loop until t_end."""
        t = 0.0

        # Initial states
        u_k = np.array([0.0])
        y_k = np.array([0.0])

        while t <= self.t_end:
            # 1. Reference Generation
            ref_k = self.generate_reference(t)

            # 2. Measurement
            # Read the Plant's output via the Sensor at time t
            y_mea, sensor_log = self.sensor.step(t, y_k)

            # 3. Estimation
            # Calculate the state estimate using the measurement and previous input
            x_hat, estim_log = self.estimator.step(t, y_mea, u_k)

            # 4. Control
            # Controller steps, using ref and x_hat.
            u_k, ctrl_log = self.controller.step(t, ref_k, x_hat)

            # 5. Actuation
            # Plant steps, using u_k.
            y_k, plant_log = self.plant.step(t, u_k)

            # 6. Logging
            uni_log = UniversalLog(
                t=t,
                y=y_k.flatten(),
                y_mea=y_mea.flatten(),
                x_hat=x_hat.flatten(),
                u=u_k.flatten(),
                ref=ref_k.flatten(),
            )
            comp_logs = {
                "plant": plant_log,
                "sensor": sensor_log,
                "estimator": estim_log,
                "controller": ctrl_log,
            }
            self.logger.log(uni_log, comp_logs)

            # Advance time
            t += self.dt

    def export_results(self, directory: str | Path, prefix: str = "sim") -> None:
        """Export simulation results via the Logger."""
        self.logger.export_csv(directory, prefix)
        self.logger.export_npz(directory, prefix)
