import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from simulate.config import load_config
from simulate.controller import Controller
from simulate.estimator import Estimator
from simulate.logger import Logger, UniversalLog
from simulate.plant import Plant
from simulate.sensor import Sensor


class Simulation:
    """Central orchestrator for the simulation loop."""

    def __init__(
        self,
        t_end: float,
        plant: Plant[Any],
        sensor: Sensor[Any],
        estimator: Estimator[Any],
        controller: Controller[Any],
    ) -> None:
        """Initialize the simulation with instantiated components."""
        self.t_end = t_end
        self.plant = plant
        self.sensor = sensor
        self.estimator = estimator
        self.controller = controller
        self.logger = Logger()

        # The base tick is dictated by the plant's update period
        self.dt = self.plant.dt

        # Multi-rate timing validation
        base_dt = self.plant.dt
        for name, comp in [
            ("sensor", self.sensor),
            ("estimator", self.estimator),
            ("controller", self.controller),
        ]:
            dt = comp.dt
            ratio = dt / base_dt
            if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
                msg = f"{name.capitalize()} dt ({dt}) must be an integer multiple of plant dt ({base_dt})"
                raise ValueError(msg)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "Simulation":
        """Instantiate a simulation from a YAML configuration file using dynamic loading."""
        config = load_config(filepath)

        # 1. Instantiate Components dynamically
        components: dict[str, Any] = {}
        for key in ["plant", "sensor", "estimator", "controller"]:
            comp_config = config[key]
            class_path = comp_config.pop("class_path")
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            comp_class = getattr(module, class_name)
            components[key] = comp_class.from_config(comp_config)

        return cls(
            t_end=float(config["t_end"]),
            plant=components["plant"],
            sensor=components["sensor"],
            estimator=components["estimator"],
            controller=components["controller"],
        )

    def generate_reference(self, t: float) -> float | np.ndarray:
        """
        Generate the reference signal for the current time.

        In a full implementation, this might be a separate component.
        For now, we provide a simple step response scalar.
        """
        val = 1.0 if t >= 0.5 else 0.0  # noqa: PLR2004
        return float(val)

    def run(self) -> None:
        """Run the simulation loop until t_end."""
        t = 0.0

        # Initial states - using floats for SISO cases
        u_k: float | np.ndarray = 0.0
        y_k: float | np.ndarray = 0.0

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
                y=y_k,
                y_mea=y_mea,
                x_hat=x_hat,
                u=u_k,
                ref=ref_k,
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
