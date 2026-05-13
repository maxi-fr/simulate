from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from simulate.config import load_config
from simulate.logger import Logger, UniversalLog

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from simulate.component import Component
    from simulate.controller import Controller
    from simulate.estimator import Estimator
    from simulate.plant import Plant
    from simulate.reference import Reference
    from simulate.sensor import Sensor


class Simulation:
    """Central orchestrator for the simulation loop."""

    def __init__(  # noqa: PLR0913
        self,
        t_end: float,
        plant: Plant,
        reference: Reference,
        sensor: Sensor,
        estimator: Estimator,
        controller: Controller,
    ) -> None:
        """Initialize the simulation with instantiated components."""
        self.t_end = t_end
        self.plant = plant
        self.reference = reference
        self.sensor = sensor
        self.estimator = estimator
        self.controller = controller
        self.logger = Logger()

        self.dt = self.plant.dt

        base_dt = self.plant.dt

        components: dict[str, Component] = {
            "reference": self.reference,
            "sensor": self.sensor,
            "estimator": self.estimator,
            "controller": self.controller,
        }
        for name, comp in components.items():
            dt = comp.dt
            ratio = dt / base_dt
            if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
                msg = f"{name.capitalize()} dt ({dt}) must be an integer multiple of plant dt ({base_dt})"
                raise ValueError(msg)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Simulation:
        """Instantiate a simulation from a configuration dictionary using dynamic loading."""
        components: dict[str, Any] = {}
        for key in ("plant", "reference", "sensor", "estimator", "controller"):
            comp_config: dict[str, Any] = config[key].copy()
            class_path: str = comp_config.pop("class_path")
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            comp_class: Component = getattr(module, class_name)
            components[key] = comp_class.from_config(comp_config)

        return cls(
            t_end=float(config["t_end"]),
            plant=components["plant"],
            reference=components["reference"],
            sensor=components["sensor"],
            estimator=components["estimator"],
            controller=components["controller"],
        )

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> Simulation:
        """Instantiate a simulation from a YAML configuration file using dynamic loading."""
        config = load_config(filepath)
        return cls.from_config(config)

    def run(self) -> None:
        """Run the simulation loop until t_end."""
        t = 0.0

        u_k: float | np.ndarray = 0.0
        y_k: float | np.ndarray = 0.0

        while t <= self.t_end:
            ref_k, ref_log = self.reference.step(t)

            y_mea, sensor_log = self.sensor.step(t, y_k)

            x_hat, estim_log = self.estimator.step(t, y_mea, u_k)

            u_k, ctrl_log = self.controller.step(t, ref_k, x_hat)

            y_k, plant_log = self.plant.step(t, u_k)

            uni_log = UniversalLog(
                t=t,
                y=y_k,
                y_mea=y_mea,
                x_hat=x_hat,
                u=u_k,
                ref=ref_k,
            )
            comp_logs = {
                "reference": ref_log,
                "plant": plant_log,
                "sensor": sensor_log,
                "estimator": estim_log,
                "controller": ctrl_log,
            }
            self.logger.log(uni_log, comp_logs)

            t += self.dt

    def export_results(self, directory: str | Path, prefix: str = "sim") -> None:
        """Export simulation results via the Logger."""
        self.logger.export_csv(directory, prefix)
        self.logger.export_npz(directory, prefix)
