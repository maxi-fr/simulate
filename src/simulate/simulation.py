from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from .config import build_component, load_config
from .logger import Logger, UniversalLog

if TYPE_CHECKING:
    from pathlib import Path

    from .component import Component
    from .controller import Controller
    from .dynamics import Dynamics
    from .estimator import Estimator
    from .reference import Reference
    from .sensor import Sensor


class Simulation:
    """Central orchestrator for the simulation loop."""

    def __init__(  # noqa: PLR0913
        self,
        t_end: float,
        dynamics: Dynamics,
        reference: Reference,
        sensors: Sensor | list[Sensor],
        estimator: Estimator,
        controller: Controller,
    ) -> None:
        """Initialize the simulation with instantiated components.

        Each sensor owns a measurement model (deterministic truth ``y = h(t, x, u)``) and
        adds noise on top, sampling at its own ``dt``. ``sensors`` may be a single sensor or
        a list of independent measurement channels.
        """
        sensors_list = [sensors] if not isinstance(sensors, list) else sensors

        self.t_end = t_end
        self.dynamics = dynamics
        self.reference = reference
        self.sensors: list[Sensor] = sensors_list  # ty:ignore[invalid-assignment]
        self.estimator = estimator
        self.controller = controller
        self.logger = Logger()

        self.dt = self.dynamics.dt

        base_dt = self.dynamics.dt

        named_components: list[tuple[str, Component]] = [
            ("dynamics", self.dynamics),
            ("reference", self.reference),
            ("estimator", self.estimator),
            ("controller", self.controller),
            *((f"sensor_{i}", sen) for i, sen in enumerate(self.sensors)),
        ]
        for name, comp in named_components:
            dt = comp.dt
            ratio = dt / base_dt
            if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
                msg = f"{name.capitalize()} dt ({dt}) must be an integer multiple of plant dt ({base_dt})"
                raise ValueError(msg)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Simulation:
        """Instantiate a simulation from a configuration dictionary using dynamic loading.

        ``sensors`` is a single ``{class_path, ...}`` dict or a list of them (independent
        measurement channels); each carries a nested ``measurement`` model. The remaining
        components are single.

        Returns
        -------
        Simulation
            The simulation assembled from the configuration.
        """
        singles = {key: build_component(config[key]) for key in ("dynamics", "reference", "estimator", "controller")}

        raw_sensors = config["sensors"]
        sensors = (
            [build_component(raw_sensors)]
            if not isinstance(raw_sensors, list)
            else [build_component(c) for c in raw_sensors]
        )

        return cls(
            t_end=float(config["t_end"]),
            dynamics=singles["dynamics"],
            reference=singles["reference"],
            sensors=sensors,
            estimator=singles["estimator"],
            controller=singles["controller"],
        )

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> Simulation:
        """Instantiate a simulation from a YAML configuration file using dynamic loading."""
        config = load_config(filepath)
        return cls.from_config(config)

    def run(
        self,
        output_dir: str | Path | None = None,
        prefix: str = "sim",
        chunk_size: int | None = 10_000,
        *,
        compress: bool = False,
    ) -> None:
        """Run the simulation loop until t_end."""
        self.logger.compress = compress
        t = 0.0
        step_count: int = 0

        u_k: float | np.ndarray = np.zeros(self.dynamics.n_inputs)

        total_steps = round(self.t_end / self.dt) + 1
        buffer_size = chunk_size if (chunk_size is not None and output_dir is not None) else total_steps
        self.logger.set_buffer_size(buffer_size)

        with tqdm(total=total_steps, desc="Running simulation") as pbar:
            while t <= self.t_end:
                x_k = self.dynamics.x

                ref_k, ref_log = self.reference.evaluate(t)

                sensor_logs = [sensor.evaluate(t, x_k, u_k) for sensor in self.sensors]
                y_mea_list = [np.atleast_1d(res) for res, _ in sensor_logs]
                y_mea = np.concatenate(y_mea_list) if y_mea_list else np.zeros(0)

                x_hat, estim_log = self.estimator.evaluate(t, y_mea, u_k)

                u_k, ctrl_log = self.controller.evaluate(t, ref_k, x_hat)

                # Advance the plant; ``self.dynamics.x`` becomes the next step's state.
                _, dynamics_log = self.dynamics.evaluate(t, u_k)

                y_mea_val = sensor_logs[0][0] if len(self.sensors) == 1 else y_mea

                uni_log = UniversalLog(
                    t=t,
                    x=x_k,
                    x_hat=x_hat,
                    u=u_k,
                    ref=ref_k,
                    y_mea=y_mea_val,
                )
                comp_logs: dict[str, Any] = {
                    "reference": ref_log,
                    "dynamics": dynamics_log,
                    "estimator": estim_log,
                    "controller": ctrl_log,
                }
                for i, (_, sen_log) in enumerate(sensor_logs):
                    comp_logs[f"sensor_{i}"] = sen_log
                self.logger.log(uni_log, comp_logs)
                step_count += 1

                if output_dir is not None and chunk_size is not None and step_count % chunk_size == 0:
                    self.logger.flush_chunk(output_dir, prefix)

                t += self.dt
                pbar.update(1)

    def export_results(self, directory: str | Path, prefix: str = "sim", *, compress: bool = False) -> None:
        """Flush remaining in-memory data then merge all chunks into {prefix}.npz."""
        self.logger.flush_chunk(directory, prefix, compress=compress)
        Logger.merge_chunks(directory, prefix, compress=compress)
