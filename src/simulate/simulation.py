from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from simulate.config import load_config
from simulate.logger import Logger, UniversalLog

if TYPE_CHECKING:
    from pathlib import Path

    from simulate.component import Component
    from simulate.controller import Controller
    from simulate.dynamics import Dynamics
    from simulate.estimator import Estimator
    from simulate.output import Output
    from simulate.reference import Reference
    from simulate.sensor import Sensor


def _to_col_vec(val: float | np.ndarray) -> np.ndarray:
    """Convert a float or array to a 2D column vector ``(N, 1)``."""
    arr = np.atleast_1d(val)
    return arr.reshape((-1, 1)) if arr.ndim == 1 else arr


class Simulation:
    """Central orchestrator for the simulation loop."""

    def __init__(  # noqa: PLR0913
        self,
        t_end: float,
        dynamics: Dynamics,
        outputs: Output | list[Output],
        reference: Reference,
        sensors: Sensor | list[Sensor],
        estimator: Estimator,
        controller: Controller,
    ) -> None:
        """Initialize the simulation with instantiated components.

        ``outputs`` and ``sensors`` are parallel measurement channels: ``sensors[i]`` adds
        noise to the truth produced by ``outputs[i]``. Outputs transform the state at the
        base ``dt`` (always-fresh truth); each sensor may run at its own (slower) rate.
        """
        outputs_list = [outputs] if not isinstance(outputs, list) else outputs
        sensors_list = [sensors] if not isinstance(sensors, list) else sensors

        if len(outputs_list) != len(sensors_list):
            msg = f"outputs ({len(outputs_list)}) and sensors ({len(sensors_list)}) must be the same length"
            raise ValueError(msg)

        self.t_end = t_end
        self.dynamics = dynamics
        self.outputs = outputs_list
        self.reference = reference
        self.sensors = sensors_list
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
            *((f"output_{i}", out) for i, out in enumerate(self.outputs)),
            *((f"sensor_{i}", sen) for i, sen in enumerate(self.sensors)),
        ]
        for name, comp in named_components:
            dt = comp.dt
            ratio = dt / base_dt
            if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
                msg = f"{name.capitalize()} dt ({dt}) must be an integer multiple of plant dt ({base_dt})"
                raise ValueError(msg)

    @staticmethod
    def _build_component(comp_config: dict[str, Any]) -> Any:  # noqa: ANN401
        """Instantiate a single component from a ``{class_path, ...}`` config dict."""
        cfg = comp_config.copy()
        class_path: str = cfg.pop("class_path")
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        comp_class: Component = getattr(module, class_name)
        return comp_class.from_config(cfg)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Simulation:
        """Instantiate a simulation from a configuration dictionary using dynamic loading.

        ``outputs`` and ``sensors`` are lists of ``{class_path, ...}`` dicts (parallel
        measurement channels); the remaining components are single.
        """
        singles = {
            key: cls._build_component(config[key]) for key in ("dynamics", "reference", "estimator", "controller")
        }
        raw_outputs = config["outputs"]
        outputs = (
            [cls._build_component(raw_outputs)]
            if not isinstance(raw_outputs, list)
            else [cls._build_component(c) for c in raw_outputs]
        )

        raw_sensors = config["sensors"]
        sensors = (
            [cls._build_component(raw_sensors)]
            if not isinstance(raw_sensors, list)
            else [cls._build_component(c) for c in raw_sensors]
        )

        return cls(
            t_end=float(config["t_end"]),
            dynamics=singles["dynamics"],
            outputs=outputs,
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

        u_k: float | np.ndarray = 0.0
        y_list: list[float | np.ndarray] = [0.0] * len(self.outputs)

        total_steps = round(self.t_end / self.dt) + 1
        buffer_size = chunk_size if (chunk_size is not None and output_dir is not None) else total_steps
        self.logger.set_buffer_size(buffer_size)

        with tqdm(total=total_steps, desc="Running simulation") as pbar:
            while t <= self.t_end:
                ref_k, ref_log = self.reference.evaluate(t)

                # Each sensor samples (at its own rate, ZOH-held) the previous step's truth.
                sensor_logs = [sensor.evaluate(t, y_list[i]) for i, sensor in enumerate(self.sensors)]
                y_mea_list = [_to_col_vec(res) for res, _ in sensor_logs]
                y_mea = np.vstack(y_mea_list) if y_mea_list else np.zeros((0, 1))

                x_hat, estim_log = self.estimator.evaluate(t, y_mea, u_k)

                u_k, ctrl_log = self.controller.evaluate(t, ref_k, x_hat)

                x_k, dynamics_log = self.dynamics.evaluate(t, u_k)

                # Outputs run at the base dt: always-fresh truth for the next step's sensors.
                output_results = [out.evaluate(t, x_k, u_k) for out in self.outputs]
                y_list = [res for res, _ in output_results]

                if len(self.outputs) == 1:
                    y_val = y_list[0]
                    y_mea_val = sensor_logs[0][0]
                else:
                    y_val = self.dynamics.from_col_vec(np.vstack([_to_col_vec(res) for res in y_list]))
                    y_mea_val = self.dynamics.from_col_vec(y_mea)

                uni_log = UniversalLog(
                    t=t,
                    x=x_k,
                    x_hat=x_hat,
                    u=u_k,
                    ref=ref_k,
                    y=y_val,
                    y_mea=y_mea_val,
                )
                comp_logs: dict[str, Any] = {
                    "reference": ref_log,
                    "dynamics": dynamics_log,
                    "estimator": estim_log,
                    "controller": ctrl_log,
                }
                for i, (_, out_log) in enumerate(output_results):
                    comp_logs[f"output_{i}"] = out_log
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
