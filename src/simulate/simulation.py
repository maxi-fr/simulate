from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from .config import build_component, load_config
from .logger import BaseLogger, CoreLog, create_logger

if TYPE_CHECKING:
    from pathlib import Path

    from .component import Component
    from .controller import Controller
    from .dynamics import Dynamics
    from .estimator import Estimator
    from .reference import Reference
    from .sensor import Sensor


def _time_unit(seconds: float) -> tuple[float, str]:
    """Pick a human-friendly unit for a duration given in seconds.

    Parameters
    ----------
    seconds : float
        The duration to display, in seconds.

    Returns
    -------
    divisor : float
        Factor to convert seconds into the chosen unit.
    label : str
        tqdm unit label for the chosen unit.
    """
    if seconds >= 2 * 3600:
        return 3600.0, "sim h"
    if seconds >= 2 * 60:
        return 60.0, "sim min"
    return 1.0, "sim s"


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
        # The storage backend depends on run()'s output_dir, so the logger is built there.
        self.logger: BaseLogger | None = None

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
        prefix: str = "log",
        *,
        use_mmap: bool = False,
        compress: bool = False,
    ) -> None:
        """Run the simulation loop until t_end.

        When ``use_mmap`` is True, each signal is logged straight into a
        memory-mapped ``.npy`` file (sized to the known step count), so resident
        memory stays bounded for runs of any length. Call :meth:`export_results` to
        pack them into ``{prefix}.npz``. With ``use_mmap=False`` the logs are kept
        in RAM and exposed via ``self.logger.core_logs`` / ``component_logs``.
        """
        u_k: np.ndarray = np.zeros(self.dynamics.n_inputs)

        total_steps = round(self.t_end / self.dt) + 1
        self.logger = create_logger(total_steps, directory=output_dir, prefix=prefix, use_mmap=use_mmap, compress=compress)

        divisor, unit = _time_unit(self.t_end)
        bar_format = "{l_bar}{bar}| {n:.1f}/{total:.1f} {unit} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        with tqdm(total=self.t_end / divisor, desc="Simulation time", unit=unit, bar_format=bar_format) as pbar:
            # Step by integer index so the run logs exactly total_steps rows; deriving t
            # from t += dt drifts with floating point and can drop (or add) the final step.
            for step in range(total_steps):
                t = step * self.dt
                x_k = self.dynamics.x

                ref_k, ref_log = self.reference.evaluate(t)

                sensor_logs = [sensor.evaluate(t, x_k, u_k) for sensor in self.sensors]
                y_mea_list = [res for res, _ in sensor_logs]
                y_mea = np.concatenate(y_mea_list) if y_mea_list else np.zeros(0)

                x_hat, estim_log = self.estimator.evaluate(t, y_mea, u_k)

                u_k, ctrl_log = self.controller.evaluate(t, ref_k, x_hat)

                # Advance the plant; ``self.dynamics.x`` becomes the next step's state.
                _x_next, dynamics_log = self.dynamics.evaluate(t, u_k)

                y_mea_val = sensor_logs[0][0] if len(self.sensors) == 1 else y_mea

                core_log = CoreLog(
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
                self.logger.log(core_log, comp_logs)

                pbar.update(min(self.dt / divisor, max(0.0, self.t_end / divisor - pbar.n)))

    def export_results(self, directory: str | Path, prefix: str = "log", *, compress: bool = False) -> None:
        """Pack the logged signals into a single {prefix}.npz archive."""
        if self.logger is None:
            msg = "run() must be called before export_results()."
            raise RuntimeError(msg)
        self.logger.finalize(directory, prefix, compress=compress)
