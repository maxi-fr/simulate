from pathlib import Path

from simulate.config import SimulationConfig
from simulate.controller import Controller
from simulate.logger import Logger, UniversalLog
from simulate.plant import DiscretePlant


class Simulation:
    """Central orchestrator for the simulation loop."""

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize the simulation with the given configuration."""
        self.config = config
        self.plant = DiscretePlant(config.plant)
        self.controller = Controller(config.controller)
        self.logger = Logger()

        # The base tick is dictated by the plant's update period
        self.dt = self.config.plant.dt
        self.t_end = self.config.t_end

    def generate_reference(self, t: float) -> float:
        """
        Generate the reference signal for the current time.

        In a full implementation, this might be a separate component.
        For now, we provide a simple step response.
        """
        # Step response at t=0.5
        return 1.0 if t >= 0.5 else 0.0

    def run(self) -> None:
        """Run the simulation loop until t_end."""
        t = 0.0

        # Initial states
        u_k = 0.0
        y_k = 0.0

        # Use round(t, 9) to prevent floating point accumulation drift in the loop condition
        while round(t, 9) <= self.t_end:
            # 1. Reference Generation
            ref_k = self.generate_reference(t)

            # 2 & 3. Measurement & Estimation skipped for this iteration.
            # y_k represents the true plant output from the previous tick.

            # 4. Control
            # Controller steps, using ref and y_k.
            u_k, ctrl_log = self.controller.step(t, ref_k, y_k)

            # 5. Actuation
            # Plant steps, using u_k.
            y_k, plant_log = self.plant.step(t, u_k)

            # 6. Logging
            uni_log = UniversalLog(t=t, y=y_k, u=u_k)
            comp_logs = {"plant": plant_log, "controller": ctrl_log}
            self.logger.log(uni_log, comp_logs)

            # Advance time
            t += self.dt

    def export_results(self, directory: str | Path, prefix: str = "sim") -> None:
        """Export simulation results via the Logger."""
        self.logger.export_csv(directory, prefix)
        self.logger.export_npz(directory, prefix)
