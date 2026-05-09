import multiprocessing
from pathlib import Path
from typing import Any

from simulate.simulation import Simulation


def _run_worker(task: tuple[dict[str, Any], Path, str]) -> bool:
    """
    Worker function to run a single simulation.

    Args:
        task: A tuple containing (config_dict, output_dir, prefix)

    Returns
    -------
        True if successful, False otherwise.
    """
    config, output_dir, prefix = task
    try:
        sim = Simulation.from_config(config)
        sim.run()
        sim.export_results(output_dir, prefix)
    except Exception as e:  # noqa: BLE001
        print(f"Error running simulation: {e}")  # noqa: T201
        return False
    else:
        return True


class ExperimentManager:
    """Manager for batch simulation execution using multiprocessing."""

    def __init__(self, output_dir: str | Path = "results") -> None:
        """Initialize the experiment manager."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_batch(
        self, configs: list[dict[str, Any]], prefixes: list[str] | None = None, max_num_processes: int = 1
    ) -> list[bool]:
        """
        Execute a batch of simulations in parallel.

        Args:
            configs: A list of simulation configuration dictionaries.
            prefixes: Optional list of prefixes for result filenames.

        Returns
        -------
            A list of boolean success statuses.
        """
        if prefixes is None:
            prefixes = [f"sim_{i:03d}" for i in range(len(configs))]

        if len(configs) != len(prefixes):
            msg = "Number of configs and prefixes must match."
            raise ValueError(msg)

        tasks = [(config, self.output_dir, prefix) for config, prefix in zip(configs, prefixes, strict=True)]

        num_processes = min(multiprocessing.cpu_count(), len(configs), max_num_processes)

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(_run_worker, tasks)

        sum(results)

        return results
