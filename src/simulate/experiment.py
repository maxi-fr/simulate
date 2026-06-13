import multiprocessing
from pathlib import Path
from typing import Any

from simulate.simulation import Simulation


def _run_worker(task: tuple[dict[str, Any], Path, str, int | None, bool]) -> bool:
    """
    Worker function to run a single simulation.

    Parameters
    ----------
    task : tuple
        A tuple containing (config_dict, output_dir, prefix, chunk_size, compress).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    config, output_dir, prefix, chunk_size, compress = task
    try:
        sim = Simulation.from_config(config)
        sim.run(output_dir=output_dir, prefix=prefix, chunk_size=chunk_size, compress=compress)
        sim.export_results(output_dir, prefix, compress=compress)
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
        self,
        configs: list[dict[str, Any]],
        prefixes: list[str] | None = None,
        max_num_processes: int = 1,
        chunk_size: int | None = 10_000,
        *,
        compress: bool = False,
    ) -> list[bool]:
        """
        Execute a batch of simulations in parallel.

        Parameters
        ----------
        configs : list of dict
            A list of simulation configuration dictionaries.
        prefixes : list of str, optional
            Optional list of prefixes for result filenames.
        chunk_size : int or None, optional
            Steps per chunk file. None disables mid-run flushing.
        compress : bool, optional
            Enable compression for simulation logs.

        Returns
        -------
        list of bool
            A list of boolean success statuses.
        """
        if prefixes is None:
            prefixes = [f"sim_{i:03d}" for i in range(len(configs))]

        if len(configs) != len(prefixes):
            msg = "Number of configs and prefixes must match."
            raise ValueError(msg)

        tasks = [
            (config, self.output_dir, prefix, chunk_size, compress)
            for config, prefix in zip(configs, prefixes, strict=True)
        ]

        num_processes = min(multiprocessing.cpu_count(), len(configs), max_num_processes)

        with multiprocessing.Pool(processes=num_processes) as pool:
            return pool.map(_run_worker, tasks)
