import multiprocessing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .simulation import Simulation


def _run_worker(task: tuple[dict[str, Any], Path, str, bool, bool]) -> bool:
    """
    Worker function to run a single simulation.

    Parameters
    ----------
    task : tuple
        A tuple containing (config_dict, output_dir, prefix, use_mmap, compress).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    config, output_dir, prefix, use_mmap, compress = task
    try:
        sim = Simulation.from_config(config)
        sim.run(output_dir=output_dir, prefix=prefix, use_mmap=use_mmap, compress=compress)
        sim.export_results(output_dir, prefix, compress=compress)
    except Exception as e:  # noqa: BLE001
        print(f"Error running simulation: {e}")  # noqa: T201
        return False
    else:
        return True


class ExperimentManager:
    """Manager for batch simulation execution using multiprocessing."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        """Initialize the experiment manager."""
        if output_dir is None:
            local_now = datetime.now(UTC).astimezone()
            output_dir = f"results/experiment_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_batch(
        self,
        configs: list[dict[str, Any]],
        prefixes: list[str] | None = None,
        max_num_processes: int = 1,
        *,
        use_mmap: bool = False,
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
        use_mmap : bool, optional
            If True, use memory-mapped files for logging.
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

        tasks = [(config, self.output_dir, prefix, use_mmap, compress) for config, prefix in zip(configs, prefixes, strict=True)]

        num_processes = min(multiprocessing.cpu_count(), len(configs), max_num_processes)

        with multiprocessing.Pool(processes=num_processes) as pool:
            return pool.map(_run_worker, tasks)
