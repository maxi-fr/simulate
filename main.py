import argparse
import sys
from pathlib import Path

from simulate.config import load_config
from simulate.experiment import ExperimentManager
from simulate.simulation import Simulation


def main() -> None:
    """Execute the main entry point for the simulation CLI."""
    parser = argparse.ArgumentParser(description="Modular Python Framework for Control System Simulation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--export",
        type=str,
        choices=["csv", "npz", "both"],
        default="npz",
        help="Export format for simulation results (default: npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save simulation results (default: results).",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(1)

    config = load_config(config_path)

    # Check if it's a batch experiment or a single simulation
    if "experiments" in config:
        # Batch experiment
        manager = ExperimentManager(output_dir=args.output_dir)

        # In a real scenario, we might want to generate configs from a sweep definition.
        # For now, we assume 'experiments' is a list of full simulation configs.
        configs = config["experiments"]
        manager.run_batch(configs)
    else:
        # Single simulation
        sim = Simulation.from_config(config)
        sim.run()

        output_dir = Path(args.output_dir)
        if args.export in ["csv", "both"]:
            sim.logger.export_csv(output_dir)
        if args.export in ["npz", "both"]:
            sim.logger.export_npz(output_dir)


if __name__ == "__main__":
    main()
