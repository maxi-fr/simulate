"""Run the simulation benchmarks against a single engine source tree.

This script measures, per workload, the wall-clock time (median of N repeats)
and the peak memory (via :mod:`tracemalloc`) of running a simulation including
its logging/merge phase, and writes the results as JSON.

It is meant to be invoked twice by :mod:`compare` -- once with ``--src``
pointing at the working-tree ``src`` and once at a ``src`` checked out at an
earlier commit -- so the two JSON files can be diffed. ``--src`` is inserted at
the front of ``sys.path`` *before* :mod:`simulate` is imported, which selects
the engine source under test (the project is installed via a plain path
``.pth``, so the inserted entry shadows it).

Example
-------
``uv run python benchmarks/benchmark.py --src src --out new.json``
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING

from workloads import WORKLOADS, Workload

if TYPE_CHECKING:
    from simulate.simulation import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFIX = "bench"


def _ensure_importable(path: Path) -> None:
    """Insert ``path`` at the front of ``sys.path`` if not already present."""
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)

    repo_entry = str(REPO_ROOT)
    if repo_entry not in sys.path:
        sys.path.insert(0, repo_entry)


def _load_simulation(config: Path) -> Simulation:
    """Build a fresh simulation from a YAML config.

    The config's directory is added to ``sys.path`` first, so configs that
    reference a local module (e.g. the DC motor's ``dc_motor.py``) resolve.

    Returns
    -------
    Simulation
        A freshly instantiated simulation ready to run.
    """
    from simulate.simulation import Simulation

    _ensure_importable(config.parent)
    return Simulation.from_yaml(config)


def _time_workload(config: Path, chunk_size: int, repeats: int) -> dict[str, float]:
    """Time the run and export phases across ``repeats`` fresh simulations.

    A new :class:`~simulate.simulation.Simulation` is built for every repeat
    because :meth:`~simulate.simulation.Simulation.run` mutates component state.

    Returns
    -------
    dict
        Median ``time_run``, ``time_export`` and ``time_total`` in seconds.
    """
    run_times: list[float] = []
    export_times: list[float] = []
    for i in range(repeats + 1):
        sim = _load_simulation(config)
        with tempfile.TemporaryDirectory() as tmp:
            start = time.perf_counter()
            sim.run(output_dir=tmp, prefix=PREFIX, chunk_size=chunk_size)
            after_run = time.perf_counter()
            sim.export_results(tmp, prefix=PREFIX)
            after_export = time.perf_counter()

        if i == 0:
            continue

        run_times.append(after_run - start)
        export_times.append(after_export - after_run)

    time_run = statistics.median(run_times)
    time_export = statistics.median(export_times)
    return {
        "time_run": time_run,
        "time_export": time_export,
        "time_total": time_run + time_export,
    }


def _measure_memory(config: Path, chunk_size: int) -> dict[str, int]:
    """Measure peak Python+numpy memory of the run and export phases.

    Memory is deterministic (sensors are seeded), so a single traced run
    suffices. ``tracemalloc`` totals all allocation domains, including numpy's,
    so the logger's pre-allocated buffers and the chunk merge are captured.

    Returns
    -------
    dict
        ``peak_run_bytes``, ``peak_export_bytes`` and their max ``peak_total_bytes``.
    """
    sim = _load_simulation(config)
    with tempfile.TemporaryDirectory() as tmp:
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            sim.run(output_dir=tmp, prefix=PREFIX, chunk_size=chunk_size)
            _, peak_run = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            sim.export_results(tmp, prefix=PREFIX)
            _, peak_export = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    return {
        "peak_run_bytes": peak_run,
        "peak_export_bytes": peak_export,
        "peak_total_bytes": max(peak_run, peak_export),
    }


def _measure_workload(workload: Workload, repeats: int) -> dict[str, float]:
    """Time and profile a single workload.

    Returns
    -------
    dict
        Combined timing (seconds) and peak-memory (bytes) metrics.
    """
    config = (REPO_ROOT / workload.config).resolve()
    if not config.exists():
        msg = f"Config not found for workload {workload.name!r}: {config}"
        raise FileNotFoundError(msg)

    # Memory is measured first, before the timing repeats warm numpy's
    # allocation pools, so the peak reflects a representative cold run.
    metrics: dict[str, float] = {}
    metrics.update(_measure_memory(config, workload.chunk_size))
    metrics.update(_time_workload(config, workload.chunk_size, repeats))
    return metrics


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed ``src``, ``out``, ``workloads`` and ``repeats``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Engine source dir to put first on sys.path (e.g. 'src').")
    parser.add_argument("--out", required=True, help="Path to write the result JSON.")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=sorted(WORKLOADS),
        default=sorted(WORKLOADS),
        help="Subset of workloads to run (default: all).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override the per-workload timed repetitions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the selected workloads against ``--src`` and write JSON to ``--out``.

    Returns
    -------
    int
        Process exit code (always ``0``; comparison/flagging lives in compare.py).
    """
    args = _parse_args(argv)
    _ensure_importable(Path(args.src).resolve())

    results: dict[str, dict[str, float]] = {}
    for name in args.workloads:
        workload = WORKLOADS[name]
        repeats = args.repeats if args.repeats is not None else workload.repeats
        print(f"benchmarking {name} (repeats={repeats}) against src={args.src} ...", file=sys.stderr)
        results[name] = _measure_workload(workload, repeats)

    payload = {
        "metadata": {
            "src": str(Path(args.src).resolve()),
            "python": platform.python_version(),
            "platform": sys.platform,
        },
        "workloads": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
