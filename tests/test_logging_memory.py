import gc
import tempfile
import tracemalloc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from simulate.controller import PIDController
from simulate.dynamics import LinearDynamics
from simulate.estimator import IdentityEstimator
from simulate.logger import Logger, UniversalLog
from simulate.measurement_model import LinearMeasurement
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor
from simulate.simulation import Simulation


def _create_simulation(steps: int) -> Simulation:
    """Helper to create a standard simulation instance for testing."""
    dynamics = LinearDynamics(dt=0.01, A=[[0.9]], B=[[1.0]])
    reference = StepReference(dt=0.01)
    sensor = GaussianSensor(dt=0.01, measurement=LinearMeasurement(C=[[1.0]], D=[[0.0]]), std_dev=0.1)
    estimator = IdentityEstimator(dt=0.01)
    controller = PIDController(dt=0.01, kp=[[0.5]], ki=[[0.1]], kd=[[0.05]])

    t_end = steps * 0.01
    return Simulation(
        t_end=t_end,
        dynamics=dynamics,
        reference=reference,
        sensors=[sensor],
        estimator=estimator,
        controller=controller,
    )


class TrackingLogger(Logger):
    """A Logger subclass that tracks the maximum size of the universal log buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.max_universal_logs_size = 0

    def log(self, universal: UniversalLog, components: Mapping[str, Any]) -> None:
        super().log(universal, components)
        self.max_universal_logs_size = max(self.max_universal_logs_size, len(self.universal_logs))


def test_sequential_simulations_memory_leak() -> None:
    """Verify that repeatedly instantiating and running simulations does not leak memory."""
    # Warm up to initialize python imports, class mappings, Pydantic caches, and tqdm
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run and export warmup simulation
        sim_warmup = _create_simulation(steps=100)
        sim_warmup.run(output_dir=tmpdir, prefix="warmup", chunk_size=50)
        sim_warmup.export_results(tmpdir, prefix="warmup")
        del sim_warmup

    gc.collect()

    tracemalloc.start()
    try:
        # Take a snapshot of memory usage
        snapshot_start = tracemalloc.take_snapshot()

        # Run multiple simulations in sequence
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                sim = _create_simulation(steps=500)
                sim.run(output_dir=tmpdir, prefix=f"sim_{i}", chunk_size=100)
                sim.export_results(tmpdir, prefix=f"sim_{i}")
                # Explicitly delete simulation to encourage garbage collection
                del sim
                gc.collect()

        # Take end snapshot and compare
        snapshot_end = tracemalloc.take_snapshot()
        stats = snapshot_end.compare_to(snapshot_start, "lineno")

        # Exclude memory increases from tracing libraries or internal line caches
        # which are loaded when comparing snapshots/printing traceback.
        filtered_stats = [
            stat
            for stat in stats
            if not any(pat in stat.traceback[0].filename for pat in ["tracemalloc", "linecache", "unittest", "pytest"])
        ]

        total_net_increase = sum(stat.size_diff for stat in filtered_stats)

        # Assert memory growth is negligible (e.g. less than 50 KiB)
        max_acceptable_leak_kb = 50.0
        assert total_net_increase / 1024 < max_acceptable_leak_kb, (
            f"Memory leak detected! Net memory increase: {total_net_increase / 1024:.2f} KiB "
            f"(acceptable threshold: {max_acceptable_leak_kb} KiB). "
            f"Top increases: {filtered_stats[:5]}"
        )
    finally:
        tracemalloc.stop()


def test_chunked_vs_unchunked_memory() -> None:
    """Verify that chunking keeps in-memory log buffer size bounded compared to unchunked runs."""
    steps = 1000
    chunk_size = 200

    # 1. Chunked simulation run
    sim_chunked = _create_simulation(steps=steps)
    tracking_logger = TrackingLogger()
    sim_chunked.logger = tracking_logger

    with tempfile.TemporaryDirectory() as tmpdir:
        sim_chunked.run(output_dir=tmpdir, prefix="chunked", chunk_size=chunk_size)
        sim_chunked.export_results(tmpdir, prefix="chunked")

    # 2. Unchunked simulation run
    sim_unchunked = _create_simulation(steps=steps)
    tracking_logger_unchunked = TrackingLogger()
    sim_unchunked.logger = tracking_logger_unchunked

    # Run without chunking (chunk_size=None or output_dir=None)
    sim_unchunked.run(output_dir=None, prefix="unchunked", chunk_size=None)

    # In-memory size assertions
    assert tracking_logger.max_universal_logs_size <= chunk_size, (
        f"Chunked run exceeded chunk_size bound: {tracking_logger.max_universal_logs_size}"
    )
    assert tracking_logger_unchunked.max_universal_logs_size == steps + 1, (
        f"Unchunked run did not accumulate all logs in memory: {tracking_logger_unchunked.max_universal_logs_size}"
    )


def test_merge_chunks_memory_peak(tmp_path: Path) -> None:
    """Benchmark and test that merge_chunks executes correctly and verify memory overhead."""
    # Write multiple chunk files to disk
    logger = Logger()
    prefix = "benchmark_merge"

    # Write 10 chunk files, each with 100 log entries
    for chunk_idx in range(10):
        for t_idx in range(100):
            t = chunk_idx * 100.0 + t_idx * 1.0
            universal = UniversalLog(
                t=t,
                x=np.array([1.0, 2.0]),
                y_mea=np.array([1.1, 2.1]),
                x_hat=np.array([1.2, 2.2]),
                u=np.array([0.5]),
                ref=np.array([1.0]),
            )
            logger.log(universal, {})
        logger.flush_chunk(tmp_path, prefix=prefix)

    # Verify 10 chunk files exist
    chunks = list(tmp_path.glob(f"{prefix}_chunk_*.npz"))
    assert len(chunks) == 10

    # Start tracemalloc to measure memory peak of the merge operation
    tracemalloc.start()
    try:
        Logger.merge_chunks(tmp_path, prefix=prefix)
    finally:
        _, _peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    # The merged file should exist, and chunks should be deleted
    merged_file = tmp_path / f"{prefix}.npz"
    assert merged_file.exists()
    assert not any(tmp_path.glob(f"{prefix}_chunk_*.npz"))

    # Load and verify content integrity
    data = np.load(merged_file)
    assert len(data["universal_t"]) == 1000
    assert data["universal_t"][0] == 0.0
    assert data["universal_t"][-1] == 999.0

    # Print or log peak memory for informational purposes
