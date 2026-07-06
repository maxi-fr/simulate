import gc
import tempfile
import tracemalloc
from pathlib import Path

import numpy as np

from simulate.controller import PIController
from simulate.dynamics import LinearDynamics
from simulate.estimator import IdentityEstimator
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor, LinearMeasurement
from simulate.simulation import Simulation


def _create_simulation(steps: int) -> Simulation:
    """Helper to create a standard simulation instance for testing."""
    dynamics = LinearDynamics(dt=0.01, A=[[0.9]], B=[[1.0]])
    reference = StepReference(dt=0.01)
    sensor = GaussianSensor(dt=0.01, measurement=LinearMeasurement(C=[[1.0]], D=[[0.0]]), std_dev=0.1)
    estimator = IdentityEstimator(dt=0.01)
    controller = PIController(dt=0.01, kp=[[0.5]], ki=[[0.1]])

    t_end = steps * 0.01
    return Simulation(
        t_end=t_end,
        dynamics=dynamics,
        reference=reference,
        sensors=[sensor],
        estimator=estimator,
        controller=controller,
    )


def test_sequential_simulations_memory_leak() -> None:
    """Verify that repeatedly instantiating and running simulations does not leak memory."""
    # Warm up to initialize python imports, class mappings, Pydantic caches, and tqdm
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run and export warmup simulation
        sim_warmup = _create_simulation(steps=100)
        sim_warmup.run(output_dir=f"{tmpdir}/warmup")
        sim_warmup.export_results(f"{tmpdir}/warmup")
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
                sim.run(output_dir=f"{tmpdir}/sim_{i}")
                sim.export_results(f"{tmpdir}/sim_{i}")
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


def test_memmap_run_uses_less_memory_than_ram(tmp_path: Path) -> None:
    """Verify a disk-backed (memmap) run keeps resident memory well below an in-RAM run.

    The memory-mapped buffers live in the OS page cache rather than the Python heap,
    so ``tracemalloc`` (which only sees heap allocations) reports a far lower peak than
    the equivalent ``output_dir=None`` run that accumulates every step in RAM.
    """
    steps = 20_000

    sim_ram = _create_simulation(steps=steps)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        sim_ram.run(output_dir=None)
        _, peak_ram = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    sim_mmap = _create_simulation(steps=steps)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        sim_mmap.run(output_dir=tmp_path, prefix="mem")
        _, peak_mmap = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    sim_mmap.export_results(tmp_path, prefix="mem")

    assert peak_mmap < peak_ram, f"Memmap run peak ({peak_mmap} B) was not below the in-RAM run peak ({peak_ram} B)."


def test_export_produces_single_npz(tmp_path: Path) -> None:
    """Verify export_results yields exactly one {prefix}.npz with all logged rows and no leftovers."""
    steps = 1000
    prefix = "run"

    sim = _create_simulation(steps=steps)
    sim.run(output_dir=tmp_path, prefix=prefix)
    sim.export_results(tmp_path, prefix=prefix)

    merged_file = tmp_path / f"{prefix}.npz"
    assert merged_file.exists()

    # No temporary memmap directory or .npy files remain.
    assert not (tmp_path / f".{prefix}_arrays").exists()
    assert not list(tmp_path.glob("**/*.npy"))

    data = np.load(merged_file)
    assert len(data["t"]) == steps + 1
    assert data["t"][0] == 0.0
