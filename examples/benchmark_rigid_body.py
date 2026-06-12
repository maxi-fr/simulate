# ruff: noqa: T201, PTH123
"""Benchmark script for the rigid body system dynamics.

This script benchmarks the performance of the rigid body simulation under
three different scenarios:
1. Raw Physics: Direct integration of RigidBodyDynamics with a Wrench and ReactionWheel.
2. Raw Physics with Gravity Gradient: Direct integration including EarthGravity (gravity gradient torque).
3. Orchestrated Simulation: The full OOP simulation loop including reference tracking,
   sensors, estimators, PID controllers, outputs, and logging.
"""

import argparse
import contextlib
import os
import sys
import time
from collections.abc import Generator
from typing import Any, Self

import numpy as np

from rigid_body.effector import EarthGravity, ReactionWheelArray, Wrench
from rigid_body.rigid_body import RigidBodyDynamics
from simulate.controller import PIDController
from simulate.dynamics import NoLog
from simulate.estimator import IdentityEstimator
from simulate.output import Output
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor, Sensor
from simulate.simulation import Simulation


class StateElementOutput(Output[NoLog]):
    """Output component that extracts a single index from the state vector."""

    def __init__(self, dt: float, index: int) -> None:
        """Initialize StateElementOutput with sample rate and state index."""
        super().__init__(dt)
        self.index = index

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate StateElementOutput from a configuration dictionary."""
        return cls(dt=float(config["dt"]), index=int(config["index"]))

    def update(
        self,
        t: float,  # noqa: ARG002
        x: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, NoLog]:
        """Extract the state element at the configured index."""
        x_arr = np.atleast_1d(x)
        return x_arr[self.index], NoLog()


@contextlib.contextmanager
def silence_outputs() -> Generator[None]:
    """Context manager to silence stdout and stderr during benchmark runs."""
    with open(os.devnull, "w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = f
            sys.stderr = f
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def run_raw_physics(dt: float, steps: int) -> float:
    """Benchmark raw physics integration without orchestrator or logging."""
    rw_array = ReactionWheelArray(
        axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        inertia=0.05,
        torque_constant=0.08,
        time_constant=0.04,
        max_current=2.5,
    )
    dynamics = RigidBodyDynamics(
        dt=dt,
        mass=5.0,
        inertia=[1.0, 2.0, 3.0],
        effectors=[Wrench(), rw_array],
    )
    cmd = np.array([2.0, 0.1, -0.1, 0.05, -0.05, 0.1, 0.0, 0.0, 0.625])  # 6 wrench + 3 wheel currents

    start = time.perf_counter()
    for k in range(steps):
        dynamics.evaluate(k * dt, cmd)
    end = time.perf_counter()
    return end - start


def run_raw_physics_gg(dt: float, steps: int) -> float:
    """Benchmark raw physics integration with gravity-gradient torque."""
    dynamics = RigidBodyDynamics(
        dt=dt,
        mass=500.0,
        inertia=[100.0, 200.0, 300.0],
        effectors=[EarthGravity(mu=3.986e14)],
    )
    # Set orbit radius and initial attitude tilt to make gravity gradient active
    dynamics.x[0:3] = np.array([7.0e6, 0.0, 0.0])
    half = np.pi / 8
    dynamics.x[6:10] = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    cmd = np.zeros(0)

    start = time.perf_counter()
    for k in range(steps):
        dynamics.evaluate(k * dt, cmd)
    end = time.perf_counter()
    return end - start


def run_orchestrated_simulation(dt: float, steps: int) -> float:
    """Benchmark the full OOP simulation loop using the orchestrator and logger."""
    t_end = steps * dt
    rw_array = ReactionWheelArray(
        axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        inertia=0.05,
        torque_constant=0.08,
        time_constant=0.04,
        max_current=2.5,
    )
    dynamics = RigidBodyDynamics(
        dt=dt,
        mass=5.0,
        inertia=[1.0, 2.0, 3.0],
        effectors=[Wrench(), rw_array],
    )

    # 9 single-element output channels, one per measured state index.
    outputs: list[Output[Any]] = [StateElementOutput(dt=dt, index=i) for i in range(9)]
    sensors: list[Sensor[Any]] = [GaussianSensor(dt=dt, std_dev=0.01) for _ in range(9)]
    reference = StepReference(dt=dt, step_value=np.ones(9))
    estimator = IdentityEstimator(dt=dt)
    controller = PIDController(
        dt=dt,
        kp=np.eye(9) * 0.1,
        ki=np.eye(9) * 0.01,
        kd=np.eye(9) * 0.01,
    )

    sim = Simulation(
        t_end=t_end,
        dynamics=dynamics,
        outputs=outputs,
        reference=reference,
        sensors=sensors,
        estimator=estimator,
        controller=controller,
    )

    # We silence stdout/stderr to suppress tqdm progress bar printing during benchmark
    with silence_outputs():
        start = time.perf_counter()
        sim.run()
        end = time.perf_counter()

    return end - start


def main() -> None:
    """Run the benchmarking suite."""
    parser = argparse.ArgumentParser(description="Benchmark suite for the rigid body system.")
    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="Number of simulation steps per scenario (default: 20000).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Integration time step in seconds (default: 0.01).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs to average over (default: 3).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("        RIGID BODY SIMULATION BENCHMARK SUITE")
    print("=" * 60)
    print("Configuration:")
    print(f"  Steps per run:  {args.steps:,}")
    print(f"  Time step (dt): {args.dt} s")
    print(f"  Simulated time: {args.steps * args.dt:.1f} s")
    print(f"  Runs to average: {args.runs}")
    print("-" * 60)

    # Run benchmarks
    results: dict[str, list[float]] = {
        "Raw Physics (No GG)": [],
        "Raw Physics (With GG)": [],
        "Orchestrated Loop": [],
    }

    for run in range(1, args.runs + 1):
        print(f"Executing Run {run}/{args.runs}...")

        t_raw = run_raw_physics(args.dt, args.steps)
        results["Raw Physics (No GG)"].append(t_raw)

        t_gg = run_raw_physics_gg(args.dt, args.steps)
        results["Raw Physics (With GG)"].append(t_gg)

        t_orch = run_orchestrated_simulation(args.dt, args.steps)
        results["Orchestrated Loop"].append(t_orch)

    print("-" * 60)
    print("BENCHMARK RESULTS (Average over runs):")
    print("-" * 60)
    print(f"{'Scenario':<25} | {'Avg Time (s)':<12} | {'Steps/sec':<12} | {'Real-time Ratio':<15}")
    print("-" * 70)

    avg_times = {}
    for scenario, times in results.items():
        avg_time = float(np.mean(times))
        avg_times[scenario] = avg_time
        steps_per_sec = args.steps / avg_time
        rt_ratio = (args.steps * args.dt) / avg_time
        print(f"{scenario:<25} | {avg_time:<12.4f} | {steps_per_sec:<12.1f} | {rt_ratio:<15.1f}x")

    print("=" * 60)
    # Calculate and report orchestration overhead
    overhead_ratio = avg_times["Orchestrated Loop"] / avg_times["Raw Physics (No GG)"]
    print(f"Orchestration & Logging Overhead Factor: {overhead_ratio:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
