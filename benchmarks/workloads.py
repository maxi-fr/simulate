"""Registry of simulation workloads exercised by the benchmark harness.

Each workload points at an existing example configuration and is run end-to-end
*with logging active* (memory-mapped output written to a temporary directory,
then packed into a single ``.npz``), so that the logger / export path is part of
every measurement.

This module holds data only -- it must not import :mod:`simulate`, so that
:mod:`benchmark` can import it before selecting which engine source to load.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Workload:
    """A single benchmark case.

    Attributes
    ----------
    name : str
        Identifier used on the command line and in the result JSON.
    config : str
        Path to the YAML config, relative to the repository root.
    repeats : int
        Number of timed repetitions; the reported time is the median.
    """

    name: str
    config: str
    repeats: int


WORKLOADS: dict[str, Workload] = {
    # Light, core-deps-only loop (~2000 steps): RK4 + Luenberger observer + PI.
    "dc_motor": Workload(
        name="dc_motor",
        config="examples/01_dc_motor/config.yaml",
        repeats=5,
    ),
    # Heavy, memory-relevant case (~500 steps): 6-DOF rigid body with many
    # effectors/sensors and large per-step logs. Needs the ``spacecraft`` group.
    "satellite": Workload(
        name="satellite",
        config="examples/03_satellite/quat_feedback.yaml",
        repeats=3,
    ),
}
