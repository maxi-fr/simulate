"""Shared setup for the differential port tests.

These tests import BOTH the ported package (``spacecraft``) and the original, temporarily-stored
repo (``adaptive-satellite-control``) in the same process and compare corresponding functions on
identical inputs. The old repo is a plain ``sys.path`` source tree (hatchling layout) exposing the
top-level modules ``utils``, ``simulation`` and ``flight_software``.

If the old repo has been removed, every test module skips cleanly: the ``sys.path`` entry is simply
not added and the ``pytest.importorskip(...)`` calls at the top of each test module raise ``Skip``.
All cross-repo and CasADi imports are funnelled through ``importorskip`` (string-based) so the static
type checker never tries to resolve them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# adaptive-satellite-control/src holds the legacy `utils`, `simulation`, `flight_software` modules.
# The old repo lives alongside this conftest, inside tests/differential/ (and is gitignored).
OLD_ROOT = Path(__file__).resolve().parent / "adaptive-satellite-control" / "src"

if OLD_ROOT.is_dir() and str(OLD_ROOT) not in sys.path:
    # Insert at the front so the legacy top-level `utils` module wins over anything else.
    sys.path.insert(0, str(OLD_ROOT))


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG, re-seeded per test for reproducible inputs."""
    return np.random.default_rng(0)
