"""Differential tests for the space-environment models (same underlying astropy/IGRF/MSIS libraries).

These are near-verbatim ports; the tests confirm the ported wrappers return the same values at a
fixed epoch and geodetic point.
"""

import datetime

import numpy as np
import pytest
from diffhelpers import rand_unit_vec

from rigid_body.environment import (
    is_in_shadow,
    magnetic_field_vector,
    moon_position,
    solar_radiation_pressure_constant,
    sun_position,
)

_EPOCH = datetime.datetime(2024, 3, 21, 6, 30, 0)  # noqa: DTZ001 -- naive UTC, matches legacy usage
_LAT, _LON, _ALT = 12.5, -47.0, 5.2e5


def test_sun_position_matches() -> None:
    """Sun ECI position agrees at a fixed epoch (both wrap the same astropy ephemeris)."""
    old = pytest.importorskip("simulation.environment")
    np.testing.assert_allclose(sun_position(_EPOCH), old.sun_position(_EPOCH), rtol=1e-9, atol=1e-3)


def test_moon_position_matches() -> None:
    """Moon ECI position agrees at a fixed epoch (both wrap the same astropy ephemeris)."""
    old = pytest.importorskip("simulation.environment")
    np.testing.assert_allclose(moon_position(_EPOCH), old.moon_position(_EPOCH), rtol=1e-9, atol=1e-3)


def test_magnetic_field_library_swap_changes_value() -> None:
    """DOCUMENTED DIFFERENCE: the port swapped the IGRF library (old ``pyIGRF`` -> new ``ppigrf``).

    The field direction is preserved (nearly parallel), but the magnitude differs by a few percent
    because the two libraries use different IGRF coefficient sets. This is an intended dependency
    change, not a port bug, so the magnetometer truth model is not bit-for-bit identical.
    """
    old = pytest.importorskip("simulation.environment")
    new_b = magnetic_field_vector(_EPOCH, _LAT, _LON, _ALT)
    old_b = old.magnetic_field_vector(_EPOCH, _LAT, _LON, _ALT)

    assert not np.allclose(new_b, old_b, rtol=1e-3, atol=0.0), "expected the IGRF library swap to change the value"
    cos = float(np.dot(new_b, old_b) / (np.linalg.norm(new_b) * np.linalg.norm(old_b)))
    assert cos > 0.999, "field directions should stay nearly parallel across the library swap"
    np.testing.assert_allclose(np.linalg.norm(new_b), np.linalg.norm(old_b), rtol=0.1)


def test_solar_radiation_pressure_constant_matches(rng: np.random.Generator) -> None:
    """The solar-radiation-pressure magnitude (a ``1/r^2`` scaling from 1 AU) is identical."""
    old = pytest.importorskip("simulation.environment")
    dist = rng.uniform(-1.5e11, 1.5e11, size=3)
    np.testing.assert_allclose(
        solar_radiation_pressure_constant(dist),
        old.solar_radiation_pressure_constant(dist),
        rtol=1e-12,
    )


def test_is_in_shadow_matches(rng: np.random.Generator) -> None:
    """The cylindrical eclipse test returns the same boolean for random LEO positions."""
    old = pytest.importorskip("simulation.environment")
    sun_eci = sun_position(_EPOCH)
    for _ in range(20):
        r_eci = rand_unit_vec(rng) * 7.0e6  # LEO radius (above the Earth's surface)
        assert is_in_shadow(r_eci, sun_eci) == old.is_in_shadow(r_eci, sun_eci)
