import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from rigid_body.environment import moon_position, sun_position


@patch("rigid_body.environment.coord.get_sun")
def test_sun_position_caching(mock_get_sun: MagicMock) -> None:
    """Verify that sun_position caches results within the same minute and clears for a new minute."""
    # Setup mock return value
    mock_sun = MagicMock()
    mock_sun.cartesian.xyz.to.return_value.value = np.array([1.0, 2.0, 3.0])
    mock_get_sun.return_value = mock_sun

    # Clear cache to ensure test independence
    if hasattr(sun_position, "cache"):
        sun_position.cache.clear()  # type: ignore  # noqa: PGH003

    t1 = datetime.datetime(2030, 6, 10, 12, 0, 5, tzinfo=datetime.UTC)
    t2 = datetime.datetime(2030, 6, 10, 12, 0, 45, tzinfo=datetime.UTC)
    t3 = datetime.datetime(2030, 6, 10, 12, 1, 15, tzinfo=datetime.UTC)

    # First call (cache miss)
    r1 = sun_position(t1)
    assert mock_get_sun.call_count == 1

    # Second call within the same minute (cache hit)
    r2 = sun_position(t2)
    assert mock_get_sun.call_count == 1
    assert np.allclose(r1, r2)

    # Third call in the next minute (cache miss)
    r3 = sun_position(t3)
    assert mock_get_sun.call_count == 2
    assert np.allclose(r1, r3)


@patch("rigid_body.environment.coord.get_body")
def test_moon_position_caching(mock_get_body: MagicMock) -> None:
    """Verify that moon_position caches results within the same minute and clears for a new minute."""
    # Setup mock return value
    mock_moon = MagicMock()
    mock_moon.cartesian.xyz.to.return_value.value = np.array([4.0, 5.0, 6.0])
    mock_get_body.return_value = mock_moon

    # Clear cache to ensure test independence
    if hasattr(moon_position, "cache"):
        moon_position.cache.clear()  # type: ignore  # noqa: PGH003

    t1 = datetime.datetime(2030, 6, 10, 12, 0, 5, tzinfo=datetime.UTC)
    t2 = datetime.datetime(2030, 6, 10, 12, 0, 45, tzinfo=datetime.UTC)
    t3 = datetime.datetime(2030, 6, 10, 12, 1, 15, tzinfo=datetime.UTC)

    # First call (cache miss)
    r1 = moon_position(t1)
    assert mock_get_body.call_count == 1

    # Second call within the same minute (cache hit)
    r2 = moon_position(t2)
    assert mock_get_body.call_count == 1
    assert np.allclose(r1, r2)

    # Third call in the next minute (cache miss)
    r3 = moon_position(t3)
    assert mock_get_body.call_count == 2
    assert np.allclose(r1, r3)


def test_cached_arrays_are_copied() -> None:
    """Verify that returned arrays are copies, protecting the cache from mutation."""
    t = datetime.datetime(2030, 6, 10, 12, 0, 5, tzinfo=datetime.UTC)

    r1 = sun_position(t)
    r2 = sun_position(t)

    # Check they have the same value but are different objects
    assert np.allclose(r1, r2)
    assert r1 is not r2

    # Mutating one should not affect the other or future cache returns
    original_val = r1[0]
    r1[0] += 100.0
    assert r2[0] == original_val

    r3 = sun_position(t)
    assert r3[0] == original_val
