import datetime
import functools
from collections.abc import Callable
from typing import Any

import astropy.coordinates as coord
import numpy as np
import pyIGRF
import pymap3d
import pymsis
from astropy import units as u
from astropy.time import Time


def cache_per_interval(
    seconds: float,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Cache the output of a function for a given time interval in seconds."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache: dict[datetime.datetime, Any] = {}

        @functools.wraps(func)
        def wrapper(dt_utc: datetime.datetime, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            dt_utc = dt_utc.replace(tzinfo=datetime.UTC) if dt_utc.tzinfo is None else dt_utc.astimezone(datetime.UTC)

            ts = dt_utc.timestamp()
            bucket_ts = int(ts // seconds) * seconds
            dt_bucket = datetime.datetime.fromtimestamp(bucket_ts, tz=datetime.UTC)

            if dt_bucket not in cache:
                cache.clear()
                cache[dt_bucket] = func(dt_utc, *args, **kwargs)

            val = cache[dt_bucket]
            if isinstance(val, np.ndarray):
                return val.copy()
            return val

        wrapper.cache = cache  # type: ignore  # noqa: PGH003
        return wrapper

    return decorator


@cache_per_interval(1.0)
def atmosphere_density_msis(  # noqa: PLR0913
    dt_utc: datetime.datetime,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    f107: float = 150,
    f107a: float = 150,
    ap: int = 4,
) -> float:
    """
    Calculate atmospheric density using the pymsis library.

    This function calls the MSIS model to get atmospheric density for a
    specific time and location.

    Parameters
    ----------
    dt_utc : datetime.datetime
        The UTC datetime for the density calculation.
    lat_deg : float
        Latitude in degrees.
    lon_deg : float
        Longitude in degrees.
    alt_m : float
        Altitude in meters.
    f107 : float, optional
        Daily F10.7 solar flux, by default 150.
    f107a : float, optional
        81-day average of F10.7 solar flux, by default 150.
    ap : int, optional
        The Ap geomagnetic index, by default 4.

    Returns
    -------
    float
        The calculated total mass density in kg/m^3.

    """
    alt_km = alt_m / 1000.0

    result = pymsis.calculate(
        np.array([dt_utc.astimezone(datetime.UTC).replace(tzinfo=None)], dtype="datetime64[s]"),
        lat_deg,
        lon_deg,
        alt_km,
        f107,
        f107a,
        ap,
    )

    rho_kg_m3: float = result[:, 0].item(0)

    return rho_kg_m3


@cache_per_interval(1.0)
def magnetic_field_vector(dt_utc: datetime.datetime, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """
    Calculate the Earth's magnetic field vector in the ECI frame.

    This function uses the IGRF model to get the magnetic field components
    in the North-East-Down (NED) frame, then transforms them to the
    Earth-Centered Inertial (ECI) J2000 frame.

    Parameters
    ----------
    dt_utc : datetime.datetime
        The UTC datetime for the calculation.
    lat_deg : float
        Geodetic latitude in degrees.
    lon_deg : float
        Geodetic longitude in degrees.
    alt_m : float
        Altitude above the WGS84 ellipsoid in meters.

    Returns
    -------
    np.ndarray, shape (3,)
        The magnetic field vector [Bx, By, Bz] in the ECI frame, in Tesla [T].
    """
    _d_dec, _i_inc, _h_hor, b_n, b_e, b_v, b_tot = pyIGRF.igrf_value(lat_deg, lon_deg, alt_m / 1000)

    b_ecef = pymap3d.ned2ecef(b_n, b_e, b_v, lat_deg, lon_deg, alt_m, pymap3d.Ellipsoid.from_name("wgs84"))

    b_eci = np.asarray(pymap3d.ecef2eci(*b_ecef, time=dt_utc))

    return np.asarray((b_eci / np.linalg.norm(b_eci)) * b_tot * 1e-9, dtype=np.float64)  # Convert from nT to T


def cache_per_minute(func: Callable[[datetime.datetime], np.ndarray]) -> Callable[[datetime.datetime], np.ndarray]:
    """Cache the output of a function for the same simulation minute."""
    cache: dict[datetime.datetime, np.ndarray] = {}

    @functools.wraps(func)
    def wrapper(dt_utc: datetime.datetime, *args: Any, **kwargs: Any) -> np.ndarray:  # noqa: ANN401
        if dt_utc.tzinfo is not None:
            dt_utc = dt_utc.astimezone(datetime.UTC)
        dt_minute = dt_utc.replace(second=0, microsecond=0)

        if dt_minute not in cache:
            cache.clear()
            cache[dt_minute] = func(dt_utc, *args, **kwargs)
        return cache[dt_minute].copy()

    wrapper.cache = cache  # type: ignore  # noqa: PGH003
    return wrapper


@cache_per_minute
def sun_position(dt_utc: datetime.datetime) -> np.ndarray:
    """
    Calculate the Sun's position vector in the GCRS (ECI) frame.

    Parameters
    ----------
    dt_utc : datetime.datetime
        The UTC datetime for the calculation.

    Returns
    -------
    np.ndarray, shape (3,)
        The Sun's position vector [x, y, z] in the GCRS frame, in meters [m].
    """
    time = Time(dt_utc.strftime("%Y-%m-%d %H:%M:%S"), scale="utc")
    sun = coord.get_sun(time)

    return sun.cartesian.xyz.to(u.m).value


@cache_per_minute
def moon_position(dt_utc: datetime.datetime) -> np.ndarray:
    """
    Calculate the Moon's position vector in the GCRS (ECI) frame.

    Parameters
    ----------
    dt_utc : datetime.datetime
        The UTC datetime for the calculation.

    Returns
    -------
    np.ndarray, shape (3,)
        The Moon's position vector [x, y, z] in the GCRS frame, in meters [m].
    """
    time = Time(dt_utc.strftime("%Y-%m-%d %H:%M:%S"), scale="utc")
    moon = coord.get_body("moon", time)

    return moon.cartesian.xyz.to(u.m).value


def solar_radiation_pressure_constant(dist: np.ndarray) -> float:
    """
    Calculate the solar radiation pressure at the satellite's location.

    Parameters
    ----------
    dist : np.ndarray, shape (3,)
        Distance vector from the satellite to the Sun [m].

    Returns
    -------
    float
        Solar radiation pressure [N/m^2].
    """
    p_1au = 4.56e-6  # Solar radiation pressure at 1 AU [N/m^2]
    au_m = 149597870700.0  # 1 Astronomical Unit in meters

    return float(p_1au * (au_m / np.linalg.norm(dist)) ** 2)  # FIXME: maybe wrong formula


E_RADIUS_M = 6378137.0  # Earth's equatorial radius in meters


def is_in_shadow(r_eci: np.ndarray, sun_pos_eci: np.ndarray) -> bool:
    """
    Determine if the satellite is in Earth's shadow using a cylindrical model.

    This model assumes the Sun's rays are parallel and casts a cylindrical
    shadow behind the Earth.

    Parameters
    ----------
    r_eci : np.ndarray, shape (3,)
        Satellite position vector in the ECI frame [m].
    sun_pos_eci : np.ndarray, shape (3,)
        Sun position vector in the ECI frame [m].

    Returns
    -------
    bool
        True if the satellite is in shadow, False otherwise.
    """
    sun_position_unit = sun_pos_eci / np.linalg.norm(sun_pos_eci)
    return float(np.dot(r_eci, sun_position_unit)) < float(-np.sqrt(np.linalg.norm(r_eci) ** 2 - E_RADIUS_M**2))
