import datetime

import numpy as np
from astropy import units as u
from astropy.coordinates import TEME, CartesianRepresentation, SkyCoord
from astropy.time import Time
from numpy.typing import NDArray
from sgp4.api import WGS84, Satrec
from sgp4.conveniences import jday_datetime

G = 6.67430e-11  # universal gravitational constant
M = 5.972e24  # mass of earth
MU = G * M  # gravitational parameter


def orbit_dynamics(m: float, r: np.ndarray, ext_force: np.ndarray) -> NDArray[np.float64]:
    """
    Compute orbital acceleration of the center of mass of the satellite according to Newtons laws of motion.

    Parameters
    ----------
    m : float
        Mass of the satellite [kg].
    r : np.ndarray, shape (3,)
        Position vector in the ECI frame [m].
    ext_force : np.ndarray, shape (3,)
        External force vector in the ECI frame [N].

    Returns
    -------
    np.ndarray, shape (3,)
        Acceleration vector (d^2r/dt^2) in the ECI frame [m/s^2].
    """
    r_norm = np.linalg.norm(r)
    d_v: NDArray[np.float64] = -(MU / r_norm**3) * r + ext_force / m
    return d_v


class SGP4:
    """
    A wrapper for the SGP4 propagator that simplifies initialization and propagation.

    This class handles the conversion from orbital elements or TLEs to the `Satrec`
    object and provides a convenient method to propagate the orbit to a specific
    time, returning the position and velocity in the ECI (GCRS) frame.
    """

    def __init__(self, satrec: Satrec) -> None:
        """
        Initialize the SGP4 wrapper with a Satrec object.

        Parameters
        ----------
        satrec : Satrec
            An initialized `Satrec` object from the `sgp4` library.
        """
        self.satrec = satrec

    @classmethod
    def from_elements(  # noqa: PLR0913
        cls,
        e: float,
        i: float,
        raan: float,
        arg_pe: float,
        M0: float,
        MM: float,
        t0: datetime.datetime,
        B_star: float = 0.0,
    ) -> "SGP4":
        """
        Classical Keplerian orbital elements.

        Parameters
        ----------
        e : float
            Eccentricity [-]
        i : float
            Inclination [deg]
        raan : float
            Right ascension of ascending node Ω [deg]
        arg_pe : float
            Argument of perigee ω [deg]
        M0 : float
            Mean anomaly at epoch t0 [deg]
        MM : float
            Mean motion in [rev/day]
        t0 : datetime.datetime
            The epoch of the orbital elements.
        B_star : float, optional
            The B* drag term. Defaults to 0.0.

        Returns
        -------
        SGP4
            An instance of the SGP4 class.
        """
        epoch = t0 - datetime.datetime.fromisoformat("1949-12-31T00:00:00Z")

        epoch = epoch.days + epoch.seconds / (3600.0 * 24.0)  # type: ignore[assignment]

        no_kozai = 2 * np.pi / (24 * 60) * MM

        satrec = Satrec()
        satrec.sgp4init(
            WGS84,
            "i",
            25544,
            epoch,
            B_star,
            0.0,
            0.0,
            e,
            np.deg2rad(arg_pe),
            np.deg2rad(i),
            np.deg2rad(M0),
            no_kozai,
            np.deg2rad(raan),
        )

        return cls(satrec)

    @classmethod
    def from_tle(cls, tle1: str, tle2: str, earth_grav=WGS84) -> "SGP4":  # noqa: ANN001
        """
        Initialize the orbit from Two-Line Element (TLE) set strings.

        Parameters
        ----------
        tle1 : str
            The first line of the TLE.
        tle2 : str
            The second line of the TLE.
        earth_grav : Any, optional
            Gravitational constants. Defaults to WGS84.

        Returns
        -------
        SGP4
            An instance of the SGP4 class.
        """
        satrec = Satrec.twoline2rv(tle1, tle2, earth_grav)
        return cls(satrec)

    def propagate(self, time: list[datetime.datetime] | datetime.datetime) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagates the satellite orbit to a given time.

        Note: make sure to call this function vectorized. It is much faster than inside a for loop

        Parameters
        ----------
        time : list[datetime.datetime] or datetime.datetime
            The time to which to propagate the orbit.

        Returns
        -------
        r : np.ndarray
            ECI position vectors [m].
        v : np.ndarray
            ECI velocity vectors [m/s].
        """
        if isinstance(time, datetime.datetime):
            time = [time]

        jd = np.empty_like(time)
        fr = np.empty_like(time)
        for i, t in enumerate(time):
            jd[i], fr[i] = jday_datetime(t)

        error_code, r_TEME, v_TEME = self.satrec.sgp4_array(jd, fr)
        if not np.all(error_code == 0):
            msg = f"SGP4 propagation failed with error codes: {error_code}"
            raise RuntimeError(msg)

        r_TEME = np.atleast_2d(r_TEME)
        v_TEME = np.atleast_2d(v_TEME)

        teme = TEME(obstime=Time(time, format="datetime", scale="utc"))
        r_ECI = (
            SkyCoord(
                CartesianRepresentation(r_TEME[:, 0], r_TEME[:, 1], r_TEME[:, 2], unit=u.km),
                frame=teme,
                representation_type="cartesian",
            )
            .transform_to("gcrs")
            .cartesian.xyz.to(u.m)
            .value
        )
        v_ECI = (
            SkyCoord(
                CartesianRepresentation(v_TEME[:, 0], v_TEME[:, 1], v_TEME[:, 2], unit=u.km),
                frame=teme,
                representation_type="cartesian",
            )
            .transform_to("gcrs")
            .cartesian.xyz.to(u.m)
            .value
        )

        return r_ECI.T.squeeze(), v_ECI.T.squeeze()
