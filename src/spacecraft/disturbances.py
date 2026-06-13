# ruff: noqa: N806, N803
import numpy as np

from .environment import solar_radiation_pressure_constant
from .quaternion import Quaternion
from .surface import Surface, VectorizedSurfaces

# Earth constants
R_EARTH = 6.378137e6  # Earth's equatorial radius in meters
G = 6.67430e-11  # universal gravitational constant
M = 5.972e24  # mass of earth
MU = G * M  # gravitational parameter

J2 = 1.08262668e-3  # J2 zonal harmonic
J3 = -2.5327e-6  # J3 zonal harmonic
J4 = -1.6196e-6  # J4 zonal harmonic


def non_spherical_gravity_forces(r_eci: np.ndarray, m: float) -> np.ndarray:
    """
    Calculate the disturbance forces due to Earth's non-spherical gravity.

    This function computes the perturbing acceleration from the J2, J3, and J4
    zonal harmonic coefficients and returns the corresponding force.

    Parameters
    ----------
    r_eci : np.ndarray, shape (3,)
        Position vector of the satellite in the ECI frame [m].
    m : float
        Mass of the satellite [kg].

    Returns
    -------
    np.ndarray, shape (3,)
        The total disturbance force vector in the ECI frame [N].
    """
    r = np.linalg.norm(r_eci)
    x, y, z = r_eci
    x_r = x / r
    y_r = y / r

    z_r = z / r
    z_r2 = z_r * z_r
    z_r3 = z_r2 * z_r
    z_r4 = z_r3 * z_r

    MU_r = MU / r**2
    RE_r = R_EARTH / r

    factor_J2 = -1.5 * J2 * MU_r * RE_r**2
    term_J2 = 1 - 5 * z_r2
    a_J2 = factor_J2 * np.array([term_J2 * x_r, term_J2 * y_r, (3 - 5 * z_r2) * z_r])

    factor_J3 = -0.5 * J3 * MU_r * RE_r**3
    term_J3_xy = 5 * (7 * z_r3 - 3 * z_r)
    a_J3 = factor_J3 * np.array([term_J3_xy * x_r, term_J3_xy * y_r, 3 * (10 * z_r2 - (35 / 3) * z_r4 - 1)])

    factor_J4 = -0.625 * J4 * MU_r * RE_r**4
    term_J4_xy = 3 - 42 * z_r2 + 63 * z_r4
    a_J4 = factor_J4 * np.array([term_J4_xy * x_r, term_J4_xy * y_r, -(15 - 70 * z_r2 + 63 * z_r4) * z_r])

    a_total = a_J2 + a_J3 + a_J4
    return np.asarray(a_total * m, dtype=np.float64)


# Gravitational parameters (m^3 / s^2)
MU_SUN = 1.32712440018e20  # standard GM of Sun
MU_MOON = 4.9048695e12  # standard GM of Moon


def third_body_forces(r_eci: np.ndarray, m: float, sun_pos_eci: np.ndarray, moon_pos_eci: np.ndarray) -> np.ndarray:
    """
    Calculate the gravitational disturbance forces from the Sun and Moon.

    Parameters
    ----------
    r_eci : np.ndarray
        Position vector of the satellite in the ECI frame [m].
    m : float
        Mass of the satellite [kg].
    sun_pos_eci : np.ndarray
        Position vector of the Sun in the ECI frame [m].
    moon_pos_eci : np.ndarray
        Position vector of the Moon in the ECI frame [m].

    Returns
    -------
    np.ndarray
        The total third-body disturbance force vector in the ECI frame [N].
    """
    a = np.zeros(3)
    for mu, r_third in zip([MU_SUN, MU_MOON], [sun_pos_eci, moon_pos_eci], strict=False):
        dist = r_third - r_eci
        a += mu * (dist / np.linalg.norm(dist) ** 3 - r_third / np.linalg.norm(r_third) ** 3)

    return a * m


# rad/s earth rotates about the z axis of the eci frame with angular velocity OMEGA_E
OMEGA_E = np.array([0, 0, 0.000_072_921_158_553])


def aerodynamic_drag(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    R_BI: Quaternion,
    surfaces: list[Surface] | VectorizedSurfaces,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the aerodynamic drag force and torque on the satellite.

    This function computes the total aerodynamic force and torque using a vectorized model.

    Parameters
    ----------
    r_eci : np.ndarray
        Position vector of the satellite in the ECI frame [m].
    v_eci : np.ndarray
        Velocity of the satellite in the ECI frame [m/s].
    R_BI : Quaternion
        Rotation from the inertial frame (I) to the body frame (B).
    surfaces : List[Surface] or VectorizedSurfaces
        The satellite's surfaces.
    rho : float
        Atmospheric density at the satellite's position [kg/m^3].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the total aerodynamic force [N] and torque [N*m] vectors in the body frame.
    """
    v_atm_I = np.cross(OMEGA_E, r_eci)  # = np.array([OMEGA_E[2] * r_eci[1], OMEGA_E[2] * r_eci[0], 0])

    v_rel_B = R_BI.apply(v_eci - v_atm_I)

    v_rel_B_norm = np.linalg.norm(v_rel_B)
    v_rel_B_unit = v_rel_B / v_rel_B_norm

    if isinstance(surfaces, VectorizedSurfaces):
        normals = surfaces.normals
        centers = surfaces.centers
        areas = surfaces.areas
        sigma_t = surfaces.sigma_t
        sigma_n = surfaces.sigma_n
        S = surfaces.S
    else:
        # Convert list of Surface objects on the fly
        normals = np.array([s.normal for s in surfaces])
        centers = np.array([s.center for s in surfaces])
        areas = np.array([s.area for s in surfaces])
        sigma_t = np.array([s.sigma_t for s in surfaces])
        sigma_n = np.array([s.sigma_n for s in surfaces])
        S = np.array([s.S for s in surfaces])

    cos_thetas = normals @ v_rel_B_unit
    mask = cos_thetas >= 0.0
    if not np.any(mask):
        return np.zeros(3), np.zeros(3)

    normals_active = normals[mask]
    centers_active = centers[mask]
    areas_active = areas[mask]
    sigma_t_active = sigma_t[mask]
    sigma_n_active = sigma_n[mask]
    S_active = S[mask]
    cos_thetas_active = cos_thetas[mask]

    coeffs = -rho * (v_rel_B_norm**2) * areas_active * cos_thetas_active
    term1 = sigma_t_active[:, np.newaxis] * v_rel_B_unit[np.newaxis, :]
    term2_scalar = sigma_n_active * S_active + (2.0 - sigma_n_active - sigma_t_active) * cos_thetas_active
    term2 = term2_scalar[:, np.newaxis] * normals_active
    F_active = coeffs[:, np.newaxis] * (term1 + term2)

    F = np.sum(F_active, axis=0)
    tau = np.sum(np.cross(centers_active, F_active), axis=0)

    return F, tau


def solar_radiation_pressure(
    r_eci: np.ndarray,
    sun_pos_eci: np.ndarray,
    in_shadow: bool,  # noqa: FBT001
    R_BI: Quaternion,
    surfaces: list[Surface] | VectorizedSurfaces,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the solar radiation pressure force and torque on the satellite.

    Parameters
    ----------
    r_eci : np.ndarray
        Position vector of the satellite in the ECI frame [m].
    sun_pos_eci : np.ndarray
        Position vector of the Sun in the ECI frame [m].
    in_shadow : bool
        Flag indicating if the satellite is in Earth's shadow.
    R_BI : Quaternion
        Rotation from the inertial frame (I) to the body frame (B).
    surfaces : List[Surface] or VectorizedSurfaces
        The satellite's surfaces.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the total SRP force [N] and torque [N*m] vectors in the body frame.
    """
    if in_shadow:
        return np.zeros(3), np.zeros(3)

    sc_to_sun = sun_pos_eci - r_eci
    P = solar_radiation_pressure_constant(sc_to_sun)

    sun_dir = R_BI.apply(sc_to_sun / np.linalg.norm(sc_to_sun))

    if isinstance(surfaces, VectorizedSurfaces):
        normals = surfaces.normals
        centers = surfaces.centers
        areas = surfaces.areas
        rho_s = surfaces.rho_s
        rho_t = surfaces.rho_t
        rho_d = surfaces.rho_d
    else:
        # Convert list of Surface objects on the fly
        normals = np.array([s.normal for s in surfaces])
        centers = np.array([s.center for s in surfaces])
        areas = np.array([s.area for s in surfaces])
        rho_s = np.array([s.rho_s for s in surfaces])
        rho_t = np.array([s.rho_t for s in surfaces])
        rho_d = np.array([s.rho_d for s in surfaces])

    cos_thetas = normals @ sun_dir
    mask = cos_thetas >= 0.0
    if not np.any(mask):
        return np.zeros(3), np.zeros(3)

    normals_active = normals[mask]
    centers_active = centers[mask]
    areas_active = areas[mask]
    rho_s_active = rho_s[mask]
    rho_t_active = rho_t[mask]
    rho_d_active = rho_d[mask]
    cos_thetas_active = cos_thetas[mask]

    coeffs = -P * areas_active * cos_thetas_active
    term1 = (1.0 - rho_s_active - rho_t_active)[:, np.newaxis] * sun_dir[np.newaxis, :]
    term2_scalar = 2.0 * rho_s_active * cos_thetas_active + (2.0 / 3.0) * rho_d_active
    term2 = term2_scalar[:, np.newaxis] * normals_active
    F_active = coeffs[:, np.newaxis] * (term1 + term2)

    F = np.sum(F_active, axis=0)
    tau = np.sum(np.cross(centers_active, F_active), axis=0)

    return F, tau
