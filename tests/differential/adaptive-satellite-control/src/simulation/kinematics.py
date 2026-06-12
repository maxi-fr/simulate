import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from scipy.spatial.transform import Rotation as R
from utils import Quaternion


def orc_to_eci(r: np.ndarray, v: np.ndarray) -> R:
    """
    Calculates the rotation from the Orbital Reference Frame (ORC) to the Earth-Centered Inertial (ECI) frame.

    Parameters
    ----------
    r : np.ndarray, shape (3,) or (N, 3)
        Position vector in the ECI frame.
    v : np.ndarray, shape (3,) or (N, 3)
        Velocity vector in the ECI frame.

    Returns
    -------
    scipy.spatial.transform.Rotation
        Rotation object representing the transformation from ORC to ECI.

    """
    o_3I = -r / np.linalg.norm(r, axis=-1, keepdims=True)

    cross_v_z = np.cross(v, -o_3I)
    o_2I = cross_v_z / np.linalg.norm(cross_v_z, axis=-1, keepdims=True)

    o_1I = np.cross(o_2I, o_3I)

    return R.from_matrix(np.stack([o_1I, o_2I, o_3I], axis=-1))


def euler_ocr_to_sbc(roll_deg: float, pitch_deg: float, yaw_deg: float) -> R:
    """
    Creates a Rotation object from Euler angles (Roll, Pitch, Yaw).

    The intrinsic rotation sequence is defined as Y-X-Z (Pitch-Roll-Yaw).

    Parameters
    ----------
    roll_deg : float
        Roll angle [deg].
    pitch_deg : float
        Pitch angle [deg].
    yaw_deg : float
        Yaw angle [deg].

    Returns
    -------
    scipy.spatial.transform.Rotation
        Rotation object representing the transformation from ORC to SBC.
    """
    return R.from_euler("YXZ", [pitch_deg, roll_deg, yaw_deg], degrees=True)


def orc_to_sbc(q_BI: Quaternion, r_eci: np.ndarray, v_eci: np.ndarray) -> Quaternion:
    """
    Calculates rotation from ORC to SBC using the attitude quaternion as well as position and velocity vectors.

    This is achieved by composing the rotation from ECI to the body frame (from the quaternion)
    with the rotation from the ORC to the ECI frame.

    Parameters
    ----------
    q_BI : Quaternion
        Attitude quaternion for the rotation from ECI (I) to the body frame (B).
    r_eci : np.ndarray, shape (3,)
        Position vector in the ECI frame.
    v_eci : np.ndarray, shape (3,)
        Velocity vector in the ECI frame.

    Returns
    -------
    Quaternion
        Rotation quaternion representing the transformation from ORC to SBC.
    """
    return q_BI * Quaternion.from_scipy(orc_to_eci(r_eci, v_eci), canonical=False)  # R_BI * R_IO = R_BO


def to_euler(q_BI: Quaternion, r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """
    Calculates the Euler angles (Roll, Pitch, Yaw) from the attitude quaternion and orbital state.

    The Euler angles represent the rotation from the Orbital Reference Frame (ORC) to the
    Satellite Body Frame (SBC). The intrinsic rotation sequence is Y-X-Z (Pitch-Roll-Yaw).

    Parameters
    ----------
    q_BI : Quaternion
        Attitude quaternion for the rotation from ECI (I) to the body frame (B).
    r_eci : np.ndarray, shape (3,)
        Position vector in the ECI frame.
    v_eci : np.ndarray, shape (3,)
        Velocity vector in the ECI frame.

    Returns
    -------
    np.ndarray, shape (3,)
        Euler angles [Roll, Pitch, Yaw] in degrees.

    """
    R_BO = orc_to_sbc(q_BI, r_eci, v_eci)
    euler = np.atleast_2d(R_BO.to_scipy().as_euler("YXZ", degrees=True))

    euler[:, [0, 1]] = euler[:, [1, 0]]
    return euler.squeeze()


def eci_to_sbc(q_BI: np.ndarray) -> Quaternion:
    """
    Creates a Quaternion object from the attitude quaternion array.

    Parameters
    ----------
    q_BI : np.ndarray
        Attitude quaternion [qx, qy, qz, qw] (scalar last) representing the rotation
        from the ECI frame to the Body frame.

    Returns
    -------
    Quaternion
        Quaternion object representing the transformation from ECI to SBC.
    """
    return Quaternion.from_array(q_BI, scalar_first=False)


def eci_to_geodedic(pos_eci: np.ndarray) -> tuple[float, float, float]:
    """
    Converts ECI position to geodetic coordinates.

    Parameters
    ----------
    pos_eci : np.ndarray
        Position vector in the ECI frame [m].

    Returns
    -------
    tuple[float, float, float]
        A tuple containing (latitude [deg], longitude [deg], altitude [m]).
    """
    loc = EarthLocation.from_geocentric(*(pos_eci * u.m)).to_geodetic("WGS84")

    lat = loc.lat.value
    lon = loc.lon.value
    alt = loc.height.to(u.m).value

    return lat, lon, alt
