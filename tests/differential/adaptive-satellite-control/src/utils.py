import contextlib
import csv
import datetime
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation as R


class Logger:
    """
    A CSV logger for recording simulation data.
    """

    def __init__(self, log_file: str, header: list[str]) -> None:
        """
        Initializes the Logger.

        Parameters
        ----------
        log_file : str
            Base path for the log file. If it exists, a numbered suffix is appended.
        header : List[str]
            List of column names for the CSV file.
        """
        base = log_file.rsplit(".", 1)[0]
        ext = ".csv"
        candidate = base + ext

        if Path(candidate).exists():
            for i in range(1000):
                candidate = f"{base}_{i}{ext}"
                if not Path(candidate).exists():
                    break

        self.log_file = open(candidate, "a", buffering=8192, newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(header)
        self.row_len = len(header)

    def log(self, row: list[Any]) -> None:
        """
        Writes a row of data to the log file.

        Parameters
        ----------
        row : List[Any]
            Data row to be logged. Must match the length of the header.

        Raises
        ------
        ValueError
            If the row length does not match the header length.
        """
        if len(row) != self.row_len:
            msg = f"Logger: expected {self.row_len} columns, got {len(row)}"
            raise ValueError(msg)

        self.csv_writer.writerow(row)

    def close(self) -> None:
        """
        Closes the log file.
        """
        if not self.log_file.closed:
            self.log_file.close()

    def __del__(self) -> None:
        """
        Destructor to ensure the log file is closed.
        """
        with contextlib.suppress(Exception):
            self.close()


def string_to_timedelta(total_time: str) -> datetime.timedelta:
    """
    Converts a time string in format "h:m:s", "m:s", or "s" to a timedelta.

    Parameters
    ----------
    total_time : str
        The time string.

    Returns
    -------
    datetime.timedelta
        The corresponding timedelta object.

    Raises
    ------
    ValueError
        If the string format is incorrect.
    """
    match tuple(map(float, total_time.split(":"))):
        case (hours, minutes, seconds):
            return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        case (minutes, seconds):
            return datetime.timedelta(minutes=minutes, seconds=seconds)
        case (seconds,):
            return datetime.timedelta(seconds=seconds)
    msg = f"String '{total_time}' not in format h:m:s"
    raise ValueError(msg)


class PiecewiseConstant:
    """
    Represents a piecewise constant function.
    """

    def __init__(self, fn: Callable[..., Any], time_bucket_fn: Callable[[Any], Any]) -> None:
        """
        Initializes the PiecewiseConstant function wrapper.

        Parameters
        ----------
        fn : Callable
            The function to be evaluated.
        time_bucket_fn : Callable
            Function that maps time to a bucket identifier.
        """
        self.fn = fn
        self.time_bucket_fn = time_bucket_fn
        self._last_bucket = None
        self._value = None

    def __call__(self, t: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Evaluates the function.

        Parameters
        ----------
        t : Any
            The time or input variable used for bucketing.
        *args : Any
            Additional positional arguments for the function.
        **kwargs : Any
            Additional keyword arguments for the function.

        Returns
        -------
        Any
            The result of the function evaluation.
        """
        bucket = self.time_bucket_fn(t)

        if bucket != self._last_bucket:
            self._value = self.fn(bucket, *args, **kwargs)
            self._last_bucket = bucket

        return self._value

    def reset(self) -> None:
        """
        Resets the cached bucket and value.
        """
        self._last_bucket = None
        self._value = None


def floor_time_to_minute(t: datetime.datetime) -> datetime.datetime:
    """
    Floors the time to the nearest minute.

    Parameters
    ----------
    t : datetime.datetime
        Input datetime.

    Returns
    -------
    datetime.datetime
        Floored datetime.
    """
    return t.replace(second=0, microsecond=0)


def floor_time_to_second(t: datetime.datetime) -> datetime.datetime:
    """
    Floors the time to the nearest second.

    Parameters
    ----------
    t : datetime.datetime
        Input datetime.

    Returns
    -------
    datetime.datetime
        Floored datetime.
    """
    return t.replace(microsecond=0)


# Type aliases for numpy arrays with known shapes
FloatArray = NDArray[np.float64]
Vec3 = Annotated[FloatArray, Literal[3]]
Vec4 = Annotated[FloatArray, Literal[4]]
Mat3x3 = Annotated[FloatArray, Literal[3]]
Mat4x3 = Annotated[FloatArray, Literal[4, 3]]


@dataclass(frozen=True)
class Quaternion:
    vec: Vec3
    scalar: float

    def __post_init__(self) -> None:
        """
        Validates the quaternion vector.
        """
        object.__setattr__(self, "vec", np.asarray(self.vec))

    @classmethod
    def from_array(cls, q: ArrayLike, scalar_first: bool = False) -> "Quaternion":
        """
        Creates a Quaternion from an array-like object.

        Parameters
        ----------
        q : ArrayLike
            Input array representing the quaternion. Expected shape is (4,).
        scalar_first : bool, optional
            If True, the input array is interpreted as [scalar, v1, v2, v3].
            If False, it is interpreted as [v1, v2, v3, scalar].
            Default is False.

        Returns
        -------
        Quaternion
            The created Quaternion object.
        """
        q = np.asarray(q)
        if scalar_first:
            return cls(q[1:], q[0])
        return cls(q[:3], q[3])

    def to_array(self, scalar_first: bool = False) -> Vec4:
        """
        Converts the Quaternion to a numpy array.

        Parameters
        ----------
        scalar_first : bool, optional
            If True, returns [scalar, v1, v2, v3].
            If False, returns [v1, v2, v3, scalar].
            Default is False.

        Returns
        -------
        np.ndarray
            The quaternion as a numpy array.
        """
        if scalar_first:
            return np.array((self.scalar, *self.vec))

        return np.array((*self.vec, self.scalar))

    @classmethod
    def from_scipy(cls, rot: R, canonical: bool = True) -> "Quaternion":
        """
        Creates a Quaternion from a scipy.spatial.transform.Rotation object.

        Note: Scipy uses the Hamilton convention for quaternions. This class uses the JPL convention.
        The conversion handles the difference (JPL q = [-v, w] relative to Hamilton q = [v, w] for the same rotation).

        Parameters
        ----------
        rot : scipy.spatial.transform.Rotation
            The rotation object.
        canonical : bool, optional
            Whether to map the quaternion to the canonical hemisphere (w > 0). Default is True.

        Returns
        -------
        Quaternion
            The corresponding JPL quaternion.
        """
        q = rot.as_quat(canonical=canonical, scalar_first=False)
        return cls(-q[:3], q[3])

    def to_scipy(self) -> R:
        """
        Converts the Quaternion to a scipy.spatial.transform.Rotation object.

        Note: Handles the conversion from JPL convention to Hamilton convention used by Scipy.

        Returns
        -------
        scipy.spatial.transform.Rotation
            The rotation object.
        """
        return R.from_quat(
            self.conjugate().to_array(scalar_first=False)
        )  # conjugate because scipy uses hamilton convention

    def to_rot_mat(self) -> Mat3x3:
        """
        Converts the quaternion to a rotation matrix.

        Returns
        -------
        np.ndarray
            The 3x3 rotation matrix.
        """
        return self.to_scipy().as_matrix()

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        r"""
        Multiplies two quaternions according to the JPL convention.

        Note: This is NOT the Hamilton product.

        Formula:
        .. math::
            \begin{bmatrix}
            q_4 \overline{\mathbf{q}}_{1: 3}+\bar{q}_4 \mathbf{q}_{1: 3}-\overline{\mathbf{q}}_{1: 3} \times \mathbf{q}_{1: 3} \\
            \bar{q}_4 q_4-\overline{\mathbf{q}}_{1: 3} \cdot \mathbf{q}_{1: 3}
            \end{bmatrix}

        Parameters
        ----------
        other : Quaternion
            The right-hand side quaternion.

        Returns
        -------
        Quaternion
            The product of the two quaternions.
        """
        qv = self.vec
        w = self.scalar

        qv_other = other.vec
        w_other = other.scalar

        w_ret = w * w_other - np.dot(qv, qv_other)
        v_ret = w_other * qv + w * qv_other - np.cross(qv, qv_other)

        return Quaternion(v_ret, w_ret)

    def mult_jpl(self, rhs: "Quaternion") -> "Quaternion":
        """
        Performs JPL quaternion multiplication (same as * operator).

        Parameters
        ----------
        rhs : Quaternion
            The right-hand side quaternion.

        Returns
        -------
        Quaternion
            The result of self * rhs.
        """
        return self * rhs

    def mult_hamilton(self, rhs: "Quaternion") -> "Quaternion":
        """
        Performs Hamilton quaternion multiplication.

        Parameters
        ----------
        rhs : Quaternion
            The right-hand side quaternion.

        Returns
        -------
        Quaternion
            The Hamilton product of self and rhs.
        """
        return rhs * self

    def conjugate(self) -> "Quaternion":
        """
        Calculates the conjugate of the quaternion.

        For a unit quaternion, this represents the inverse rotation.

        Returns
        -------
        Quaternion
            The conjugate quaternion [-v, w].
        """
        return Quaternion(-self.vec, self.scalar)

    def apply(self, v: ArrayLike) -> Vec3:
        """
        Rotates a vector v using this quaternion.

        Parameters
        ----------
        v : ArrayLike
            The 3D vector to rotate.

        Returns
        -------
        np.ndarray
            The rotated vector.
        """
        v = np.asarray(v)

        qv = self.vec
        w = self.scalar

        t = -2.0 * np.cross(qv, v)

        return v + w * t - np.cross(qv, t)

    def kinematics(self, omega: Vec3, scalar_first: bool = False) -> Vec4:
        """
        Calculates the time derivative of the quaternion given an angular velocity.

        dq/dt = 0.5 * Xi(q) * omega

        Parameters
        ----------
        omega : np.ndarray
            Angular velocity vector in the body frame [rad/s].
        scalar_first : bool, optional
            If True, returns the derivative as [dw, dx, dy, dz].
            If False, returns [dx, dy, dz, dw].
            Default is False.

        Returns
        -------
        np.ndarray
            The time derivative of the quaternion.
        """
        if scalar_first:
            return np.roll(self.xi @ omega, 1)

        return 0.5 * self.xi @ omega

    def exact_integration(self, omega: Vec3, dt: float) -> "Quaternion":
        """
        Integrates the quaternion given a constant angular velocity over a time step.

        Parameters
        ----------
        omega : np.ndarray
            Constant angular velocity vector in the body frame [rad/s].
        dt : float
             Time step [s].

        Returns
        -------
        Quaternion
            The new quaternion after integration.
        """
        omega = np.asarray(omega) * dt

        omega_norm = np.linalg.norm(omega)
        if omega_norm < 1e-9:  # Avoid division by zero for very small angular velocities
            return self

        delta_theta = omega_norm
        delta_q_vec = (omega / omega_norm) * np.sin(delta_theta / 2)
        delta_q_scalar = np.cos(delta_theta / 2)
        delta_q = Quaternion(delta_q_vec, delta_q_scalar)

        return self * delta_q

    @cached_property
    def xi(self) -> Mat4x3:
        r"""
        Computes the xi matrix for JPL quaternion kinematics.

        The kinematics equation is:
        .. math::
            \dot{q} = \frac{1}{2} \Xi(\mathbf{q}) \omega

        where:
        .. math::
            \Xi(\mathbf{q}) :=
            \begin{bmatrix}
                q_4 I_3+\left[\mathbf{q}_{1: 3} \times\right] \\
                -\mathbf{q}_{1: 3}^T
            \end{bmatrix} =
            \begin{bmatrix}
                q_4 & -q_3 & q_2 \\
                q_3 & q_4 & -q_1 \\
                -q_2 & q_1 & q_4 \\
                -q_1 & -q_2 & -q_3
            \end{bmatrix}

        Returns
        -------
        np.ndarray
            The 4x3 xi matrix.
        """
        qw = self.scalar
        qx, qy, qz = self.vec

        return np.array([[qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw], [-qx, -qy, -qz]])
