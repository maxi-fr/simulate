from typing import Any

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


class Surface:
    """
    Represents a flat rectangular surface of a satellite for aerodynamic calculations.
    """

    def __init__(
        self,
        position: np.ndarray,
        x_len: float,
        y_len: float,
        R_BS: np.ndarray,
        sigma_t: float = 0.8,
        sigma_n: float = 0.8,
        S: float = 0.05,
        rho_s: float = 0.83,
        rho_d: float = 0.0,
        rho_t: float = 0.0,
        rho_a: float = 0.17,
        name: str = "-",
    ) -> None:
        """
        Initializes the Surface.

        Parameters
        ----------
        position : np.ndarray
            Center of the surface in the body frame [m].
        x_len : float
            Width of the surface [m].
        y_len : float
            Height of the surface [m].
        R_BS : np.ndarray
            Rotation matrix from Surface frame to Body frame.
        sigma_t : float, optional
            Tangential momentum accommodation coefficient, by default 0.8.
        sigma_n : float, optional
            Normal momentum accommodation coefficient, by default 0.8.
        S : float, optional
            A parameter for the aerodynamic model, by default 0.05.
        rho_s : float, optional
            Specular reflectivity coefficient for solar radiation pressure, by default 0.83.
        rho_d : float, optional
            Diffuse reflectivity coefficient for solar radiation pressure, by default 0.0.
        rho_t : float, optional
            Transmissivity coefficient for solar radiation pressure, by default 0.0.
        rho_a : float, optional
            Absorptivity coefficient for solar radiation pressure, by default 0.17.
        name : str, optional
            Name of the surface, by default "-".
        """
        self.pos = position
        self.x_len = x_len
        self.y_len = y_len
        self.x_half = x_len / 2
        self.y_half = y_len / 2

        self.R_BS = R_BS

        self.normal = self.R_BS[:, 2]
        self.x_axis = x_len * self.R_BS[:, 0]
        self.y_axis = y_len * self.R_BS[:, 1]

        self.center = self.pos + self.x_axis / 2 + self.y_axis / 2

        self.area = self.x_len * self.y_len

        self.sigma_t = sigma_t
        self.sigma_n = sigma_n
        self.S = S
        self.rho_s = rho_s
        self.rho_d = rho_d
        self.rho_t = rho_t
        self.rho_a = rho_a
        self.name = name

    @classmethod
    def from_dict(cls, name: str, surface_dict: dict[str, Any]) -> "Surface":
        """
        Creates a Surface from a dictionary.

        Parameters
        ----------
        name : str
            Name of the surface.
        surface_dict : dict
            Dictionary containing surface parameters.

        Returns
        -------
        Surface
            The created Surface object.

        Raises
        ------
        ValueError
            If the orientation matrix is invalid.
        """
        R_BS = np.array(surface_dict["Rotation (Surface frame to Body)"])

        if not np.allclose(R_BS.T @ R_BS, np.eye(3), atol=1e-6):
            msg = "Surface.from_eos_panel: Orientation matrix is not orthonormal after transpose."
            raise ValueError(msg)
        if np.linalg.det(R_BS) < 0.0:
            msg = "Surface.from_eos_panel: Orientation matrix has det < 0 (reflection), expected proper rotation."
            raise ValueError(msg)

        return cls(
            np.array(surface_dict["Origin"]),
            surface_dict.get("DimX", 0.1),
            surface_dict.get("DimY", 0.1),
            R_BS,
            name=name,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the Surface to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the surface.
        """
        return {
            "Origin": self.pos.tolist(),
            "DimX": self.x_len,
            "DimY": self.y_len,
            "Rotation (Surface frame to Body)": self.R_BS.tolist(),
            "sigma_t": self.sigma_t,
            "sigma_n": self.sigma_n,
            "S": self.S,
            "rho_s": self.rho_s,
            "rho_d": self.rho_d,
            "rho_t": self.rho_t,
            "rho_a": self.rho_a,
        }

    def corners(self) -> np.ndarray:
        """
        Returns the corners of the surface.

        Returns
        -------
        np.ndarray
            Array of corner positions.
        """
        p = self.pos
        return np.array([p, p + self.x_axis, p + self.x_axis + self.y_axis, p + self.y_axis])

    def self_occlusion(self, direction: np.ndarray, surfaces: list["Surface"]) -> bool:
        """
        Checks if this surface is occluded by any other surface from a given direction.
        Occlusion is determined/approximated by if a ray coming from a specific direction passing through
        the geometric center of the surface has passed through other surfaces on the way.

        Parameters
        ----------
        direction : np.ndarray, shape (3,)
            The direction vector of incoming atmospheric flow.
        surfaces : List[Surface]
            A list of all surfaces on the satellite.

        Returns
        -------
        bool
            True if the surface is occluded, False otherwise.
        """
        for s in surfaces:
            if s == self:
                continue

            if s.passed_through(self.center, direction):
                return True

        return False

    def passed_through(self, point: np.ndarray, direction: np.ndarray) -> bool:
        """
        Checks if a ray defined by a point and direction has intersected this surface before passing through the point.

        Parameters
        ----------
        point : np.ndarray, shape (3,)
            The origin point of the ray.
        direction : np.ndarray, shape (3,)
            The direction vector of the ray.

        Returns
        -------
        bool
            True if the ray passes through the surface, False otherwise.
        """
        dn = np.dot(direction, self.normal)

        # ray is parallel
        if dn == 0:
            return False

        t = np.dot(self.center - point, self.normal) / dn

        if t >= 0:
            return False

        p = point + t * direction

        p_xy = self.R_BS[:, :2].T @ (p - self.center)

        return bool(np.all(np.abs(p_xy) <= np.array([self.x_half, self.x_half])))
