# ruff: noqa: N806, N803
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


class Surface:
    """Represents a flat rectangular surface of a satellite for aerodynamic calculations."""

    def __init__(  # noqa: PLR0913
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
        Initialize the Surface.

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

        self.center: np.ndarray = self.pos + self.x_axis / 2 + self.y_axis / 2

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
        Create a Surface from a dictionary.

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

    def plot(
        self,
        ax: Axes,
        R_FB: Rotation | None = None,
        color: str = "cyan",
        alpha: float = 0.4,
        normal_scale: float = 0.05,
    ) -> None:
        """
        Plot the rectangular surface and its normal vector in a 3D matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The 3D axis to plot on.
        R_FB : scipy.spatial.transform.Rotation, optional
            A rotation from the body frame to another frame, by default None.
        color : str, optional
            Color of the surface, by default "cyan".
        alpha : float, optional
            Transparency of the surface, by default 0.4.
        normal_scale : float, optional
            Length scaling factor for the normal vector, by default 0.05.
        """
        if R_FB is None:
            pos = self.pos
            x_axis = self.x_axis
            y_axis = self.y_axis
            center = self.center
            normal = self.normal
        else:
            pos = R_FB.apply(self.pos)
            x_axis = R_FB.apply(self.x_axis)
            y_axis = R_FB.apply(self.y_axis)
            center = R_FB.apply(self.center)
            normal = R_FB.apply(self.normal)

        corners = np.array([pos, pos + x_axis, pos + x_axis + y_axis, pos + y_axis])

        corners_closed = np.vstack([corners, corners[0]])

        ax.plot(corners_closed[:, 0], corners_closed[:, 1], corners_closed[:, 2], color=color)
        ax.add_collection3d(Poly3DCollection([corners], alpha=alpha, facecolor=color))  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
        ax.quiver(*center, *normal, length=normal_scale, color="red")

    def corners(self) -> np.ndarray:
        """
        Return the corners of the surface.

        Returns
        -------
        np.ndarray
            Array of corner positions.
        """
        p = self.pos
        return np.array([p, p + self.x_axis, p + self.x_axis + self.y_axis, p + self.y_axis])
