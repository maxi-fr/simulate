from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

# A measurement model maps (t, state, input) to a truth output ``y = h(t, x, u)``. Simple
# models are plain functions; parametrized ones are classes implementing ``__call__``. The
# owning :class:`~simulate.sensor.Sensor` provides the sample rate (``dt``), ZOH and noise.
type MeasurementModel = Callable[[float, float | np.ndarray, float | np.ndarray], float | np.ndarray]


class LinearMeasurement:
    """Generic linear measurement model using state space matrices C and D (``y = C x + D u``)."""

    def __init__(self, c: ArrayLike, d: ArrayLike) -> None:
        """Initialize the linear measurement with the output matrices C and D."""
        self.c = np.atleast_2d(c)
        self.d = np.atleast_2d(d)

    def __call__(
        self,
        _t: float,
        x: float | np.ndarray,
        u: float | np.ndarray,
    ) -> np.ndarray:
        """Compute the output from the current state and input."""
        x_arr = np.atleast_1d(x)
        u_arr = np.atleast_1d(u)
        return cast("np.ndarray", self.c @ x_arr + self.d @ u_arr)
