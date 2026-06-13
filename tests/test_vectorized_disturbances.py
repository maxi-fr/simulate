# ruff: noqa: N803, N806
import numpy as np
import pytest

from spacecraft.disturbances import (
    OMEGA_E,
    aerodynamic_drag,
    solar_radiation_pressure,
)
from spacecraft.environment import solar_radiation_pressure_constant
from spacecraft.quaternion import Quaternion
from spacecraft.surface import Surface, VectorizedSurfaces


def _old_aerodynamic_drag(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    R_BI: Quaternion,
    surfaces: list[Surface],
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Old loop-based aerodynamic drag model to verify equivalence."""
    v_atm_I = np.cross(OMEGA_E, r_eci)
    v_rel_B = R_BI.apply(v_eci - v_atm_I)
    v_rel_B_norm = np.linalg.norm(v_rel_B)
    v_rel_B_unit = v_rel_B / v_rel_B_norm

    F = np.zeros(3)
    tau = np.zeros(3)

    for s in surfaces:
        cos_theta_i = np.dot(v_rel_B_unit, s.normal)
        if cos_theta_i < 0:
            continue
        F_i = (
            -rho
            * v_rel_B_norm**2
            * s.area
            * cos_theta_i
            * (s.sigma_t * v_rel_B_unit + (s.sigma_n * s.S + (2 - s.sigma_n - s.sigma_t) * cos_theta_i) * s.normal)
        )
        tau += np.cross(s.center, F_i)
        F += F_i

    return F, tau


def _old_solar_radiation_pressure(
    r_eci: np.ndarray,
    sun_pos_eci: np.ndarray,
    in_shadow: bool,  # noqa: FBT001
    R_BI: Quaternion,
    surfaces: list[Surface],
) -> tuple[np.ndarray, np.ndarray]:
    """Old loop-based SRP model to verify equivalence."""
    if in_shadow:
        return np.zeros(3), np.zeros(3)

    sc_to_sun = sun_pos_eci - r_eci
    P = solar_radiation_pressure_constant(sc_to_sun)
    sun_dir = R_BI.apply(sc_to_sun / np.linalg.norm(sc_to_sun))

    F = np.zeros(3)
    tau = np.zeros(3)

    for s in surfaces:
        cos_theta_i = np.dot(sun_dir, s.normal)
        if cos_theta_i < 0:
            continue
        F_i = (
            -P
            * s.area
            * cos_theta_i
            * ((1 - s.rho_s - s.rho_t) * sun_dir + (2 * s.rho_s * cos_theta_i + 2 / 3 * s.rho_d) * s.normal)
        )
        tau += np.cross(s.center, F_i)
        F += F_i

    return F, tau


def _generate_random_surfaces(num_surfaces: int = 6) -> list[Surface]:
    """Generate a list of random Surfaces with varying normals, areas, and coefficients."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    surfaces = []
    for i in range(num_surfaces):
        pos = rng.uniform(-0.5, 0.5, size=3)
        x_len = rng.uniform(0.1, 0.5)
        y_len = rng.uniform(0.1, 0.5)

        # Generate a random orthonormal rotation matrix R_BS
        u, _, vh = np.linalg.svd(rng.normal(size=(3, 3)))
        R_BS = u @ vh
        if np.linalg.det(R_BS) < 0:
            R_BS[:, 0] = -R_BS[:, 0]

        s = Surface(
            position=pos,
            x_len=x_len,
            y_len=y_len,
            R_BS=R_BS,
            sigma_t=rng.uniform(0.5, 1.0),
            sigma_n=rng.uniform(0.5, 1.0),
            S=rng.uniform(0.01, 0.1),
            rho_s=rng.uniform(0.1, 0.9),
            rho_d=rng.uniform(0.0, 0.1),
            rho_t=rng.uniform(0.0, 0.1),
            rho_a=rng.uniform(0.0, 0.1),
            name=f"Surface_{i}",
        )
        surfaces.append(s)
    return surfaces


@pytest.mark.parametrize("num_cases", [10])
def test_aerodynamic_drag_equivalence(num_cases: int) -> None:
    """Verify that vectorized aerodynamic_drag matches the loop-based implementation."""
    rng = np.random.default_rng(100)
    surfaces = _generate_random_surfaces(6)
    vectorized_surfaces = VectorizedSurfaces(surfaces)

    for _ in range(num_cases):
        r_eci = rng.uniform(-7.5e6, 7.5e6, size=3)
        v_eci = rng.uniform(-8.0e3, 8.0e3, size=3)
        q_arr = rng.normal(size=4)
        q_arr /= np.linalg.norm(q_arr)
        R_BI = Quaternion.from_array(q_arr)
        rho = rng.uniform(1.0e-13, 1.0e-11)

        # 1. Compute using the old loop-based reference model
        F_old, tau_old = _old_aerodynamic_drag(r_eci, v_eci, R_BI, surfaces, rho)

        # 2. Compute using the new model with raw list of Surface
        F_new_list, tau_new_list = aerodynamic_drag(r_eci, v_eci, R_BI, surfaces, rho)

        # 3. Compute using the new model with VectorizedSurfaces
        F_new_vec, tau_new_vec = aerodynamic_drag(r_eci, v_eci, R_BI, vectorized_surfaces, rho)

        # Verify equivalence
        np.testing.assert_allclose(F_new_list, F_old, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(tau_new_list, tau_old, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(F_new_vec, F_old, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(tau_new_vec, tau_old, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("num_cases", [10])
def test_solar_radiation_pressure_equivalence(num_cases: int) -> None:
    """Verify that vectorized solar_radiation_pressure matches the loop-based implementation."""
    rng = np.random.default_rng(200)
    surfaces = _generate_random_surfaces(6)
    vectorized_surfaces = VectorizedSurfaces(surfaces)

    for _ in range(num_cases):
        r_eci = rng.uniform(-7.5e6, 7.5e6, size=3)
        sun_pos_eci = rng.uniform(-1.5e11, 1.5e11, size=3)
        in_shadow = rng.choice([True, False])
        q_arr = rng.normal(size=4)
        q_arr /= np.linalg.norm(q_arr)
        R_BI = Quaternion.from_array(q_arr)

        # 1. Compute using the old loop-based reference model
        F_old, tau_old = _old_solar_radiation_pressure(r_eci, sun_pos_eci, in_shadow, R_BI, surfaces)

        # 2. Compute using the new model with raw list of Surface
        F_new_list, tau_new_list = solar_radiation_pressure(r_eci, sun_pos_eci, in_shadow, R_BI, surfaces)

        # 3. Compute using the new model with VectorizedSurfaces
        F_new_vec, tau_new_vec = solar_radiation_pressure(r_eci, sun_pos_eci, in_shadow, R_BI, vectorized_surfaces)

        # Verify equivalence
        np.testing.assert_allclose(F_new_list, F_old, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(tau_new_list, tau_old, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(F_new_vec, F_old, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(tau_new_vec, tau_old, rtol=1e-12, atol=1e-18)
