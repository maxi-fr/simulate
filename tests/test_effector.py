import datetime

import numpy as np
import pytest

from rigid_body.disturbances import aerodynamic_drag, solar_radiation_pressure, third_body_forces
from rigid_body.effector import (
    AerodynamicDrag,
    EarthGravity,
    RigidBodyState,
    SolarRadiationPressure,
    ThirdBody,
)
from rigid_body.environment import atmosphere_density_msis, is_in_shadow, moon_position, sun_position
from rigid_body.frames import eci_to_geodedic
from rigid_body.orbit_dynamics import MU
from rigid_body.quaternion import Quaternion
from rigid_body.rigid_body import RigidBodyDynamics
from rigid_body.surface import Surface

MU_EARTH = 3.986e14


def _state(r: np.ndarray, q: np.ndarray) -> RigidBodyState:
    """Build a RigidBodyState with the given position and attitude (velocity/omega zero)."""
    return RigidBodyState(
        r_eci=r, v_eci=np.zeros(3), q_bi=Quaternion.from_array(q, scalar_first=False), omega_b_bi=np.zeros(3)
    )


def _bound_eg(inertia: np.ndarray) -> EarthGravity:
    """An EarthGravity with its mass and inertia bound, as RigidBodyDynamics would."""
    eg = EarthGravity(mu=MU_EARTH)
    eg.bind(mass=500.0, inertia=inertia)
    return eg


def test_zero_torque_at_principal_axis_equilibrium() -> None:
    """With a principal axis aligned with nadir, the gravity-gradient torque vanishes."""
    inertia = np.diag([100.0, 200.0, 300.0])
    eg = _bound_eg(inertia)
    r = np.array([7.0e6, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])  # identity: nadir = -x = body principal axis

    _, torque, _ = eg.calc_contributions(0.0, _state(r, q), np.zeros(0), np.zeros(0))
    assert np.allclose(torque, 0.0, atol=1e-12)


def test_known_torque_value() -> None:
    """Gravity-gradient torque matches the hand-computed analytic value (30 deg about z)."""
    inertia = np.diag([100.0, 200.0, 300.0])
    eg = _bound_eg(inertia)
    r = np.array([7.0e6, 0.0, 0.0])
    half = np.deg2rad(15.0)  # 30 deg rotation about body z
    q = np.array([0.0, 0.0, np.sin(half), np.cos(half)])

    _, torque, _ = eg.calc_contributions(0.0, _state(r, q), np.zeros(0), np.zeros(0))
    assert np.allclose(torque, np.array([0.0, 0.0, -1.5096e-4]), rtol=1e-3, atol=1e-9)


def test_torque_scales_as_inverse_r_cubed() -> None:
    """Doubling the orbital radius reduces the torque magnitude by a factor of 8."""
    inertia = np.diag([100.0, 200.0, 300.0])
    eg = _bound_eg(inertia)
    half = np.deg2rad(15.0)
    q = np.array([0.0, 0.0, np.sin(half), np.cos(half)])

    _, torque_near, _ = eg.calc_contributions(0.0, _state(np.array([7.0e6, 0.0, 0.0]), q), np.zeros(0), np.zeros(0))
    _, torque_far, _ = eg.calc_contributions(0.0, _state(np.array([1.4e7, 0.0, 0.0]), q), np.zeros(0), np.zeros(0))

    assert np.isclose(np.linalg.norm(torque_near) / np.linalg.norm(torque_far), 8.0, rtol=1e-9)


# Fixed epoch for the environmental-effector tests (drives the hardcoded ephemerides).
EPOCH = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)


def _x_surface() -> Surface:
    """A 1 m^2 surface centred at the origin whose outward normal is the body +x axis."""
    r_bs = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # third column = +x normal
    return Surface(position=np.zeros(3), x_len=1.0, y_len=1.0, R_BS=r_bs)


def test_third_body_returns_inertial_force() -> None:
    """ThirdBody returns the Sun/Moon disturbance force in the inertial frame."""
    eff = ThirdBody(epoch=EPOCH)
    eff.bind(mass=500.0, inertia=np.eye(3))
    half = np.deg2rad(20.0)
    state = _state(np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 0.0, np.sin(half), np.cos(half)]))

    force, torque, momentum = eff.calc_contributions(0.0, state, np.zeros(0), np.zeros(0))

    expected_eci = third_body_forces(state.r_eci, 500.0, sun_position(EPOCH), moon_position(EPOCH))
    assert np.allclose(force, expected_eci)
    assert np.allclose(torque, 0.0)
    assert np.allclose(momentum, 0.0)


def test_third_body_unbound_mass_raises() -> None:
    """Evaluating ThirdBody before its mass is bound is an error."""
    eff = ThirdBody(epoch=EPOCH)
    state = _state(np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    with pytest.raises(RuntimeError, match="unbound"):
        eff.calc_contributions(0.0, state, np.zeros(0), np.zeros(0))


def test_solar_radiation_pressure_matches_disturbance() -> None:
    """SRP effector reproduces the disturbance function for the hardcoded Sun position."""
    surfaces = [_x_surface()]
    eff = SolarRadiationPressure(surfaces=surfaces, epoch=EPOCH)
    sun_pos = sun_position(EPOCH)
    # Position on the sun side of Earth so the satellite is not eclipsed.
    state = _state(7.0e6 * sun_pos / np.linalg.norm(sun_pos), np.array([0.0, 0.0, 0.0, 1.0]))

    force, torque, momentum = eff.calc_contributions(0.0, state, np.zeros(0), np.zeros(0))

    shadow = is_in_shadow(state.r_eci, sun_pos)
    f_exp, tau_exp = solar_radiation_pressure(state.r_eci, sun_pos, shadow, state.q_bi, surfaces)
    assert not shadow
    assert np.allclose(force, state.q_bi.conjugate().apply(f_exp))
    assert np.allclose(torque, tau_exp)
    assert np.allclose(momentum, 0.0)


def test_solar_radiation_pressure_zero_in_shadow() -> None:
    """A satellite behind Earth (anti-sun) is eclipsed and feels no SRP."""
    eff = SolarRadiationPressure(surfaces=[_x_surface()], epoch=EPOCH)
    sun_pos = sun_position(EPOCH)
    state = _state(-7.0e6 * sun_pos / np.linalg.norm(sun_pos), np.array([0.0, 0.0, 0.0, 1.0]))

    assert is_in_shadow(state.r_eci, sun_pos)
    force, torque, _ = eff.calc_contributions(0.0, state, np.zeros(0), np.zeros(0))
    assert np.allclose(force, 0.0)
    assert np.allclose(torque, 0.0)


def test_aerodynamic_drag_matches_disturbance() -> None:
    """Aerodynamic drag effector reproduces the disturbance function with the MSIS density."""
    surfaces = [_x_surface()]
    eff = AerodynamicDrag(surfaces=surfaces, epoch=EPOCH)
    r_eci = np.array([7.0e6, 0.0, 0.0])
    state = RigidBodyState(
        r_eci=r_eci,
        v_eci=np.array([0.0, 7.5e3, 0.0]),  # orbital-speed ram flow
        q_bi=Quaternion.from_array(np.array([0.0, 0.0, 0.0, 1.0])),
        omega_b_bi=np.zeros(3),
    )

    force, torque, momentum = eff.calc_contributions(0.0, state, np.zeros(0), np.zeros(0))

    lat_deg, lon_deg, alt_m = eci_to_geodedic(r_eci)
    rho = atmosphere_density_msis(EPOCH, float(lat_deg), float(lon_deg), float(alt_m))
    f_exp, tau_exp = aerodynamic_drag(state.r_eci, state.v_eci, state.q_bi, surfaces, rho)

    assert np.allclose(force, state.q_bi.conjugate().apply(f_exp))
    assert np.allclose(torque, tau_exp)
    assert np.allclose(momentum, 0.0)


def test_earth_gravity_points_to_nadir_with_inverse_square_magnitude() -> None:
    """Central gravity force points at -r and has magnitude mu*m/r^2 (zero torque/momentum)."""
    mass = 2.304
    eff = EarthGravity()
    eff.bind(mass=mass, inertia=np.eye(3))
    r = np.array([7.0e6, 0.0, 0.0])
    state = _state(r, np.array([0.0, 0.0, 0.0, 1.0]))

    force, torque, momentum = eff.calc_contributions(0.0, state, np.zeros(0), np.zeros(0))

    r_norm = float(np.linalg.norm(r))
    assert np.allclose(force, -MU * mass * r / r_norm**3)
    assert np.isclose(np.linalg.norm(force), MU * mass / r_norm**2)
    assert np.allclose(force / np.linalg.norm(force), -r / r_norm)
    assert np.allclose(torque, 0.0)
    assert np.allclose(momentum, 0.0)


def test_earth_gravity_unbound_mass_raises() -> None:
    """Evaluating EarthGravity before its mass is bound is an error."""
    eff = EarthGravity()
    state = _state(np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    with pytest.raises(RuntimeError, match="unbound"):
        eff.calc_contributions(0.0, state, np.zeros(0), np.zeros(0))


def test_earth_gravity_closes_a_circular_orbit() -> None:
    """Integrated under central gravity a circular state returns near its start after one period."""
    body = RigidBodyDynamics(dt=1.0, mass=2.304, inertia=np.eye(3), effectors=[EarthGravity()])
    r0 = np.array([7.0e6, 0.0, 0.0])
    v_circ = np.sqrt(MU / np.linalg.norm(r0))
    body.x[0:3] = r0
    body.x[3:6] = np.array([0.0, v_circ, 0.0])

    period = 2.0 * np.pi * np.sqrt(np.linalg.norm(r0) ** 3 / MU)
    n_steps = round(period / body.dt)
    for k in range(n_steps):
        body.evaluate(k * body.dt, np.zeros(0))

    # Orbit closes to well under 1% of the radius (a straight line would be several radii off).
    assert np.linalg.norm(body.x[0:3] - r0) < 0.01 * np.linalg.norm(r0)
