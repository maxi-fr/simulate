import numpy as np

from simulate.controller import PIController
from simulate.dynamics import LinearDynamics
from simulate.estimator import IdentityEstimator
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor, LinearMeasurement
from simulate.simulation import Simulation


def _two_channel_sim(t_end: float, sensor1_dt: float) -> Simulation:
    """A 2-state plant measured by two channels: one at base dt, one at sensor1_dt."""
    base = 0.01
    dynamics = LinearDynamics(dt=base, A=[[1.0, 0.0], [0.0, 1.0]], B=[[1.0, 0.0], [0.0, 1.0]])
    # Each sensor owns a measurement model selecting one state component.
    sen0 = GaussianSensor(dt=base, measurement=LinearMeasurement(C=[[1.0, 0.0]], D=[[0.0, 0.0]]), std_dev=0.0)
    sen1 = GaussianSensor(dt=sensor1_dt, measurement=LinearMeasurement(C=[[0.0, 1.0]], D=[[0.0, 0.0]]), std_dev=0.0)
    reference = StepReference(dt=base, step_value=np.array([1.0, 2.0]))
    estimator = IdentityEstimator(dt=base)
    controller = PIController(dt=base, kp=[[0.5, 0.0], [0.0, 0.5]], ki=[[0.0, 0.0], [0.0, 0.0]])
    return Simulation(
        t_end=t_end,
        dynamics=dynamics,
        reference=reference,
        sensors=[sen0, sen1],
        estimator=estimator,
        controller=controller,
    )


def test_two_channels_log_per_channel() -> None:
    """Each sensor channel logs its own truth/noise under an indexed name; y_mea is merged."""
    sim = _two_channel_sim(t_end=0.05, sensor1_dt=0.01)
    sim.run()

    logs = sim.logger.component_logs
    n = len(sim.logger.core_logs)
    for name in ("sensor_0", "sensor_1"):
        assert len(logs[name]) == n
        assert "truth" in logs[name][0]

    # Truth is per-channel only; the core log carries the merged measurement.
    assert "y_mea" in sim.logger.core_logs[0]
    assert "y" not in sim.logger.core_logs[0]


def test_estimator_receives_concatenated_measurement() -> None:
    """The loop reassembles the two channels into one (2,) vector for the estimator."""
    sim = _two_channel_sim(t_end=0.03, sensor1_dt=0.01)
    sim.run()
    # IdentityEstimator passes the (2,) concatenated measurement through as x_hat.
    x_hat = sim.logger.core_logs[-1]["x_hat"]
    assert np.asarray(x_hat).shape == (2,)


def test_slow_sensor_is_zoh_held() -> None:
    """A sensor at 2x the base dt holds its sample between updates (ZOH), unlike the base sensor."""
    sim = _two_channel_sim(t_end=0.06, sensor1_dt=0.02)
    sim.run()

    fast = [e["y_mea"][0] for e in sim.logger.core_logs]
    slow = [e["y_mea"][1] for e in sim.logger.core_logs]

    # The fast channel updates every base step once the truth starts moving.
    assert fast[2] != fast[1]
    # The slow channel (dt = 2 * base) updates at steps 0, 2, 4 and holds in between,
    # so it repeats its sample in consecutive pairs.
    assert slow[1] == slow[0]
    assert slow[3] == slow[2]
    assert slow[2] != slow[1]
