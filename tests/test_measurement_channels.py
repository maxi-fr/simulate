import numpy as np

from simulate.component import NoLog
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
    """Each sensor channel logs its own y_mea/truth/noise under an indexed name."""
    sim = _two_channel_sim(t_end=0.05, sensor1_dt=0.01)
    sim.run()

    assert sim.logger is not None
    n = len(sim.logger.t)
    for name in ("sensor_0", "sensor_1"):
        assert len(sim.logger.signal(name, "truth")) == n
        assert len(sim.logger.signal(name, "y_mea")) == n


def test_estimator_receives_concatenated_measurement() -> None:
    """IdentityEstimator passes the (2,) concatenated measurement vector through as x_hat and returns NoLog."""
    estimator = IdentityEstimator(dt=0.01)
    x_hat, log = estimator.evaluate(0.0, np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert np.asarray(x_hat).shape == (2,)
    assert isinstance(log, NoLog)


def test_slow_sensor_is_zoh_held() -> None:
    """A sensor at 2x the base dt holds its sample between updates (ZOH), unlike the base sensor."""
    sim = _two_channel_sim(t_end=0.06, sensor1_dt=0.02)
    sim.run()

    assert sim.logger is not None
    fast = sim.logger.signal("sensor_0", "y_mea")[:, 0]
    slow = sim.logger.signal("sensor_1", "y_mea")[:, 0]

    # The fast channel updates every base step once the truth starts moving.
    assert fast[2] != fast[1]
    # The slow channel (dt = 2 * base) updates at steps 0, 2, 4 and holds in between,
    # so it repeats its sample in consecutive pairs.
    assert slow[1] == slow[0]
    assert slow[3] == slow[2]
    assert slow[2] != slow[1]
