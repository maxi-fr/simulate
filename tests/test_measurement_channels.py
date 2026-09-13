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
    for name in ("sensor_0", "sensor_1"):
        t_sen, truth = sim.logger.signal(name, "truth")
        _, y_mea = sim.logger.signal(name, "y_mea")
        assert len(t_sen) == 6
        assert len(truth) == 6
        assert len(y_mea) == 6


def test_estimator_receives_concatenated_measurement() -> None:
    """IdentityEstimator passes the (2,) concatenated measurement vector through as x_hat and returns NoLog."""
    estimator = IdentityEstimator(dt=0.01)
    x_hat, log = estimator.evaluate(0.0, np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert np.asarray(x_hat).shape == (2,)
    assert isinstance(log, NoLog)


def test_slow_sensor_logs_at_own_rate() -> None:
    """A sensor at 2x base dt logs only at its own update times, not every base step."""
    sim = _two_channel_sim(t_end=0.06, sensor1_dt=0.02)
    sim.run()

    assert sim.logger is not None
    t_fast, fast = sim.logger.signal("sensor_0", "y_mea")
    t_slow, slow = sim.logger.signal("sensor_1", "y_mea")

    assert len(t_fast) == 7
    assert len(t_slow) == 4
    assert np.allclose(t_slow, [0.0, 0.02, 0.04, 0.06])
    assert fast[:, 0][2] != fast[:, 0][1]
    assert slow[:, 0][1] != slow[:, 0][0]
