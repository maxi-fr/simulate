"""Tests for how the loggers fix each signal's dtype, shared by both backends.

A buffer's dtype is decided once -- from the first logged value, or from the field's
``dtype`` metadata -- and numpy assignment into it truncates silently afterwards. These
tests pin the declaration channel that lets a signal state the width it needs, and the
guard that rejects the writes which would otherwise be clipped without a word.
"""

import dataclasses
import functools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from simulate.logger import BaseLogger, MmapLogger, RamLogger

LoggerFactory = Callable[[int], BaseLogger]


@pytest.fixture(params=["ram", "mmap"])
def make_logger(request: pytest.FixtureRequest, tmp_path: Path) -> LoggerFactory:
    """Build a logger of either backend, so shared behaviour is asserted against both."""
    if request.param == "ram":
        return RamLogger
    return functools.partial(MmapLogger, directory=tmp_path, prefix="test")


@dataclasses.dataclass(frozen=True)
class SolverLog:
    """A log whose status is wider than its first value -- the case that motivated the guard."""

    status: str = dataclasses.field(metadata={"dtype": "<U32"})
    n_iter: int = 0


@dataclasses.dataclass(frozen=True)
class UndeclaredLog:
    """The same signal without a declared width, so the first value fixes it."""

    status: str


def test_declared_dtype_fits_statuses_longer_than_the_first(make_logger: LoggerFactory) -> None:
    """Test that field metadata sizes the buffer, so a later, longer status survives intact."""
    logger = make_logger(2)

    logger.log(0.0, {"mpc": SolverLog(status="Solve_Succeeded", n_iter=3)})
    logger.log(1.0, {"mpc": SolverLog(status="Maximum_Iterations_Exceeded", n_iter=100)})

    _, status = logger.signal("mpc", "status")
    assert status.dtype == np.dtype("<U32")
    assert list(status) == ["Solve_Succeeded", "Maximum_Iterations_Exceeded"]


def test_declared_dtype_survives_the_npz_roundtrip(make_logger: LoggerFactory, tmp_path: Path) -> None:
    """Test that a declared-width string signal reaches the archive uncorrupted on both backends."""
    logger = make_logger(2)

    logger.log(0.0, {"mpc": SolverLog(status="Solve_Succeeded")})
    logger.log(1.0, {"mpc": SolverLog(status="Maximum_Iterations_Exceeded")})
    logger.finalize(tmp_path, prefix="test")

    data = np.load(tmp_path / "test.npz")
    assert list(data["mpc.status"]) == ["Solve_Succeeded", "Maximum_Iterations_Exceeded"]


def test_string_longer_than_the_buffer_raises(make_logger: LoggerFactory) -> None:
    """Test that an undeclared status is clipped to the first value's width -- and so raises."""
    logger = make_logger(2)
    logger.log(0.0, {"mpc": UndeclaredLog(status="Solve_Succeeded")})  # fixes the buffer at <U15

    with pytest.raises(ValueError, match=r"'mpc.status' is 15 characters wide but got a 27-character"):
        logger.log(1.0, {"mpc": UndeclaredLog(status="Maximum_Iterations_Exceeded")})


def test_string_longer_than_the_declared_width_raises(make_logger: LoggerFactory) -> None:
    """Test that a declared width too short for a later status raises rather than clipping."""
    logger = make_logger(2)
    logger.log(0.0, {"mpc": SolverLog(status="Solve_Succeeded")})

    with pytest.raises(ValueError, match=r"is 32 characters wide but got a 33-character"):
        logger.log(1.0, {"mpc": SolverLog(status="x" * 33)})


@dataclasses.dataclass(frozen=True)
class ScalarLog:
    value: float | int | bool


@pytest.mark.parametrize(
    ("first", "later"),
    [
        (0, 1.5),  # int buffer would floor the float
        (True, 2.7),  # bool buffer would collapse it to True
        (True, 3),  # bool buffer would collapse it to True
    ],
    ids=["float-into-int", "float-into-bool", "int-into-bool"],
)
def test_lossy_numeric_write_raises(make_logger: LoggerFactory, first: float, later: float) -> None:
    """Test that a value whose kind the buffer cannot hold raises instead of being truncated."""
    logger = make_logger(2)
    logger.log(0.0, {"comp": ScalarLog(value=first)})

    with pytest.raises(ValueError, match=r"Signal 'comp.value' has dtype"):
        logger.log(1.0, {"comp": ScalarLog(value=later)})


def test_narrowing_within_a_kind_is_allowed(make_logger: LoggerFactory) -> None:
    """Test that a deliberately narrowed numeric buffer still accepts wider values of its own kind."""

    @dataclasses.dataclass(frozen=True)
    class NarrowLog:
        value: float = dataclasses.field(metadata={"dtype": "float32"})

    logger = make_logger(1)
    logger.log(0.0, {"comp": NarrowLog(value=1.5)})  # a float64 into a declared float32

    _, val = logger.signal("comp", "value")
    assert val.dtype == np.dtype("float32")
    assert val[0] == 1.5


def test_string_into_a_numeric_buffer_raises(make_logger: LoggerFactory) -> None:
    """Test that a value of an unrelated kind is refused by the guard, not left to numpy."""
    logger = make_logger(2)
    logger.log(0.0, {"comp": ScalarLog(value=1.0)})

    with pytest.raises(ValueError, match=r"Signal 'comp.value' has dtype float64 but got a 'U'-kind"):
        logger.log(1.0, {"comp": ScalarLog(value="oops")})  # ty:ignore[invalid-argument-type]


@dataclasses.dataclass(frozen=True)
class CounterLog:
    """A declared-unsigned counter, the one buffer kind a signed value silently wraps in."""

    count: int = dataclasses.field(metadata={"dtype": "uint32"})


def test_signed_array_into_an_unsigned_buffer_raises(make_logger: LoggerFactory) -> None:
    """Test that a numpy signed integer is kept out of an unsigned buffer, where -1 becomes 2**32-1."""
    logger = make_logger(1)

    with pytest.raises(ValueError, match=r"Signal 'comp.count' has dtype uint32 but got a 'i'-kind"):
        logger.log(0.0, {"comp": CounterLog(count=np.int64(-1))})  # ty:ignore[invalid-argument-type]


def test_python_int_into_an_unsigned_buffer_is_allowed(make_logger: LoggerFactory) -> None:
    """Test that a declared-unsigned signal stays writable from plain Python ints.

    numpy range-checks Python integers itself (``OverflowError`` on a negative), so the guard
    must not reject them wholesale -- that would make the declaration unusable.
    """
    logger = make_logger(1)
    logger.log(0.0, {"comp": CounterLog(count=5)})

    _, count = logger.signal("comp", "count")
    assert count[0] == 5


def test_array_signal_checks_its_longest_element(make_logger: LoggerFactory) -> None:
    """Test that the width check covers every element of an array-valued string signal."""

    @dataclasses.dataclass(frozen=True)
    class ArrayLog:
        names: np.ndarray

    logger = make_logger(2)
    logger.log(0.0, {"comp": ArrayLog(names=np.array(["ok", "ok"]))})  # fixes the buffer at <U2

    with pytest.raises(ValueError, match=r"'comp.names' is 2 characters wide but got a 7-character"):
        logger.log(1.0, {"comp": ArrayLog(names=np.array(["ok", "failure"]))})
