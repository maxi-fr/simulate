"""Compare simulation speed and memory of the working tree vs. a base commit.

This is the user-facing entry point. It benchmarks the current working-tree
engine source against the source at ``--base`` (HEAD by default), prints an
old-vs-new table of wall-clock time and peak memory with a percentage delta per
workload, and **exits non-zero** if any headline metric regresses past
``--threshold`` (default 10%).

The base source is provided by a throwaway ``git worktree`` of ``--base`` (the
dirty working tree is never touched). The *same* benchmark script and configs
(from the working tree) drive both runs -- only the engine ``src`` differs --
and each run is a clean ``uv run`` subprocess so imports never cross over.

Timing is inherently noisy: results are the median of several repeats, but for
reliable flagging run on an otherwise idle machine and raise ``--threshold`` /
``--repeats`` if you see spurious failures.

Example
-------
``uv run python benchmarks/compare.py``
``uv run python benchmarks/compare.py --workloads dc_motor --threshold 0.15``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from workloads import WORKLOADS

BYTES_PER_MIB = 1024 * 1024

# (json key, short label) for the metrics that gate the exit code. time_total
# includes the export/merge phase and peak_total_bytes is the max over run and
# export, so a logging regression surfaces in these headline numbers.
GATING_METRICS = (("time_total", "time"), ("peak_total_bytes", "mem"))
# Informational breakdown rows shown indented under each gating row.
DETAIL_METRICS = {
    "time_total": ("time_run", "time_export"),
    "peak_total_bytes": ("peak_run_bytes", "peak_export_bytes"),
}


def _repo_root() -> Path:
    """Return the git repository root.

    Returns
    -------
    Path
        Absolute path to the working-tree top level.
    """
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def _run_benchmark(repo: Path, src: Path, out: Path, workloads: list[str], repeats: int | None) -> dict:
    """Run benchmark.py in a clean subprocess and load its JSON output.

    Returns
    -------
    dict
        The parsed result payload (``metadata`` and ``workloads``).
    """
    cmd = [
        "uv",
        "run",
        "benchmarks/benchmark.py",
        "--src",
        str(src),
        "--out",
        str(out),
        "--workloads",
        *workloads,
    ]
    if repeats is not None:
        cmd += ["--repeats", str(repeats)]
    subprocess.run(cmd, check=True, cwd=repo)
    return json.loads(out.read_text())


def _is_bytes_metric(key: str) -> bool:
    """Return whether ``key`` is a byte-valued (memory) metric."""
    return key.endswith("_bytes")


def _fmt_value(key: str, value: float) -> str:
    """Format a metric value for display (MiB for memory, s/ms for time).

    Returns
    -------
    str
        Human-readable, right-padded-friendly representation.
    """
    if _is_bytes_metric(key):
        return f"{value / BYTES_PER_MIB:.2f} MiB"
    return f"{value * 1e3:.1f} ms" if value < 1.0 else f"{value:.3f} s"


def _delta_fraction(old: float, new: float) -> float:
    """Return the signed fractional change ``(new - old) / old``.

    Returns
    -------
    float
        Fractional delta; ``0.0`` if both are ~zero, ``inf`` if only old is zero.
    """
    if old <= 0.0:
        return 0.0 if new <= 0.0 else float("inf")
    return (new - old) / old


def _fmt_delta(fraction: float) -> str:
    """Format a fractional delta as a signed percentage.

    Returns
    -------
    str
        e.g. ``+3.4%``, ``-1.0%`` or ``+inf``.
    """
    if fraction == float("inf"):
        return "+inf"
    return f"{fraction * 100:+.1f}%"


def _label(key: str) -> str:
    """Return the display label for a metric key (strips the ``_bytes`` suffix)."""
    return key.removesuffix("_bytes")


def _print_row(label: str, old: float, new: float, key: str, status: str = "") -> None:
    """Print a single formatted comparison row."""
    fraction = _delta_fraction(old, new)
    print(
        f"  {label:<14}{_fmt_value(key, old):>12}{_fmt_value(key, new):>14}{_fmt_delta(fraction):>10}   {status}",
    )


def _compare(old: dict, new: dict, threshold: float) -> list[str]:
    """Print the comparison table and return the list of regressions.

    Returns
    -------
    list of str
        One message per gating metric that regressed beyond ``threshold``.
    """
    regressions: list[str] = []
    old_wl = old["workloads"]
    new_wl = new["workloads"]

    for name in old_wl:
        if name not in new_wl:
            continue
        print(f"\n[{name}]")
        print(f"  {'metric':<14}{'old':>12}{'new':>14}{'delta':>10}   status")
        for key, _short in GATING_METRICS:
            old_v = old_wl[name][key]
            new_v = new_wl[name][key]
            fraction = _delta_fraction(old_v, new_v)
            failed = fraction > threshold
            if failed:
                regressions.append(f"{name} {_label(key)} {_fmt_delta(fraction)}")
            _print_row(_label(key), old_v, new_v, key, status="FAIL" if failed else "ok")
            for detail in DETAIL_METRICS[key]:
                _print_row(f"  {_label(detail)}", old_wl[name][detail], new_wl[name][detail], detail)

    return regressions


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed ``base``, ``threshold``, ``repeats``, ``workloads`` and ``json_out``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="HEAD", help="Git ref to compare against (default: HEAD).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Fractional regression that triggers a failure (default: 0.10 = 10%%).",
    )
    parser.add_argument("--repeats", type=int, default=None, help="Override per-workload timed repetitions.")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=sorted(WORKLOADS),
        default=sorted(WORKLOADS),
        help="Subset of workloads to run (default: all).",
    )
    parser.add_argument("--json-out", default=None, help="Optional path to dump the raw old/new result payloads.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Benchmark working tree vs. ``--base`` and flag regressions.

    Returns
    -------
    int
        ``0`` if no headline metric regressed beyond the threshold, else ``1``.
    """
    args = _parse_args(argv)
    repo = _repo_root()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        worktree = tmp_path / "base"
        subprocess.run(["git", "worktree", "add", "--detach", str(worktree), args.base], check=True, cwd=repo)
        try:
            print(f"== benchmarking base ({args.base}) ==", file=sys.stderr)
            old = _run_benchmark(repo, worktree / "src", tmp_path / "old.json", args.workloads, args.repeats)
            print("== benchmarking working tree ==", file=sys.stderr)
            new = _run_benchmark(repo, repo / "src", tmp_path / "new.json", args.workloads, args.repeats)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], check=False, cwd=repo)

        if args.json_out is not None:
            Path(args.json_out).write_text(json.dumps({"old": old, "new": new}, indent=2))

        print("\n" + "=" * 60)
        print(f"Benchmark comparison: {args.base} (old) vs working tree (new)")
        print(f"threshold: {args.threshold * 100:.1f}%")
        print("=" * 60)
        regressions = _compare(old, new, args.threshold)

    print("\n" + "=" * 60)
    if regressions:
        print("RESULT: REGRESSION DETECTED")
        for msg in regressions:
            print(f"  - {msg}")
        return 1
    print(f"RESULT: OK -- no regressions beyond {args.threshold * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
