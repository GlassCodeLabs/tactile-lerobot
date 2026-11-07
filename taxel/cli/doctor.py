"""``taxel doctor``: check a recorded dataset and write ``tactile/health.json``.

::

    taxel doctor <dataset_root_or_repo_id> [--episodes 0,3-5] [--instances left_fingertip]
        [--checks drift,framerate] [--reference observation.images.front | all] [--json path]
        [--fail-on drift=0.05,dropped=0.02,gel=0.05,skew=50,saturated=0.01,integrity]
        [--thresholds file.json] [--no-write] [--guess]

The report goes to ``tactile/health.json`` inside the dataset unless ``--no-write`` is given,
and to ``--json`` when given; a per-instance table is printed either way. The exit code is 0
when no ``--fail-on`` threshold is hit, 1 when one is, 2 on a usage error. Without
``--fail-on`` a flag is an annotation only (SCHEMA.md, principle 6) and the exit code is 0.
``--guess`` lists the feature keys that look like tactile streams by name and exits, for a
dataset whose streams are not under ``observation.tactile``. Reading the dataset needs
``taxel[doctor]``; the image checks need ``taxel[lerobot]``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from taxel.doctor.loader import DoctorDataset, candidate_keys, resolve_root
from taxel.doctor.report import REFERENCE_ALL, run_doctor, write_report
from taxel.schema import (
    CHECK_NAMES,
    HealthReport,
    InstanceChecks,
    SchemaError,
    Skipped,
    Thresholds,
)

__all__ = [
    "FAIL_ON_CHECKS",
    "format_guess",
    "format_report",
    "main",
    "parse_checks",
    "parse_episodes",
    "parse_fail_on",
]

FAIL_ON_CHECKS = {
    "drift": "drift",
    "drift_units": "drift",
    "dropped": "framerate",
    "max_gap_ms": "framerate",
    "gel": "gel",
    "cluster_px": "gel",
    "cluster_pct": "gel",
    "skew": "skew",
    "saturated": "saturation",
    "integrity": "integrity",
}
"""``--fail-on`` names, each a :class:`~taxel.schema.Thresholds` field, and the check whose flag they fail."""


class UsageError(ValueError):
    """A problem with the options, reported without a traceback and with exit code 2."""


def parse_episodes(spec: str) -> list[int]:
    """``"0,3-5"`` to ``[0, 3, 4, 5]``."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        first, dash, last = part.partition("-")
        try:
            lo = int(first)
            hi = int(last) if dash else lo
        except ValueError:
            raise UsageError(f"--episodes: {part!r} is not an index or a range like 3-5") from None
        if lo < 0 or hi < lo:
            raise UsageError(f"--episodes: {part!r} is not an ascending range of non-negative indices")
        out.extend(range(lo, hi + 1))
    if not out:
        raise UsageError("--episodes: no episode given")
    return sorted(set(out))


def parse_checks(spec: str) -> list[str]:
    """``"drift,framerate"`` to the checks to run, in report order."""
    names = {s.strip() for s in spec.split(",") if s.strip()}
    unknown = sorted(names - set(CHECK_NAMES))
    if unknown:
        raise UsageError(f"--checks: unknown {unknown}; the checks are {', '.join(CHECK_NAMES)}")
    if not names:
        raise UsageError("--checks: no check given")
    return [c for c in CHECK_NAMES if c in names]


def format_guess(features: dict) -> str:
    """The ``--guess`` listing: candidate tactile feature keys and the ``--instances`` that names them."""
    found = candidate_keys(features)
    if not found:
        return (
            "no feature key contains tactile, touch, wrench or force; "
            "name the stream with --instances <feature key>"
        )
    lines = ["candidate tactile features, by name:"]
    for key in found:
        entry = features[key]
        lines.append(f"  {key}  {entry.get('dtype')}  {list(entry.get('shape') or ())}")
    lines.append("run again with --instances " + ",".join(found))
    return "\n".join(lines)


def parse_fail_on(spec: str, thresholds: Thresholds) -> tuple[Thresholds, set[str]]:
    """Apply ``name=value`` overrides to ``thresholds`` and return them with the checks that fail the run.

    Each entry is a threshold name, optionally with a value: ``drift=0.05`` sets the drift
    threshold and makes a drift flag fail the run; a bare ``integrity`` only makes that flag fail
    the run.
    """
    values = thresholds.to_dict()
    checks: set[str] = set()
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, eq, value = entry.partition("=")
        if name not in FAIL_ON_CHECKS:
            raise UsageError(f"--fail-on: unknown check {name!r}; known: {', '.join(FAIL_ON_CHECKS)}")
        if eq:
            try:
                values[name] = int(value) if name == "cluster_px" else float(value)
            except ValueError:
                raise UsageError(f"--fail-on: {name}={value!r} is not a number") from None
            if name == "integrity":
                raise UsageError("--fail-on: integrity takes no value")
        checks.add(FAIL_ON_CHECKS[name])
    try:
        return Thresholds.from_dict(values), checks
    except SchemaError as e:
        raise UsageError(f"--fail-on: {e}") from None


def read_thresholds(path: str | Path) -> Thresholds:
    """A thresholds JSON file: an object with any of the :class:`~taxel.schema.Thresholds` keys."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise UsageError(f"--thresholds: cannot read {path}: {e}") from None
    if not isinstance(data, dict):
        raise UsageError(f"--thresholds: {path} must hold a JSON object")
    try:
        return Thresholds.from_dict({**Thresholds().to_dict(), **data})
    except SchemaError as e:
        raise UsageError(f"--thresholds: {e}") from None


def failed_checks(report: HealthReport, checks: set[str]) -> list[tuple[str, str, str]]:
    """``(episode, instance, check)`` for every flagged block of a check named in ``checks``."""
    hits = []
    for episode, block in report.episodes.items():
        for instance, result in block.items():
            for check in sorted(checks):
                value = getattr(result, check)
                if not isinstance(value, Skipped) and value.flag:
                    hits.append((episode, instance, check))
    return hits


# --- Table output ------------------------------------------------------------------------------------

_COLUMNS = ("episode", "integrity", "drift", "framerate", "saturation", "gel", "skew")
_NUMBER = re.compile(r" -?\d+(\.\d+)?(?= below)")


def format_report(report: HealthReport) -> str:
    """The per-instance table: one block per instance, one row per episode, a cell per check."""
    lines: list[str] = []
    for name, instance in report.instances.items():
        limits = "" if instance.range is None else f"  range: [{instance.range.min}, {instance.range.max}]"
        rate = "" if instance.declared_rate_hz is None else f"  {instance.declared_rate_hz:g} Hz"
        lines.append(
            f"{name}  {instance.key}  {instance.family}  metadata: {instance.metadata}  "
            f"sidecar: {'yes' if instance.sidecar else 'no'}{rate}{limits}"
        )
        if instance.unknown_keys or instance.missing_keys or instance.error:
            notes = []
            if instance.unknown_keys:
                notes.append("unknown keys " + ", ".join(instance.unknown_keys))
            if instance.missing_keys:
                notes.append("missing keys " + ", ".join(instance.missing_keys))
            if instance.error:
                notes.append(instance.error)
            lines.append("  " + "; ".join(notes))
        rows = [list(_COLUMNS)]
        skipped: dict[str, dict[str, int]] = {}
        for episode, block in report.episodes.items():
            checks = block.get(name)
            if checks is None:
                continue
            rows.append(
                [episode.removeprefix("episode_"), *(_cell(checks, c, skipped) for c in _COLUMNS[1:])]
            )
        widths = [max(len(r[i]) for r in rows) for i in range(len(_COLUMNS))]
        lines.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows)
        for check, reasons in skipped.items():
            for reason, count in reasons.items():
                lines.append(f"  skipped {check}: {reason} ({count} episode{'s' if count != 1 else ''})")
        lines.append("")
    flagged = sum(
        1
        for block in report.episodes.values()
        for result in block.values()
        for check in _COLUMNS[1:]
        if not isinstance(getattr(result, check), Skipped) and getattr(result, check).flag
    )
    lines.append(
        f"{len(report.episodes)} episode(s), {len(report.instances)} instance(s), {flagged} flag(s); "
        "! marks a flag, - a skipped check"
    )
    return "\n".join(lines)


def _cell(checks: InstanceChecks, check: str, skipped: dict[str, dict[str, int]]) -> str:
    value = getattr(checks, check)
    if isinstance(value, Skipped):
        reasons = skipped.setdefault(check, {})
        reason = _NUMBER.sub("", value.skipped)  # one line per reason, whatever the measured value was
        reasons[reason] = reasons.get(reason, 0) + 1
        return "-"
    mark = " !" if value.flag else ""
    if check == "integrity":
        parts = [p for p in ("copy", "raw", "files") if not isinstance(getattr(value, p), Skipped)]
        failed = [p for p in parts if getattr(value, p).flag]
        text = ("FAIL " + ",".join(failed)) if failed else ("ok " + ",".join(parts) if parts else "skipped")
        return text
    if check == "drift":
        events = f", {len(value.rebaselines)} rebaseline" if value.rebaselines else ""
        zero = ", first frame zero" if value.first_frame_zero else ""
        if value.confidence == "low":
            zero += ", low confidence"
        if isinstance(value.offset, Skipped):
            change = max(value.level_change, key=abs)
            return f"level change {change:+.3g}{events}{zero}{mark}"
        offset = max(value.offset, key=abs)
        pct = "" if value.offset_pct_range is None else f" ({max(value.offset_pct_range):.1%})"
        return f"{offset:+.3g}{pct}{events}{zero}{mark}"
    if check == "framerate":
        if value.dropped_pct is not None:
            return f"{value.dropped_pct:.1%} dropped, gap {value.max_gap_ms:.0f} ms{mark}"
        note = "nominal" if value.timestamps == "nominal" else f"{value.effective_hz:.2f} Hz"
        return f"{note}, gap {value.max_gap_ms:.0f} ms{mark}"
    if check == "saturation":
        if value.limit == "observed" and value.at_min_pct is not None and value.at_max_pct is not None:
            return f"min {max(value.at_min_pct):.1%}, max {max(value.at_max_pct):.1%} (observed)"
        return f"{max(value.saturated_pct):.2%} ({value.limit}){mark}"
    if check == "gel":
        if value.kind == "pixels":
            return f"{value.dead_pct:.2%} dead, cluster {value.largest_cluster_px} px{mark}"
        constant = "" if value.constant_elements is None else f", {len(value.constant_elements)} constant"
        return f"{len(value.dead_elements or [])} dead{constant} element(s){mark}"
    return f"{value.lag_ms:+.0f} ms (conf {value.confidence:.2f}){mark}"


# --- Entry point --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taxel doctor",
        description=(
            "Per-episode checks of a recorded dataset: integrity, drift, framerate, saturation, gel and skew."
        ),
    )
    parser.add_argument("dataset", help="dataset root directory, or a Hub repo id to download")
    parser.add_argument("--episodes", help="episodes to check, like 0,3-5 (default all)")
    parser.add_argument(
        "--instances", help="instance ids or feature keys to check, comma-separated (default all)"
    )
    parser.add_argument(
        "--checks", help=f"checks to run, comma-separated from {','.join(CHECK_NAMES)} (default all)"
    )
    parser.add_argument(
        "--reference",
        help=(
            "feature key the skew check correlates against (default: the first video feature that is not "
            f"a checked instance), or '{REFERENCE_ALL}' to decode every video stream and take the one "
            "with the most motion"
        ),
    )
    parser.add_argument(
        "--guess",
        action="store_true",
        help="list the feature keys that look like tactile streams by name, then exit",
    )
    parser.add_argument("--json", dest="json_path", help="also write the report to this path")
    parser.add_argument(
        "--fail-on",
        dest="fail_on",
        help="checks whose flag fails the run, like drift=0.05,drift_units=0.2,integrity",
    )
    parser.add_argument("--thresholds", help="JSON file of threshold values")
    parser.add_argument(
        "--no-write", dest="no_write", action="store_true", help="do not write tactile/health.json"
    )
    parser.add_argument(
        "--window-s", dest="window_s", type=float, default=0.5, help="drift window in seconds"
    )
    return parser


def run(argv: Sequence[str]) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as e:
        return int(e.code or 0) if e.code in (0, None) else 2
    thresholds = read_thresholds(args.thresholds) if args.thresholds else Thresholds()
    checks: set[str] = set()
    if args.fail_on:
        thresholds, checks = parse_fail_on(args.fail_on, thresholds)
    episodes = parse_episodes(args.episodes) if args.episodes else None
    instances = [s.strip() for s in args.instances.split(",") if s.strip()] if args.instances else ()
    selected = parse_checks(args.checks) if args.checks else None
    if selected is not None and checks - set(selected):
        left_out = ", ".join(sorted(checks - set(selected)))
        raise UsageError(f"--fail-on names {left_out}, which --checks leaves out")
    if args.window_s <= 0:
        raise UsageError("--window-s must be positive")
    try:
        root = resolve_root(args.dataset)
        dataset = DoctorDataset(root)
        if args.guess:
            print(format_guess(dataset.features))
            return 0
        report = run_doctor(
            dataset,
            episodes=episodes,
            instances=instances,
            reference=args.reference,
            checks=selected,
            thresholds=thresholds,
            window_s=args.window_s,
        )
    except (FileNotFoundError, KeyError, SchemaError) as e:
        raise UsageError(str(e).strip("'\"")) from None
    print(format_report(report))
    if not args.no_write:
        path = write_report(report, root)
        print(f"written {path}")
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        print(f"written {args.json_path}")
    hits = failed_checks(report, checks)
    if hits:
        for episode, instance, check in hits:
            print(f"fail: {episode} {instance} {check}")
        return 1
    return 0


def main(argv: Sequence[str]) -> int:
    """Run ``taxel doctor`` with the options after the subcommand; return the exit code."""
    try:
        return run(argv)
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"taxel doctor: {e}", file=sys.stderr)
        return 2
