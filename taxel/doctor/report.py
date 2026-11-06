"""Assembling a :class:`~taxel.schema.HealthReport`: run the checks over a dataset and write the JSON.

:func:`run_doctor` walks the instances and episodes of a :class:`~taxel.doctor.loader.DoctorDataset`,
feeds the pure check functions and collects their blocks; a check whose input is missing
becomes :class:`~taxel.schema.Skipped` with the loader's reason. :func:`write_report`
puts the result at ``tactile/health.json``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from taxel import __version__
from taxel.dataset.sidecar import episode_path, raw_frames, read_episode
from taxel.doctor.checks import (
    BaselineInputs,
    drift_check,
    framerate_main_table,
    framerate_sidecar,
    integrity_copy,
    integrity_files,
    integrity_raw,
    saturation_check,
)
from taxel.doctor.loader import DoctorDataset, EpisodeInputs, InstanceSource
from taxel.schema import (
    HealthReport,
    InstanceChecks,
    IntegrityReport,
    SessionRecord,
    Skipped,
    Thresholds,
    health_report_path,
)

__all__ = ["check_episode", "run_doctor", "write_report"]

NOT_IMPLEMENTED = "check not implemented in this version"


def run_doctor(
    dataset: DoctorDataset,
    *,
    episodes: Iterable[int] | None = None,
    instances: Iterable[str] = (),
    thresholds: Thresholds | None = None,
    window_s: float = 0.5,
    generated_at: str | None = None,
) -> HealthReport:
    """Run every check on the selected episodes of the selected instances and return the report.

    ``episodes`` defaults to all; ``instances`` to every tactile-looking feature (see
    :func:`~taxel.doctor.loader.tactile_keys`). ``window_s`` is the drift check's window.
    """
    thresholds = thresholds or Thresholds()
    sources = dataset.instances(instances)
    wanted = list(episodes) if episodes is not None else dataset.episode_indices
    unknown = sorted(set(wanted) - set(dataset.episode_indices))
    if unknown:
        raise KeyError(f"no episodes {unknown}; the dataset has {dataset.episode_indices}")
    totals = _SessionTotals(dataset)
    report_episodes: dict[str, dict[str, InstanceChecks]] = {}
    for e in wanted:
        block: dict[str, InstanceChecks] = {}
        for name, source in sources.items():
            inputs = dataset.episode_inputs(source, e)
            block[name] = check_episode(inputs, thresholds, totals, window_s)
        report_episodes[f"episode_{e:06d}"] = block
    stamp = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return HealthReport(
        doctor_version=__version__,
        generated_at=stamp,
        thresholds=thresholds,
        instances={name: source.report() for name, source in sources.items()},
        episodes=report_episodes,
    )


def write_report(report: HealthReport, root: str | Path) -> Path:
    """Write the report to ``<root>/tactile/health.json`` and return the path."""
    path = health_report_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return path


def check_episode(
    inputs: EpisodeInputs, thresholds: Thresholds, totals: _SessionTotals, window_s: float = 0.5
) -> InstanceChecks:
    """The checks of one instance in one episode, from the loader's inputs."""
    source = inputs.instance
    if source.kind != "array":
        return InstanceChecks(
            integrity=Skipped("image instance: no native stream to verify"),
            drift=Skipped("drift is defined for array instances"),
            framerate=framerate_main_table(inputs.timestamps, inputs.fps, thresholds),
            saturation=Skipped("saturation is defined for array instances"),
            gel=Skipped(NOT_IMPLEMENTED),
            skew=Skipped(NOT_IMPLEMENTED),
        )
    missing = inputs.missing[0] if inputs.missing else None
    session = inputs.session
    native, rows = inputs.native, inputs.baseline_rows
    frames = inputs.frames
    assert frames is not None
    rate = source.declared_rate_hz or inputs.fps
    limits = source.range.limits(source.n_channels) if source.range is not None else None

    # Integrity, first.
    if native is None and session is None:
        integrity: IntegrityReport | Skipped = Skipped(missing or "no sidecar")
    else:
        if native is None:
            copy = Skipped(_reason(inputs, "no native stream"))
        elif inputs.origin is None:
            copy = Skipped(_reason(inputs, "no clock"))
        else:
            copy = integrity_copy(frames, native, inputs.origin, inputs.fps, source.reduction)
        if native is None:
            raw = Skipped(_reason(inputs, "no native stream"))
        elif rows is None:
            raw = Skipped(_reason(inputs, "no baseline stream"))
        else:
            raw = integrity_raw(native, rows)
        if session is None:
            files = Skipped(_reason(inputs, "no session record"))
        else:
            session_rows, session_gaps = totals.get(source, session)
            files = integrity_files(
                native is not None, rows, session, inputs.episode_index, session_rows, session_gaps
            )
        parts = [p for p in (copy, raw, files) if not isinstance(p, Skipped)]
        integrity = IntegrityReport(copy, raw, files, flag=any(p.flag for p in parts))

    # Framerate and saturation from the native stream when there is one, else from the main table.
    if native is not None:
        counters = session.frames if session is not None else None
        framerate = framerate_sidecar(native.host_ts, native.seq, rate, counters, thresholds)
        raw = raw_frames(native, rows) if rows is not None and rows.captures.any() else native.data
        saturation = saturation_check(raw, limits, "sidecar", thresholds)
    else:
        framerate = framerate_main_table(inputs.timestamps, inputs.fps, thresholds)
        saturation = saturation_check(frames, limits, "main_table", thresholds)

    # Drift from the baseline stream and the native stream, else from the main table.
    if native is not None:
        drift = drift_check(
            native.data,
            rate,
            _baseline_inputs(inputs),
            limits,
            "sidecar",
            inputs.origin,
            thresholds,
            window_s,
        )
    else:
        drift = drift_check(
            frames,
            inputs.fps,
            BaselineInputs(),
            None if limits is None else limits,
            "main_table",
            None,
            thresholds,
            window_s,
        )
    return InstanceChecks(
        integrity=integrity,
        drift=drift,
        framerate=framerate,
        saturation=saturation,
        gel=Skipped(NOT_IMPLEMENTED),
        skew=Skipped(NOT_IMPLEMENTED),
    )


def _reason(inputs: EpisodeInputs, prefix: str) -> str:
    for reason in inputs.missing:
        if reason.startswith(prefix):
            return reason
    return prefix


def _baseline_inputs(inputs: EpisodeInputs) -> BaselineInputs:
    """What the drift check needs about the baseline, from the session record and the baseline rows."""
    session, rows, source = inputs.session, inputs.baseline_rows, inputs.instance
    captures = samples = None
    capture_host_ts = None
    if rows is not None:
        if rows.captures.any():
            captures = (
                rows.sample_index[rows.captures],
                rows.host_ts[rows.captures],
                rows.values[rows.captures],
            )
            capture_host_ts = float(captures[1][0])
        if rows.samples.any():
            samples = (rows.sample_index[rows.samples], rows.values[rows.samples])
    session_values = None
    frame_range: list[int] | str | None = None
    if session is not None and session.baseline is not None and session.baseline.values is not None:
        session_values = np.asarray(session.baseline.values, dtype=np.float32)
        if session.baseline.frame_range is not None:
            frame_range = list(session.baseline.frame_range)
        else:
            which = session_index(source.sessions, session) if source.sessions is not None else None
            missing = f"meta.sessions[{which}].baseline.frame_range" in source.missing_keys
            frame_range = "not recorded" if missing else "no seq"
        if capture_host_ts is None and session.clock is not None:
            capture_host_ts = _host_time(session, session.baseline.captured_at)
    return BaselineInputs(captures, session_values, samples, frame_range, capture_host_ts)


def session_index(meta, session: SessionRecord) -> int | None:
    for i, record in enumerate(meta.sessions):
        if record is session:
            return i
    return None


def _host_time(session: SessionRecord, wall_time: str) -> float | None:
    """A wall-clock ISO time on the session's host clock, through the ``clock`` anchor."""
    clock = session.clock
    if clock is None:
        return None
    try:
        anchor = datetime.fromisoformat(clock.wall_time.replace("Z", "+00:00"))
        then = datetime.fromisoformat(wall_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    return clock.host_clock + (then - anchor).total_seconds()


class _SessionTotals:
    """Native rows and ``seq`` gaps summed over a session's episode files, computed once per session."""

    def __init__(self, dataset: DoctorDataset) -> None:
        self._dataset = dataset
        self._cache: dict[tuple[str, str], tuple[int | None, int | None]] = {}

    def get(self, source: InstanceSource, session: SessionRecord) -> tuple[int | None, int | None]:
        key = (source.instance_id, session.session_id)
        if key not in self._cache:
            self._cache[key] = self._compute(source, session)
        return self._cache[key]

    def _compute(self, source: InstanceSource, session: SessionRecord) -> tuple[int | None, int | None]:
        rows = 0
        gaps: int | None = 0
        first, last = session.episodes
        for e in range(first, last + 1):
            path = episode_path(self._dataset.root, source.instance_id, e)
            if not path.is_file():
                return None, None
            stream = read_episode(path)
            rows += len(stream)
            if stream.seq is None or len(stream) < 2:
                gaps = None if stream.seq is None else gaps
            elif gaps is not None:
                gaps += int(stream.seq[-1] - stream.seq[0] + 1 - len(stream))
        return rows, gaps
