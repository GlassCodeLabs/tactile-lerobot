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
from numpy.typing import NDArray

from taxel import __version__
from taxel.dataset.sidecar import episode_path, raw_frames, read_episode
from taxel.doctor.checks import (
    BaselineInputs,
    PixelStats,
    drift_check,
    framerate_main_table,
    framerate_sidecar,
    gel_check_elements,
    gel_check_pixels,
    gel_dead_across,
    integrity_copy,
    integrity_files,
    integrity_raw,
    pixel_stats,
    saturation_check,
    skew_check,
)
from taxel.doctor.loader import DoctorDataset, EpisodeInputs, InstanceSource, sample_indices
from taxel.doctor.windows import contact_signal
from taxel.schema import (
    CHECK_NAMES,
    GelReport,
    HealthReport,
    InstanceChecks,
    IntegrityReport,
    SessionRecord,
    SkewReport,
    Skipped,
    Thresholds,
    health_report_path,
)

__all__ = ["GEL_FRAMES", "REFERENCE_ALL", "check_episode", "run_doctor", "write_report"]

GEL_FRAMES = 60
"""Frames of an image instance sampled across an episode for the gel check."""

REFERENCE_ALL = "all"
"""The ``reference`` that searches every video stream for the one with the most motion."""

NOT_SELECTED = Skipped("not selected")
"""The block of a check the caller left out of ``checks``."""


def run_doctor(
    dataset: DoctorDataset,
    *,
    episodes: Iterable[int] | None = None,
    instances: Iterable[str] = (),
    reference: str | None = None,
    checks: Iterable[str] | None = None,
    thresholds: Thresholds | None = None,
    window_s: float = 0.5,
    generated_at: str | None = None,
) -> HealthReport:
    """Run the checks on the selected episodes of the selected instances and return the report.

    ``episodes`` defaults to all; ``instances`` to every tactile-looking feature (see
    :func:`~taxel.doctor.loader.tactile_keys`); ``checks`` to every check, and a check left
    out is reported ``skipped: not selected`` and costs nothing. ``reference`` names the
    feature the skew check correlates against: by default the first video or image feature
    that is not a checked instance, else another image instance, else another array
    instance; :data:`REFERENCE_ALL` decodes every video stream and takes the one with the
    most motion. ``window_s`` is the drift check's window.

    Raises:
        KeyError: If no instance is found, or an episode, a check or the reference is unknown.
    """
    thresholds = thresholds or Thresholds()
    sources = dataset.instances(instances)
    if not sources:
        raise KeyError(
            "no tactile instance: no feature key is observation.tactile or observation.tactile.<id>; "
            "name the stream with --instances, or run with --guess to list candidates"
        )
    wanted = list(episodes) if episodes is not None else dataset.episode_indices
    unknown = sorted(set(wanted) - set(dataset.episode_indices))
    if unknown:
        raise KeyError(f"no episodes {unknown}; the dataset has {dataset.episode_indices}")
    if reference not in (None, REFERENCE_ALL) and reference not in dataset.features:
        raise KeyError(f"reference {reference!r} is not a feature of the dataset")
    selected = set(CHECK_NAMES) if checks is None else set(checks)
    unknown_checks = sorted(selected - set(CHECK_NAMES))
    if unknown_checks:
        raise KeyError(f"unknown checks {unknown_checks}; the checks are {', '.join(CHECK_NAMES)}")
    totals = _SessionTotals(dataset)
    report_episodes: dict[str, dict[str, InstanceChecks]] = {}
    stats: dict[tuple[str, int], PixelStats] = {}
    for e in wanted:
        signals = _Signals(dataset, e, sources, gel_frames=GEL_FRAMES if "gel" in selected else None)
        block: dict[str, InstanceChecks] = {}
        for name, source in sources.items():
            inputs = dataset.episode_inputs(source, e)
            block[name] = check_episode(inputs, thresholds, totals, window_s, signals, reference, selected)
            if source.key in signals.pixel_stats:
                stats[(name, e)] = signals.pixel_stats[source.key]
        report_episodes[f"episode_{e:06d}"] = block
    _settle_gel(report_episodes, sources, wanted, stats, thresholds)
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
    inputs: EpisodeInputs,
    thresholds: Thresholds,
    totals: _SessionTotals,
    window_s: float = 0.5,
    signals: _Signals | None = None,
    reference: str | None = None,
    selected: Iterable[str] | None = None,
) -> InstanceChecks:
    """The checks of one instance in one episode, from the loader's inputs.

    ``signals`` holds the episode's decoded frames, motion-energy and contact signals, shared
    by the instances; without it the gel check on images and the skew check are skipped.
    ``selected`` names the checks to run; the others are ``skipped: not selected``.
    """
    source = inputs.instance
    on = set(CHECK_NAMES) if selected is None else set(selected)
    if signals is None:
        gel: GelReportOrSkipped = Skipped("no signals: gel on images needs the episode's frames")
        skew: SkewReportOrSkipped = Skipped("no signals: skew needs a reference")
    else:
        gel = _gel(inputs, signals, thresholds) if "gel" in on else NOT_SELECTED
        skew = _skew(inputs, signals, reference, thresholds) if "skew" in on else NOT_SELECTED
    if source.kind != "array":
        framerate = framerate_main_table(inputs.timestamps, inputs.fps, thresholds)
        return InstanceChecks(
            integrity=Skipped("image instance: no native stream to verify"),
            drift=Skipped("drift is defined for array instances"),
            framerate=framerate if "framerate" in on else NOT_SELECTED,
            saturation=Skipped("saturation is defined for array instances"),
            gel=gel,
            skew=skew,
        )
    missing = inputs.missing[0] if inputs.missing else None
    session = inputs.session
    native, rows = inputs.native, inputs.baseline_rows
    frames = inputs.frames
    assert frames is not None
    rate = source.declared_rate_hz or inputs.fps
    limits = source.range.limits(source.n_channels) if source.range is not None else None

    # Integrity, first.
    if "integrity" not in on:
        integrity: IntegrityReport | Skipped = NOT_SELECTED
    elif native is None and session is None:
        integrity = Skipped(missing or "no sidecar")
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
    framerate = saturation = NOT_SELECTED
    if native is not None:
        counters = session.frames if session is not None else None
        if "framerate" in on:
            framerate = framerate_sidecar(native.host_ts, native.seq, rate, counters, thresholds)
        if "saturation" in on:
            raw = raw_frames(native, rows) if rows is not None and rows.captures.any() else native.data
            saturation = saturation_check(raw, limits, "sidecar", thresholds)
    else:
        if "framerate" in on:
            framerate = framerate_main_table(inputs.timestamps, inputs.fps, thresholds)
        if "saturation" in on:
            saturation = saturation_check(frames, limits, "main_table", thresholds)

    # Drift from the baseline stream and the native stream, else from the main table.
    if "drift" not in on:
        drift = NOT_SELECTED
    elif native is not None:
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
        gel=gel,
        skew=skew,
    )


GelReportOrSkipped = GelReport | Skipped
SkewReportOrSkipped = SkewReport | Skipped


def _gel(inputs: EpisodeInputs, signals: _Signals, thresholds: Thresholds) -> GelReportOrSkipped:
    """The gel check of one episode; :func:`_settle_gel` revisits it once every episode is in."""
    source = inputs.instance
    if source.kind == "array":
        data = inputs.native.data if inputs.native is not None else inputs.frames
        assert data is not None
        return gel_check_elements(data, thresholds)
    try:
        frames = signals.gel_sample(source.key)
    except ImportError as e:
        return Skipped(str(e))
    stats = pixel_stats(frames)
    signals.pixel_stats[source.key] = stats
    return gel_check_pixels(stats, thresholds)


def _settle_gel(
    episodes: dict[str, dict[str, InstanceChecks]],
    sources: dict[str, InstanceSource],
    wanted: list[int],
    stats: dict[tuple[str, int], PixelStats],
    thresholds: Thresholds,
) -> None:
    """The gel check's cross-episode step, per instance and per session where a sidecar defines one.

    An image instance is judged on its quietest episode's median pixel variance, and every episode
    gets the same verdict; an array instance's dead elements are those constant in every episode
    whose check ran. Without a sidecar the session is the run's episode selection.
    """
    for name, source in sources.items():
        groups: dict[str | None, list[int]] = {}
        for e in wanted:
            session = source.session_for(e)
            groups.setdefault(None if session is None else session.session_id, []).append(e)
        for members in groups.values():
            if source.kind == "image":
                present = [(e, stats[(name, e)]) for e in members if (name, e) in stats]
                usable = [s.median_variance for _, s in present if s.frames_used >= 2]
                if not usable:
                    continue
                rest = min(usable)
                for e, s in present:
                    block = gel_check_pixels(s, thresholds, rest, len(usable))
                    episodes[f"episode_{e:06d}"][name].gel = block
            else:
                ran = [e for e in members if isinstance(episodes[f"episode_{e:06d}"][name].gel, GelReport)]
                blocks = [episodes[f"episode_{e:06d}"][name].gel for e in ran]
                for e, block in zip(ran, gel_dead_across(blocks, source.shape[0], thresholds), strict=True):
                    episodes[f"episode_{e:06d}"][name].gel = block


def _skew(
    inputs: EpisodeInputs, signals: _Signals, reference: str | None, thresholds: Thresholds
) -> SkewReportOrSkipped:
    """The skew check of one instance against the reference, both placed on one clock.

    An instance with a native stream is on the host clock, and so is a video reference when
    the session's clock gives the episode's origin; anything read from the main table is on
    LeRobot's nominal timeline, and then both sides use it.
    """
    source = inputs.instance
    try:
        key = (
            reference
            if reference not in (None, REFERENCE_ALL)
            else signals.default_reference(source.key, everything=reference == REFERENCE_ALL)
        )
    except ImportError as e:
        return Skipped(str(e))
    if key is None:
        return Skipped("no reference: the dataset has no camera and no other tactile instance")
    if key == source.key:
        return Skipped(f"reference {key!r} is the instance itself")
    nominal = inputs.native is None or inputs.origin is None
    try:
        tactile_times, tactile = signals.signal(source.key, inputs, nominal)
        ref_times, ref = signals.signal(key, None, nominal)
    except ImportError as e:
        return Skipped(str(e))
    if inputs.native is not None and inputs.origin is None:
        return Skipped(
            "no clock in the session record: the reference cannot be placed on the sidecar's timeline"
        )
    return skew_check(tactile_times, tactile, ref_times, ref, key, thresholds)


class _Signals:
    """The episode's per-feature signals, decoded once and shared by the instances.

    Each video or image feature is decoded in one sequential pass (:meth:`DoctorDataset.iter_frames`)
    that yields the motion energy for the skew check and, for an image instance when the gel check
    runs, the ``gel_frames`` sampled RGB frames the gel check reads.
    """

    def __init__(
        self,
        dataset: DoctorDataset,
        episode_index: int,
        sources: dict[str, InstanceSource],
        gel_frames: int | None = GEL_FRAMES,
    ) -> None:
        self.dataset = dataset
        self.episode_index = episode_index
        self.sources = {s.key: s for s in sources.values()}
        self.gel_frames = gel_frames
        self.pixel_stats: dict[str, PixelStats] = {}
        """Per image instance key, the statistics its gel check read this episode; see :func:`_settle_gel`."""
        self._motion: dict[str, NDArray[np.float64]] = {}
        self._samples: dict[str, NDArray[np.uint8]] = {}
        self._timestamps = dataset.timestamps(episode_index)

    @property
    def image_keys(self) -> list[str]:
        return [k for k, ft in self.dataset.features.items() if ft.get("dtype") in ("video", "image")]

    def _decode(self, key: str) -> None:
        """One pass over the feature's frames: motion energy, and the gel sample for an image instance."""
        n = self.dataset.episode(self.episode_index).length
        want_sample = self.gel_frames is not None and key in self.sources
        sampled = sample_indices(n, self.gel_frames) if want_sample else np.zeros(0, dtype=np.int64)
        energy = np.zeros(n)
        samples: list[NDArray[np.uint8]] = []
        previous: NDArray[np.float32] | None = None
        for k, gray, rgb in self.dataset.iter_frames(key, self.episode_index, rgb_at=sampled):
            if previous is not None:
                # Motion energy at frame k measures the change between frames k - 1 and k.
                energy[k] = float(np.abs(gray - previous).mean())
            previous = gray
            if rgb is not None:
                samples.append(rgb)
        self._motion[key] = energy
        if want_sample:
            self._samples[key] = np.stack(samples) if samples else np.zeros((0, 0, 0, 3), np.uint8)

    def motion(self, key: str) -> NDArray[np.float64]:
        """Motion energy per frame of an image feature; element 0 is zero."""
        if key not in self._motion:
            self._decode(key)
        return self._motion[key]

    def gel_sample(self, key: str) -> NDArray[np.uint8]:
        """The ``gel_frames`` frames of an image instance spread over the episode, uint8 ``(k, H, W, 3)``."""
        if key not in self._samples:
            self._decode(key)
        return self._samples[key]

    def default_reference(self, own_key: str, everything: bool = False) -> str | None:
        """The skew reference for an instance.

        The first video or image feature that is not a checked instance, a scene camera, so
        one stream is decoded for every instance; else the first other image instance; else
        another array instance. With ``everything`` the image feature with the most motion
        in the episode, which decodes them all.
        """
        others = [k for k in self.image_keys if k != own_key]
        if everything:
            best, best_energy = None, -1.0
            for key in others:
                energy = float(self.motion(key).sum())
                if energy > best_energy:
                    best, best_energy = key, energy
            if best is not None:
                return best
        else:
            cameras = [k for k in others if k not in self.sources]
            if cameras:
                return cameras[0]
            if others:
                return others[0]
        for key, source in self.sources.items():
            if key != own_key and source.kind == "array":
                return key
        return None

    def signal(
        self, key: str, inputs: EpisodeInputs | None, nominal: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """A feature's event signal and its times: motion energy for an image stream, the absolute first
        difference of the contact magnitude for an array stream.

        With ``nominal`` the times are the main table's; otherwise an array instance's native
        ``host_ts`` and a video's ``origin + k / fps``. A difference is placed midway between
        the two samples it compares: half a period before frame ``k`` for a video.
        """
        feature = self.dataset.features[key]
        if feature.get("dtype") in ("video", "image"):
            energy = self.motion(key)
            times = self._timestamps if nominal else self._video_times(key, len(energy))
            # Motion energy at frame k measures the change between frames k - 1 and k: it sits between them.
            return times - 0.5 / self.dataset.fps, energy
        source = self.sources.get(key)
        if inputs is None or inputs.instance.key != key:
            inputs = self.dataset.episode_inputs(source, self.episode_index) if source is not None else None
        if inputs is None:
            raise KeyError(f"{key!r} is not a tactile instance of this run")
        if not nominal and inputs.native is not None:
            times, data = inputs.native.host_ts, inputs.native.data
        else:
            assert inputs.frames is not None
            times, data = self._timestamps, inputs.frames
        return (times[1:] + times[:-1]) / 2.0, np.abs(
            np.diff(contact_signal(data))
        )  # between the two samples

    def _video_times(self, key: str, n: int) -> NDArray[np.float64]:
        origin = None
        for source in self.sources.values():
            if source.kind == "array" and source.sidecar:
                session = source.session_for(self.episode_index)
                if session is not None and session.clock is not None:
                    origin = float(session.clock.episode_origins[self.episode_index - session.episodes[0]])
                    break
        if origin is None:
            return self._timestamps
        return origin + np.arange(n) / self.dataset.fps


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
