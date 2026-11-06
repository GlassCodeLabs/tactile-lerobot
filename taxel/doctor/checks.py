"""The integrity check and the array-stream health checks, as pure functions on numpy arrays.

Each function takes arrays and metadata and returns a report block from
:mod:`taxel.schema`, flagged against a :class:`~taxel.schema.Thresholds`. Nothing here
reads a file; :mod:`taxel.doctor.report` feeds these from the loader.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from taxel.dataset.reduce import reduce_episode
from taxel.dataset.sidecar import BaselineRows, EpisodeStream, frame_times, raw_frames
from taxel.doctor.windows import cross_correlate, largest_cluster, no_contact, window_edges, window_stats
from taxel.schema import (
    CopyCheck,
    DriftReport,
    FilesCheck,
    FrameCounts,
    FramerateReport,
    GapHistogram,
    GelReport,
    RawCheck,
    RebaselineEvent,
    Reduction,
    SaturationReport,
    SessionRecord,
    SkewReport,
    Skipped,
    Thresholds,
)

__all__ = [
    "BaselineInputs",
    "GAP_EDGES_PERIODS",
    "drift_check",
    "framerate_main_table",
    "framerate_sidecar",
    "gel_check_elements",
    "gel_check_pixels",
    "integrity_copy",
    "integrity_files",
    "integrity_raw",
    "max_gap_threshold_ms",
    "saturation_check",
    "skew_check",
]

DEAD_FRACTION = 0.2
"""A pixel or element is dead when its variance over the episode is below this fraction of the median.

A fifth, not a hundredth: a constant region in a video flickers after encoding (LeRobot's default
AV1 at CRF 30 leaves it a few tenths of a count of variance), and real gel images vary by tens.
"""

GAP_EDGES_PERIODS = (1.5, 2.5, 3.5)
"""Bin edges of the gap histogram, in declared periods; the last bin is open."""

_NOMINAL_ATOL = 1e-5


# --- Integrity -------------------------------------------------------------------------------------


def integrity_copy(
    frames: NDArray[np.floating],
    native: EpisodeStream,
    origin: float,
    fps: float,
    reduction: Reduction | None,
) -> CopyCheck:
    """``copy``: the main-table frames equal the reduction recomputed from the native stream.

    ``frames`` is the main-table copy ``(n, *shape)``; the reduction is ``reduction``'s
    method and window, or ``last`` over one dataset period when none is declared.
    Equality is exact in float32.
    """
    method, window_ms = (
        ("last", 1000.0 / fps) if reduction is None else (reduction.method, reduction.window_ms)
    )
    targets = frame_times(origin, len(frames), fps)
    try:
        expected = reduce_episode(method, native.host_ts, native.data, targets, window_ms)
    except ValueError:  # no native sample at or before the first frame: nothing to compare
        expected = None
    stored = np.asarray(frames, dtype=np.float32).reshape(len(frames), *native.shape)
    if expected is None:
        return CopyCheck(
            frames=len(frames), mismatches=len(frames), first_mismatch=0, max_abs_diff=0.0, flag=True
        )
    differs = np.any(stored != expected, axis=tuple(range(1, stored.ndim)))
    mismatches = int(differs.sum())
    first = int(np.flatnonzero(differs)[0]) if mismatches else None
    max_diff = float(np.abs(stored.astype(np.float64) - expected).max()) if len(stored) else 0.0
    return CopyCheck(len(frames), mismatches, first, max_diff, flag=mismatches > 0)


def integrity_raw(native: EpisodeStream, rows: BaselineRows) -> RawCheck:
    """``raw``: every sample row equals the reconstruction from stored data plus the captures, bit for bit."""
    raw = raw_frames(native, rows)
    sampled = rows.sample_index[rows.samples]
    values = rows.values[rows.samples]
    if sampled.size == 0:
        return RawCheck(samples=0, mismatches=0, first_mismatch=None, max_abs_diff=0.0, flag=False)
    reconstructed = raw[sampled]
    differs = np.any(values != reconstructed, axis=tuple(range(1, values.ndim)))
    mismatches = int(differs.sum())
    first = int(sampled[np.flatnonzero(differs)[0]]) if mismatches else None
    max_diff = float(np.abs(values.astype(np.float64) - reconstructed).max())
    return RawCheck(int(sampled.size), mismatches, first, max_diff, flag=mismatches > 0)


def integrity_files(
    native_present: bool,
    rows: BaselineRows | None,
    session: SessionRecord,
    episode_index: int,
    session_rows: int | None,
    session_gaps: int | None,
) -> FilesCheck:
    """``files``: the episode's files exist and the session record is consistent with them.

    ``session_rows`` and ``session_gaps`` are the native rows and the ``seq`` gaps summed
    over the session's episode files, or ``None`` when a file is missing.
    """
    applied = session.baseline is not None and session.baseline.applied
    if rows is None:
        capture_at_zero: bool | None = None if not applied else False
    elif applied:
        capture_at_zero = bool(rows.captures.size and rows.captures[0] and rows.sample_index[0] == 0)
    else:
        capture_at_zero = None if not rows.captures.any() else False
    first, last = session.episodes
    origins = session.clock is not None and len(session.clock.episode_origins) == last - first + 1
    if session.frames is None or session_rows is None:
        counters: bool | None = None
    else:
        counters = session.frames.received >= session_rows and (
            session_gaps is None or session.frames.dropped >= session_gaps
        )
    flag = (
        not (native_present and rows is not None and origins) or capture_at_zero is False or counters is False
    )
    return FilesCheck(native_present, rows is not None, capture_at_zero, bool(origins), counters, flag)


# --- Framerate -------------------------------------------------------------------------------------


def max_gap_threshold_ms(thresholds: Thresholds, rate_hz: float) -> float:
    """The longest tolerated gap: ``thresholds.max_gap_ms``, or ten periods of ``rate_hz``."""
    return thresholds.max_gap_ms if thresholds.max_gap_ms is not None else 10_000.0 / rate_hz


def _gaps(times: NDArray[np.float64], rate_hz: float) -> tuple[float, GapHistogram, float]:
    """Effective rate, the gap histogram in periods and the longest gap in ms of a time series."""
    n = len(times)
    if n < 2:
        return 0.0, GapHistogram(list(GAP_EDGES_PERIODS), [0] * (len(GAP_EDGES_PERIODS) + 1)), 0.0
    gaps = np.diff(times)
    effective = (n - 1) / float(times[-1] - times[0]) if times[-1] > times[0] else 0.0
    periods = gaps * rate_hz
    counts = np.histogram(periods, bins=[-np.inf, *GAP_EDGES_PERIODS, np.inf])[0]
    histogram = GapHistogram(list(GAP_EDGES_PERIODS), [int(c) for c in counts])
    return effective, histogram, float(gaps.max() * 1000.0)


def framerate_sidecar(
    host_ts: NDArray[np.float64],
    seq: NDArray[np.int64] | None,
    declared_hz: float,
    counters: FrameCounts | None,
    thresholds: Thresholds,
) -> FramerateReport:
    """Framerate from the native stream: effective rate, gaps in ``host_ts``, drops from ``seq`` gaps."""
    times = np.asarray(host_ts, dtype=np.float64)
    effective, histogram, max_gap = _gaps(times, declared_hz)
    dropped_pct = None
    if seq is not None and len(seq) > 1:
        span = int(seq[-1] - seq[0] + 1)
        dropped_pct = float(span - len(seq)) / span
    limit = max_gap_threshold_ms(thresholds, declared_hz)
    flag = max_gap > limit or (dropped_pct is not None and dropped_pct > thresholds.dropped)
    return FramerateReport(
        effective_hz=effective,
        declared_hz=float(declared_hz),
        dropped_pct=dropped_pct,
        max_gap_ms=max_gap,
        max_gap_threshold_ms=limit,
        gap_histogram=histogram,
        frames=counters,
        source="sidecar",
        timestamps="measured",
        flag=bool(flag),
    )


def framerate_main_table(
    timestamps: NDArray[np.float64], fps: float, thresholds: Thresholds
) -> FramerateReport:
    """Framerate from the main table's ``timestamp`` column: regularity against the dataset fps.

    LeRobot writes ``frame_index / fps`` there, in which case the column is ``nominal`` and
    says nothing about the sensor; the check reports that rather than a result.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    nominal = bool(np.allclose(times, np.arange(len(times)) / fps, atol=_NOMINAL_ATOL))
    effective, histogram, max_gap = _gaps(times, fps)
    limit = max_gap_threshold_ms(thresholds, fps)
    return FramerateReport(
        effective_hz=effective,
        declared_hz=float(fps),
        dropped_pct=None,
        max_gap_ms=max_gap,
        max_gap_threshold_ms=limit,
        gap_histogram=histogram,
        frames=None,
        source="main_table",
        timestamps="nominal" if nominal else "measured",
        flag=bool(max_gap > limit),
    )


# --- Saturation ------------------------------------------------------------------------------------


def saturation_check(
    data: NDArray[np.floating],
    limits: tuple[NDArray[np.floating], NDArray[np.floating]] | None,
    source: str,
    thresholds: Thresholds,
) -> SaturationReport:
    """Per channel, the fraction of samples at or beyond the limit.

    ``data`` is ``(n, E, C)``, the raw readings (reconstructed when a baseline was
    subtracted, since a limit applies to what the sensor read). With ``limits``
    (``(min, max)`` per channel from ``range``) the result is ``declared``; without, the
    observed extremes stand in, with ``confidence`` ``low``.
    """
    values = np.asarray(data, dtype=np.float64).reshape(len(data), -1, data.shape[-1])
    if limits is not None:
        lo, hi = (np.asarray(v, dtype=np.float64) for v in limits)
        limit, confidence = "declared", "declared"
        # Raw frames reconstructed from subtracted data carry float32 rounding: a few ulps of slack.
        slack = 4 * np.finfo(np.float32).eps * np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
    else:
        lo, hi = values.min(axis=(0, 1)), values.max(axis=(0, 1))
        limit, confidence = "observed", "low"
        slack = np.zeros_like(lo)
    at_limit = (values <= lo + slack) | (values >= hi - slack)
    pct = at_limit.mean(axis=(0, 1)) if values.size else np.zeros(values.shape[-1])
    return SaturationReport(
        saturated_pct=[float(p) for p in pct],
        limit=limit,
        min=[float(v) for v in lo],
        max=[float(v) for v in hi],
        confidence=confidence,
        source=source,
        flag=bool(np.any(pct > thresholds.saturated)),
    )


# --- Drift -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineInputs:
    """What the drift check knows about the baseline, from the baseline stream and the session record.

    ``captures`` are the capture rows (``sample_index``, ``host_ts``, ``values``) when a
    baseline is applied; ``session_values`` the session record's ``baseline.values`` when
    one was captured, applied or not; ``samples`` the sample rows (``sample_index``,
    ``values``). ``frame_range`` is the session baseline's, as the report states it, and
    ``capture_host_ts`` the host time of the baseline in force at the episode's start.
    """

    captures: tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float32]] | None = None
    session_values: NDArray[np.float32] | None = None
    samples: tuple[NDArray[np.int64], NDArray[np.float32]] | None = None
    frame_range: list[int] | str | None = None
    capture_host_ts: float | None = None


def drift_check(
    stored: NDArray[np.floating],
    rate_hz: float,
    baseline: BaselineInputs,
    limits: tuple[NDArray[np.floating], NDArray[np.floating]] | None,
    source: str,
    origin: float | None,
    thresholds: Thresholds,
    window_s: float = 0.5,
) -> DriftReport | Skipped:
    """The drift check on one episode's stream ``stored`` (``(n, E, C)``, as recorded).

    Windows of ``window_s`` are cut from the stream, restarting at each capture row. The
    raw level of a window is the mean of its sample rows when it has any (``source``
    ``samples``), else the stored level plus the baseline in force (``raw_frames``); on a
    plain dataset the stored frames are the raw level (``main_table``). The offset against
    the baseline in force is fitted per segment with :func:`~taxel.doctor.windows.no_contact`;
    the report's ``offset`` is the last segment's line at the last no-contact window (never
    extrapolated past the data), per channel the element of largest magnitude, signed, and
    ``slope_per_s`` that line's slope, per channel the median over elements.
    """
    data = np.asarray(stored, dtype=np.float64)
    n, n_elements, n_channels = data.shape
    window = max(2, int(round(window_s * rate_hz)))
    captures = baseline.captures
    breaks = [int(i) for i in captures[0] if i > 0] if captures is not None else []
    starts, ends = window_edges(n, window, breaks)
    if len(starts) < 2:
        return Skipped(f"episode shorter than two drift windows of {window_s} s")
    stats = window_stats(data, starts, ends)
    t = stats.centres(rate_hz)

    # The baseline in force per window, and the raw level per window.
    in_force: NDArray[np.float64] | None = None
    kind = "none"
    if captures is not None and captures[0].size:
        order = np.searchsorted(captures[0], starts, side="right") - 1
        in_force = captures[2][np.clip(order, 0, None)].astype(np.float64)
        in_force[order < 0] = 0.0  # stored raw before the first capture
        kind = "capture"
    elif baseline.session_values is not None:
        in_force = np.broadcast_to(baseline.session_values.astype(np.float64), stats.level.shape).copy()
        kind = "session"
    subtracted = in_force if kind == "capture" else np.zeros_like(stats.level)
    raw_level = stats.level + subtracted
    used_samples = 0
    if baseline.samples is not None and baseline.samples[0].size:
        index, values = baseline.samples
        for w, (s, e) in enumerate(zip(starts, ends, strict=True)):
            inside = (index >= s) & (index < e)
            if inside.any():
                raw_level[w] = values[inside].astype(np.float64).mean(axis=0)
                t[w] = float(index[inside].mean()) / rate_hz  # the level is the samples', so is its time
                used_samples += 1
    if source == "main_table":
        level_source = "main_table"
    else:
        level_source = "samples" if used_samples == len(starts) else "raw_frames"

    # No-contact windows and the drift line, per segment between captures.
    y = raw_level - in_force if in_force is not None else raw_level
    segment_starts = [0, *breaks]
    mask = np.zeros(len(starts), bool)
    slope = np.zeros((n_elements, n_channels))
    intercept = np.zeros((n_elements, n_channels))
    candidates = 0
    for a, b in zip(segment_starts, [*segment_starts[1:], n], strict=True):
        part = (starts >= a) & (starts < b)
        if not part.any():
            continue
        fit = no_contact(y[part], t[part], stats.quiet[part], stats.noise)
        mask[part], slope, intercept, candidates = fit.mask, fit.slope, fit.intercept, fit.candidates
    no_contact_idx = np.flatnonzero(mask)
    if no_contact_idx.size == 0:
        return Skipped("no no-contact window in the episode: contact throughout")
    first, last = no_contact_idx[0], no_contact_idx[-1]
    level_first, level_last = raw_level[first], raw_level[last]
    no_contact_level = [float(v) for v in level_first.mean(axis=0)]
    level_change = _extreme_per_channel(level_last - level_first)
    slope_per_s = [float(v) for v in np.median(slope, axis=0)]  # the typical element, not the noisiest

    if in_force is None or np.any(np.isnan(intercept)):
        offset: list[float] | Skipped = Skipped("no session baseline")
        pct = None
        measure = np.abs(level_last - level_first)
    else:
        end = (
            slope * t[last] + intercept
        )  # the line at the last no-contact window: measured, not extrapolated
        offset = _extreme_per_channel(end)
        measure = np.abs(end)
        pct = None
    if limits is not None:
        lo, hi = (np.asarray(v, dtype=np.float64) for v in limits)
        span = hi - lo
        scaled = measure / span
        if not isinstance(offset, Skipped):
            pct = [float(v) for v in scaled.max(axis=0)]
        flag = bool(np.any(scaled.max(axis=0) > thresholds.drift))
    else:
        flag = bool(np.any(measure.max(axis=0) > thresholds.drift))

    events: list[RebaselineEvent] = []
    if captures is not None:
        index, host_ts, values = captures
        previous = baseline.session_values if baseline.session_values is not None else values[0]
        for k in range(len(index)):
            if index[k] > 0:
                delta = np.abs(values[k].astype(np.float64) - np.asarray(previous, dtype=np.float64))
                events.append(
                    RebaselineEvent(int(index[k]), float(host_ts[k]), [float(v) for v in delta.max(axis=0)])
                )
            previous = values[k]
    age = None
    if origin is not None and baseline.capture_host_ts is not None:
        age = float(origin - baseline.capture_host_ts)
    return DriftReport(
        no_contact_level=no_contact_level,
        level_change=level_change,
        offset=offset,
        offset_pct_range=pct,
        slope_per_s=slope_per_s,
        baseline=kind,
        source=level_source,
        baseline_age_s=age,
        frame_range=baseline.frame_range,
        windows=int(len(starts)),
        no_contact_windows=int(no_contact_idx.size),
        candidates=candidates,
        first_frame_zero=bool(np.all(data[0] == 0.0)),
        rebaselines=events,
        flag=flag,
    )


def _extreme_per_channel(values: NDArray[np.float64]) -> list[float]:
    """Per channel, the element with the largest magnitude, signed."""
    picked = np.take_along_axis(values, np.abs(values).argmax(axis=0)[None, :], axis=0)[0]
    return [float(v) for v in picked]


# --- Gel -------------------------------------------------------------------------------------------------


def gel_check_pixels(frames: NDArray[np.uint8], thresholds: Thresholds) -> GelReport | Skipped:
    """Dead pixels of an image stream: pixels whose variance over the frames is near zero while the rest
    varies.

    ``frames`` is ``(k, H, W, 3)`` uint8, a sample of the episode's frames. A pixel is dead when
    its grayscale variance is below :data:`DEAD_FRACTION` of the median pixel variance, a rule
    that holds through video encoding; the
    largest 4-connected cluster of dead pixels is reported too. Flagged on the dead fraction or
    the cluster size.
    """
    k = len(frames)
    if k < 2:
        return Skipped(f"{k} frame(s) decoded: variance needs at least two")
    variance = np.asarray(frames, dtype=np.float32).mean(axis=-1).var(axis=0)
    median = float(np.median(variance))
    if median <= 1e-9:
        return Skipped("frames do not vary: a dead pixel cannot be told from a static scene")
    dead = variance <= DEAD_FRACTION * median
    cluster = largest_cluster(dead)
    dead_pct = float(dead.mean())
    flag = dead_pct > thresholds.gel or cluster > thresholds.cluster_px
    return GelReport("pixels", dead_pct, cluster, None, k, flag)


def gel_check_elements(data: NDArray[np.floating], thresholds: Thresholds) -> GelReport | Skipped:
    """Dead elements of an array stream ``(n, E, C)``: elements whose variance over the episode is near
    zero while the others vary, listed by index. Flagged on the dead fraction."""
    n = len(data)
    if n < 2:
        return Skipped(f"{n} frame(s): variance needs at least two")
    variance = np.asarray(data, dtype=np.float64).var(axis=0).mean(axis=1)
    median = float(np.median(variance))
    if median <= 1e-12:
        return Skipped("stream does not vary: a dead element cannot be told from a constant reading")
    dead = np.flatnonzero(variance <= DEAD_FRACTION * median)
    dead_pct = float(dead.size / variance.size)
    return GelReport("elements", dead_pct, None, [int(i) for i in dead], n, dead_pct > thresholds.gel)


# --- Skew ------------------------------------------------------------------------------------------------


def skew_check(
    tactile_times: NDArray[np.floating],
    tactile_events: NDArray[np.floating],
    reference_times: NDArray[np.floating],
    reference_events: NDArray[np.floating],
    reference_key: str,
    thresholds: Thresholds,
    max_lag_s: float = 0.5,
    grid_s: float = 0.005,
) -> SkewReport | Skipped:
    """Lag between contact onsets in a tactile stream and the same onsets in a reference.

    Both inputs are event signals on one clock: large at an onset or a release, near zero
    otherwise. For an array stream that is the absolute first difference of
    :func:`~taxel.doctor.windows.contact_signal`; for an image stream it is
    :func:`~taxel.doctor.windows.motion_energy`, which is a difference already. They are
    cross-correlated on a ``grid_s`` grid over ``[-max_lag_s, max_lag_s]``, smoothed to the
    coarser series' spacing; ``lag_ms`` is positive when the tactile stream lags the reference.
    Skipped when either signal has no onset, or when the cross-correlation peak is below
    ``thresholds.confidence``. ``resolution_ms`` is the coarser of the grid and the two series'
    spacing.
    """
    events_a = np.asarray(tactile_events, dtype=np.float64)
    events_b = np.asarray(reference_events, dtype=np.float64)
    times_a = np.asarray(tactile_times, dtype=np.float64)
    times_b = np.asarray(reference_times, dtype=np.float64)
    if len(events_a) < 2 or len(events_b) < 2:
        return Skipped("too few frames to find an onset")
    if not np.any(events_a > np.median(events_a)) or not np.any(events_b > np.median(events_b)):
        return Skipped("no contact onset in the tactile stream or in the reference")
    spacing = max(float(np.median(np.diff(times_a))), float(np.median(np.diff(times_b))))
    lag_s, confidence, onsets = cross_correlate(
        times_a, events_a, times_b, events_b, max_lag_s, grid_s, smooth_s=spacing
    )
    if confidence < thresholds.confidence:
        return Skipped(
            f"no onset matched in both signals: confidence {confidence:.2f} below {thresholds.confidence}"
        )
    resolution_ms = max(grid_s, spacing) * 1000.0
    lag_ms = lag_s * 1000.0
    return SkewReport(
        lag_ms, confidence, reference_key, onsets, resolution_ms, bool(abs(lag_ms) > thresholds.skew)
    )
