"""The integrity check and the array-stream health checks, as pure functions on numpy arrays.

Each function takes arrays and metadata and returns a report block from
:mod:`taxel.schema`, flagged against a :class:`~taxel.schema.Thresholds`. Nothing here
reads a file; :mod:`taxel.doctor.report` feeds these from the loader.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

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
    "DEAD_VARIANCE",
    "ENCODER_FLOOR",
    "GAP_EDGES_PERIODS",
    "PixelStats",
    "drift_check",
    "framerate_main_table",
    "framerate_sidecar",
    "gel_check_elements",
    "gel_check_pixels",
    "gel_dead_across",
    "integrity_copy",
    "integrity_files",
    "integrity_raw",
    "max_gap_threshold_ms",
    "pixel_stats",
    "saturation_check",
    "skew_check",
]

DEAD_VARIANCE = 2.0
"""A pixel is dead when its grayscale variance over the sampled frames is at or under this, in counts².

The default comes from the simulated gel (48×64, camera noise of 6 counts, a 6×6 dead block) after
LeRobot's default encoding, AV1 at CRF 30 with a keyframe every two frames: the dead block's pixels
flicker at 0.6 to 1.7 counts² while every live pixel sits at 2.3 or above, so 2.0 takes the block
whole and no live pixel. A floor of 1.5 misses one pixel of the block; 3.0 adds a live one.
"""

ENCODER_FLOOR = 6.0
"""The rest flicker an image stream needs, in counts², for the dead-pixel rule to mean anything.

Three times :data:`DEAD_VARIANCE`. The encoder flattens weak camera noise to the same tenths of a
count a dead pixel flickers at, so on a stream quieter than this a dead pixel cannot be told from a
live one, and the check is skipped. The rest flicker is the median pixel variance of the instance's
quietest checked episode, not of each episode: contact lifts an episode's median while the regions
nothing touches stay at the rest level, and would read as dead. The simulated gel's medians are 10
to 11; a visuotactile stream encoded from a quiet camera sits near 1.
"""

CONSTANT_FRACTION = 1e-3
"""An element is constant when its standard deviation over the episode is within this fraction of its
largest magnitude: constant to one part in a thousand, on every channel."""

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
    """Per channel, the fraction of samples at or beyond the limit, and at each limit separately.

    ``data`` is ``(n, E, C)``, the raw readings (reconstructed when a baseline was
    subtracted, since a limit applies to what the sensor read). With ``limits``
    (``(min, max)`` per channel from ``range``) the result is ``declared`` and flagged on
    ``saturated_pct``. Without, the observed extremes stand in with ``confidence`` ``low``
    and the check never flags: a stream rests at an extreme as readily as it clips there
    (zero with nothing touching, a 0/1 contact flag, an ADC floor), so ``at_min_pct`` and
    ``at_max_pct`` are what a reader looks at, and a declared ``range`` is what makes the
    fraction a saturation.
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
    at_min, at_max = values <= lo + slack, values >= hi - slack
    none = np.zeros(values.shape[-1])
    pct = (at_min | at_max).mean(axis=(0, 1)) if values.size else none
    min_pct = at_min.mean(axis=(0, 1)) if values.size else none
    max_pct = at_max.mean(axis=(0, 1)) if values.size else none
    return SaturationReport(
        saturated_pct=[float(p) for p in pct],
        at_min_pct=[float(p) for p in min_pct],
        at_max_pct=[float(p) for p in max_pct],
        limit=limit,
        min=[float(v) for v in lo],
        max=[float(v) for v in hi],
        confidence=confidence,
        source=source,
        flag=limits is not None and bool(np.any(pct > thresholds.saturated)),
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
        confidence = "low" if limits is None else "normal"
    else:
        end = (
            slope * t[last] + intercept
        )  # the line at the last no-contact window: measured, not extrapolated
        offset = _extreme_per_channel(end)
        measure = np.abs(end)
        pct = None
        confidence = "normal"
    # The flag: a fraction of the declared range, or the caller's native-unit threshold; else never.
    if limits is not None:
        lo, hi = (np.asarray(v, dtype=np.float64) for v in limits)
        scaled = measure / (hi - lo)
        if not isinstance(offset, Skipped):
            pct = [float(v) for v in scaled.max(axis=0)]
        flag = bool(np.any(scaled.max(axis=0) > thresholds.drift))
    elif thresholds.drift_units is not None:
        flag = bool(np.any(measure.max(axis=0) > thresholds.drift_units))
    else:
        flag = False

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
        confidence=confidence,
        flag=flag,
    )


def _extreme_per_channel(values: NDArray[np.float64]) -> list[float]:
    """Per channel, the element with the largest magnitude, signed."""
    picked = np.take_along_axis(values, np.abs(values).argmax(axis=0)[None, :], axis=0)[0]
    return [float(v) for v in picked]


# --- Gel -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelStats:
    """What the pixel check reads off one episode's sampled frames; see :func:`pixel_stats`.

    ``median_variance`` is the median per-pixel grayscale variance in counts²; ``dead_pct`` the
    fraction of pixels at or under :data:`DEAD_VARIANCE` and ``largest_cluster_px`` the largest
    4-connected cluster of them; ``area`` the pixels in a frame; ``frames_used`` the frames sampled.
    """

    median_variance: float
    dead_pct: float
    largest_cluster_px: int
    area: int
    frames_used: int


def pixel_stats(frames: NDArray[np.uint8]) -> PixelStats:
    """The statistics :func:`gel_check_pixels` reads off an episode's sampled frames, uint8 ``(k, H, W, 3)``.

    Grayscale is the channel mean, the measure the evidence behind :data:`DEAD_VARIANCE` was taken
    with. With fewer than two frames the variance is undefined and the statistics are zero; the
    check then skips.
    """
    k = len(frames)
    area = int(frames.shape[1] * frames.shape[2]) if frames.ndim == 4 else 0
    if k < 2:
        return PixelStats(0.0, 0.0, 0, area, k)
    variance = np.asarray(frames, dtype=np.float32).mean(axis=-1).var(axis=0)
    dead = variance <= DEAD_VARIANCE
    return PixelStats(float(np.median(variance)), float(dead.mean()), largest_cluster(dead), area, k)


def gel_check_pixels(
    stats: PixelStats,
    thresholds: Thresholds,
    rest_variance: float | None = None,
    episodes_checked: int = 1,
) -> GelReport | Skipped:
    """Dead pixels of an image stream in one episode, from its :class:`PixelStats`.

    A pixel is dead when its variance is at or under :data:`DEAD_VARIANCE`, a rule that only
    means something when the stream's live pixels sit well above it. ``rest_variance`` is the
    instance's rest flicker, the median pixel variance of the quietest of its ``episodes_checked``
    episodes; when it is under :data:`ENCODER_FLOOR` the check is skipped with the number, on
    every episode of the instance, since contact lifts an episode's own median while the regions
    nothing touches stay at the rest level. Without ``rest_variance`` the episode's own median
    stands in. Flagged on the dead fraction (``gel``) or on the largest cluster as a fraction of
    the image (``cluster_pct``), or in pixels when ``cluster_px`` is set.
    """
    if stats.frames_used < 2:
        return Skipped(f"{stats.frames_used} frame(s) decoded: variance needs at least two")
    rest = stats.median_variance if rest_variance is None else rest_variance
    if rest < ENCODER_FLOOR:
        return Skipped(
            f"quiet gel: median pixel variance {rest:.2f} counts² in the quietest of {episodes_checked} "
            f"checked episode(s) is under the encoder floor {ENCODER_FLOOR:g}; a dead pixel cannot be "
            "told from a live one after encoding"
        )
    cluster_pct = stats.largest_cluster_px / stats.area if stats.area else 0.0
    if thresholds.cluster_px is not None:
        over_cluster = stats.largest_cluster_px > thresholds.cluster_px
    else:
        over_cluster = cluster_pct > thresholds.cluster_pct
    return GelReport(
        kind="pixels",
        dead_pct=stats.dead_pct,
        largest_cluster_px=stats.largest_cluster_px,
        dead_elements=None,
        frames_used=stats.frames_used,
        flag=stats.dead_pct > thresholds.gel or over_cluster,
        largest_cluster_pct=cluster_pct,
        median_variance=stats.median_variance,
        rest_variance=rest,
        episodes_checked=episodes_checked,
    )


def gel_check_elements(data: NDArray[np.floating], thresholds: Thresholds) -> GelReport | Skipped:
    """Constant elements of an array stream ``(n, E, C)`` in one episode, listed by index.

    An element is constant when, on every channel, its standard deviation over the episode is
    within :data:`CONSTANT_FRACTION` of its largest magnitude. The rule is per element, not
    relative to the other elements: a taxel array's elements share a unit but not a level of
    activity, and a plain dataset's "tactile" vector may hold quantities of different scales.
    One episode cannot tell a dead element from one nothing touched, so the block lists
    ``constant_elements`` as an observation, leaves ``dead_elements`` empty and does not flag;
    :func:`gel_dead_across` decides over the instance's episodes. A feature whose values are only
    0 and 1 is skipped: a constant zone of a contact flag is an untouched zone. So is an episode in
    which no element varies.
    """
    n = len(data)
    if n < 2:
        return Skipped(f"{n} frame(s): variance needs at least two")
    values = np.asarray(data, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and bool(np.all((finite == 0.0) | (finite == 1.0))):
        return Skipped(
            "binary feature (values 0 and 1): a constant element is an untouched zone, not a dead one"
        )
    spread = values.std(axis=0)
    scale = np.maximum(np.abs(values).max(axis=0), np.finfo(np.float32).tiny)
    constant = np.flatnonzero(np.all(spread <= CONSTANT_FRACTION * scale, axis=1))
    if constant.size == values.shape[1]:
        return Skipped("stream does not vary: no element changes in this episode")
    return GelReport(
        kind="elements",
        dead_pct=0.0,
        largest_cluster_px=None,
        dead_elements=[],
        frames_used=n,
        flag=False,
        constant_elements=[int(i) for i in constant],
        episodes_checked=1,
    )


def gel_dead_across(blocks: Sequence[GelReport], n_elements: int, thresholds: Thresholds) -> list[GelReport]:
    """The cross-episode step for the element blocks of one instance: an element constant in every
    checked episode is dead.

    Returns one block per input with ``dead_elements`` the intersection of their
    ``constant_elements``, ``dead_pct`` its fraction of ``n_elements``, ``episodes_checked`` the
    number of blocks, and the flag on ``dead_pct`` over ``gel``, set only with two or more
    episodes: one episode cannot separate a dead element from an untouched one. The caller
    groups the episodes, by session where a sidecar says which episodes share one, and leaves
    out the episodes whose check was skipped.
    """
    if not blocks:
        return []
    dead = sorted(set.intersection(*(set(b.constant_elements or []) for b in blocks)))
    pct = len(dead) / n_elements if n_elements else 0.0
    n = len(blocks)
    flag = n >= 2 and pct > thresholds.gel
    return [replace(b, dead_elements=dead, dead_pct=pct, episodes_checked=n, flag=flag) for b in blocks]


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
