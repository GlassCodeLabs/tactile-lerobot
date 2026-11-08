"""Window statistics and no-contact detection for the drift check, the onset and motion-energy
signals the skew check correlates, and the cluster labelling the gel check needs.

Everything here is numpy only and pure. A stream is cut into windows of a fixed number of
samples, restarted at every re-baseline so no window straddles one. A window is *quiet*
when, after removing its own linear trend, nine tenths of its element-channels have a spread
within a few times the stream's noise floor, which excludes press onsets and releases but
keeps a slow drift ramp and a held press. Among the quiet windows, the largest set that lies
on one line, the lower one on a tie, is *no-contact* and that line is the drift; the quiet
windows off it are *contact* (a held press is a plateau above the rest level). NaN samples
are left out of every statistic; an element-channel with fewer than two finite samples in a
window counts against that window.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MAX_CANDIDATES",
    "QUIET_FRACTION",
    "NoContactFit",
    "WindowStats",
    "contact_signal",
    "cross_correlate",
    "largest_cluster",
    "motion_energy",
    "nanmedian",
    "no_contact",
    "window_edges",
    "window_stats",
]

MAX_CANDIDATES = 300
"""Pairs of quiet windows tried as drift lines; a seeded sample of this many when there are more."""

_MAX_PAIRS = 50_000

QUIET_FRACTION = 0.9
"""The share of a window's element-channels that must sit within their noise floor for the window to be
quiet, and within a candidate line's tolerance for the window to lie on it.

Not every one: on a glove of 256 or 880 cells some cell is always moving or flickering, and a rule over
every element finds no quiet window on exactly the arrays that drift. Nine tenths still keeps out a press
on a quarter of a 16-element array; a press held through most of an episode on fewer than a tenth of the
elements reads as rest for them.
"""


@dataclass(frozen=True)
class WindowStats:
    """Per-window statistics of a ``(n, n_elements, n_channels)`` stream.

    Attributes:
        starts: First sample of each window, shape ``(W,)``.
        ends: One past the last sample of each window, shape ``(W,)``.
        level: Mean of the stream in each window, ``(W, E, C)``.
        spread: Standard deviation after removing each window's own linear trend, ``(W, E, C)``.
        noise: The noise floor per element and channel: the lower quartile of the spread over windows,
            ``(E, C)``, so that press transients in up to three quarters of the windows do not lift it.
        quiet: Windows whose spread is within ``quiet_factor`` times the noise floor on at least
            :data:`QUIET_FRACTION` of their element-channels.
    """

    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    level: NDArray[np.float64]
    spread: NDArray[np.float64]
    noise: NDArray[np.float64]
    quiet: NDArray[np.bool_]

    def __len__(self) -> int:
        return len(self.starts)

    def centres(self, rate_hz: float) -> NDArray[np.float64]:
        """Centre time of each window in seconds, at ``rate_hz`` samples per second."""
        return (self.starts + self.ends) / 2.0 / rate_hz


def window_edges(
    n: int, window: int, breaks: Sequence[int] = ()
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Cut ``n`` samples into consecutive windows of ``window`` samples, restarting at each break.

    A break is the first sample a new baseline applies to. The last window of a segment
    may be shorter; it is kept when it holds at least half a window and two samples.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2 samples, got {window}")
    edges = sorted({0, n, *(int(b) for b in breaks if 0 < b < n)})
    starts: list[int] = []
    ends: list[int] = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        for s in range(a, b, window):
            e = min(s + window, b)
            if e - s >= max(2, window // 2):
                starts.append(s)
                ends.append(e)
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def window_stats(
    data: NDArray[np.floating],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    quiet_factor: float = 4.0,
    atol: float = 1e-6,
    quiet_fraction: float = QUIET_FRACTION,
) -> WindowStats:
    """Level, detrended spread and quietness of each window of ``data`` (``(n, E, C)``).

    NaN samples are left out. An element-channel with fewer than two finite samples in a window
    has no level there (NaN) and an infinite spread, so it is never within its noise floor.
    """
    values = np.asarray(data, dtype=np.float64)
    level = np.empty((len(starts), *values.shape[1:]))
    spread = np.empty_like(level)
    for w, (s, e) in enumerate(zip(starts, ends, strict=True)):
        level[w], spread[w] = _detrended(values[s:e])
    if not len(starts):
        return WindowStats(starts, ends, level, spread, np.zeros(values.shape[1:]), np.zeros(0, bool))
    if np.isfinite(spread).all():
        noise = np.percentile(spread, 25, axis=0)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # no finite window at all: no floor (NaN)
            noise = np.nanpercentile(np.where(np.isfinite(spread), spread, np.nan), 25, axis=0)
    within = spread <= quiet_factor * noise + atol
    quiet = within.mean(axis=(1, 2)) >= quiet_fraction
    return WindowStats(starts, ends, level, spread, noise, quiet)


def _detrended(block: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A window's mean and its spread about its own linear trend, per element-channel, over finite samples.

    An element-channel with fewer than two finite samples gets a NaN mean and an infinite spread.
    """
    m = len(block)
    t = np.arange(m, dtype=np.float64) - (m - 1) / 2.0
    present = ~np.isnan(block)
    if present.all():
        mean = block.mean(axis=0)
        slope = np.tensordot(t, block - mean, axes=(0, 0)) / float(t @ t)
        residual = block - mean - t[:, None, None] * slope
        return mean, residual.std(axis=0)
    count = present.sum(axis=0)
    enough = count >= 2
    safe = np.where(enough, count, 1)
    times = np.where(present, t[:, None, None], 0.0)
    mean = np.where(present, block, 0.0).sum(axis=0) / safe
    dt = np.where(present, times - times.sum(axis=0) / safe, 0.0)
    dx = np.where(present, block - mean, 0.0)
    sxx = (dt * dt).sum(axis=0)
    slope = (dt * dx).sum(axis=0) / np.where(sxx > 0, sxx, 1.0)
    residual = np.where(present, dx - dt * slope, 0.0)
    spread = np.sqrt((residual * residual).sum(axis=0) / safe)
    return np.where(enough, mean, np.nan), np.where(enough, spread, np.inf)


@dataclass(frozen=True)
class NoContactFit:
    """The no-contact windows and the drift line through them.

    Attributes:
        mask: No-contact windows, shape ``(W,)``.
        slope: The line's slope per element and channel, ``(E, C)``.
        intercept: The line's intercept, ``(E, C)``; NaN when no window was quiet.
        candidates: Pairs of quiet windows tried as lines, at most :data:`MAX_CANDIDATES`.
    """

    mask: NDArray[np.bool_]
    slope: NDArray[np.float64]
    intercept: NDArray[np.float64]
    candidates: int


def no_contact(
    y: NDArray[np.floating],
    t: NDArray[np.floating],
    quiet: NDArray[np.bool_],
    noise: NDArray[np.floating],
    contact_factor: float = 6.0,
    atol: float = 1e-6,
    max_candidates: int = MAX_CANDIDATES,
    inlier_fraction: float = QUIET_FRACTION,
) -> NoContactFit:
    """Split the quiet windows into no-contact ones and held presses, and fit the drift line.

    ``y`` is a per-window level, ``(W, E, C)``, ``t`` the window times. Every pair of quiet
    windows proposes a line per element and channel; the line with the most quiet windows
    within ``contact_factor`` times the noise floor on ``inlier_fraction`` of their
    element-channels wins, the one at the lower level on a tie, since a press adds to the rest
    level. A held press through half the windows therefore still loses to the rest windows.
    The winning set is refitted robustly (median of pairwise slopes, median intercept) and the
    mask is the quiet windows on that line. When the quiet windows make more than
    ``max_candidates`` pairs, a seeded sample of that many is tried. With fewer than two quiet
    windows the slope is zero and the intercept the quiet level. A NaN level, an element-channel
    without finite samples in a window, is never on a line and is left out of the refit.
    """
    levels = np.asarray(y, dtype=np.float64)
    times = np.asarray(t, dtype=np.float64)
    idx = np.flatnonzero(quiet)
    shape = levels.shape[1:]
    mask = np.zeros(len(levels), bool)
    if idx.size == 0:
        return NoContactFit(mask, np.zeros(shape), np.full(shape, np.nan), 0)
    if idx.size == 1:
        mask[idx] = True
        return NoContactFit(mask, np.zeros(shape), levels[idx[0]].copy(), 0)
    yq, tq = levels[idx], times[idx]
    tolerance = np.maximum(contact_factor * np.asarray(noise, dtype=np.float64), atol)
    i, j = _pairs(idx.size, max_candidates)
    slopes = (yq[j] - yq[i]) / (tq[j] - tq[i])[:, None, None]
    intercepts = yq[i] - slopes * tq[i][:, None, None]
    best: tuple[int, float] | None = None
    inliers = np.ones(idx.size, bool)
    for p in range(len(i)):
        residual = np.abs(yq - (slopes[p] * tq[:, None, None] + intercepts[p]))
        inside = (residual <= tolerance).mean(axis=(1, 2)) >= inlier_fraction
        count = int(inside.sum())
        magnitude = _finite_mean(np.abs(yq[inside])) if count else np.inf
        key = (count, -magnitude)
        if best is None or key > best:
            best, inliers = key, inside
    slope, intercept = _robust_line(yq[inliers], tq[inliers])
    residual = np.abs(yq - (slope * tq[:, None, None] + intercept))
    mask[idx[(residual <= tolerance).mean(axis=(1, 2)) >= inlier_fraction]] = True
    return NoContactFit(mask, slope, intercept, int(len(i)))


def _finite_mean(values: NDArray[np.float64]) -> float:
    """Mean of the finite values, ``inf`` when there are none."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("inf")


def nanmedian(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Median along ``axis`` over the non-NaN values; NaN where there are none."""
    if not np.isnan(values).any():
        return np.median(values, axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN slice has no median: NaN
        return np.nanmedian(values, axis=axis)


def _pairs(k: int, limit: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """All pairs ``i < j`` of ``k`` items, or ``limit`` of them chosen with a fixed seed."""
    i, j = np.triu_indices(k, k=1)
    if i.size > limit:
        pick = np.random.default_rng(0).choice(i.size, limit, replace=False)
        i, j = i[pick], j[pick]
    return i, j


def _robust_line(
    y: NDArray[np.float64], t: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Median of pairwise slopes, then median intercept, per element and channel; one point has slope 0."""
    if len(t) < 2:
        return np.zeros(y.shape[1:]), y[0].copy()
    i, j = _pairs(len(t), _MAX_PAIRS)
    slope = nanmedian((y[j] - y[i]) / (t[j] - t[i])[:, None, None], axis=0)
    intercept = nanmedian(y - slope * t[:, None, None], axis=0)
    return slope, intercept


# --- Signals for the skew check ----------------------------------------------------------------------


def contact_signal(data: NDArray[np.floating]) -> NDArray[np.float64]:
    """Per-frame contact magnitude of an array stream ``(n, E, C)``: the L1 distance of each frame
    from the episode's median frame, summed over elements and channels. A press is a step in it.

    With NaN samples the distance is the mean over the frame's finite element-channels, scaled to
    all of them, so a frame that lost a few elements is not a step; a frame with none finite
    repeats the previous value, which is no event.
    """
    values = np.asarray(data, dtype=np.float64)
    if not np.isnan(values).any():
        return np.abs(values - np.median(values, axis=0)).sum(axis=(1, 2))
    deviation = np.abs(values - nanmedian(values, axis=0)).reshape(len(values), -1)
    present = np.isfinite(deviation)
    count = present.sum(axis=1)
    signal = np.where(present, deviation, 0.0).sum(axis=1) * deviation.shape[1] / np.maximum(count, 1)
    for k in np.flatnonzero(count == 0):
        signal[k] = signal[k - 1] if k else 0.0
    return signal


def motion_energy(frames: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Mean absolute grayscale difference between consecutive frames ``(k, H, W, 3)``, shape ``(k - 1,)``."""
    gray = np.asarray(frames, dtype=np.float32).mean(axis=-1)
    if len(gray) < 2:
        return np.zeros(0)
    return np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2)).astype(np.float64)


def cross_correlate(
    times_a: NDArray[np.floating],
    a: NDArray[np.floating],
    times_b: NDArray[np.floating],
    b: NDArray[np.floating],
    max_lag_s: float = 0.5,
    grid_s: float = 0.005,
    smooth_s: float | None = None,
) -> tuple[float, float, int]:
    """Lag of event series ``a`` behind ``b``, the normalized cross-correlation peak, and the onsets in ``a``.

    Both series are event-like (large at onsets and releases, near zero otherwise); each has its
    floor (median) removed and is binned onto a grid of ``grid_s``. With ``smooth_s`` both are
    smoothed by a triangle of that half-width, so two events whose timing is quantized to
    different phases of a coarser signal (video frames) still overlap at one lag. The lag is the
    one in ``[-max_lag_s, max_lag_s]`` that maximizes the normalized cross-correlation: positive
    when ``a``'s events come later than ``b``'s. The peak value, 0 to 1, is the confidence.
    Onsets are the local maxima of the unsmoothed ``a`` above a third of its largest value.
    """
    ta, tb = np.asarray(times_a, dtype=np.float64), np.asarray(times_b, dtype=np.float64)
    va = np.clip(np.asarray(a, dtype=np.float64) - np.median(a), 0.0, None)
    vb = np.clip(np.asarray(b, dtype=np.float64) - np.median(b), 0.0, None)
    t0 = min(ta.min(), tb.min())
    n = int(np.ceil((max(ta.max(), tb.max()) - t0) / grid_s)) + 1
    ga, gb = np.zeros(n), np.zeros(n)
    np.add.at(ga, np.rint((ta - t0) / grid_s).astype(int), va)
    np.add.at(gb, np.rint((tb - t0) / grid_s).astype(int), vb)
    peaks = (
        (ga[1:-1] > ga[:-2]) & (ga[1:-1] >= ga[2:]) & (ga[1:-1] > ga.max() / 3)
        if n > 2
        else np.zeros(0, bool)
    )
    onsets = int(peaks.sum())
    if smooth_s is not None and smooth_s > grid_s:
        half = int(round(smooth_s / grid_s))
        kernel = 1.0 - np.abs(np.arange(-half, half + 1)) / (half + 1)
        ga, gb = np.convolve(ga, kernel, mode="same"), np.convolve(gb, kernel, mode="same")
    norm = float(np.linalg.norm(ga) * np.linalg.norm(gb))
    if norm == 0.0:
        return 0.0, 0.0, onsets
    limit = min(int(round(max_lag_s / grid_s)), n - 1)
    # full[k] is sum over m of ga[m + k - (n - 1)] * gb[m]: the score of lag k - (n - 1).
    full = np.correlate(ga, gb, mode="full")
    lags = np.arange(-limit, limit + 1)
    scores = full[lags + n - 1]
    best = int(np.argmax(scores))
    return float(lags[best] * grid_s), max(float(scores[best]) / norm, 0.0), onsets


# --- Clusters for the gel check ---------------------------------------------------------------------------


def largest_cluster(mask: NDArray[np.bool_]) -> int:
    """Size in pixels of the largest 4-connected region of ``True`` in a 2-D mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    labels = np.where(mask, np.arange(mask.size).reshape(mask.shape), -1)
    while True:
        new = labels.copy()
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            neighbour = np.roll(labels, shift, axis=axis)
            valid = mask & np.roll(mask, shift, axis=axis)
            if shift == 1:
                valid[(slice(0, 1), slice(None)) if axis == 0 else (slice(None), slice(0, 1))] = False
            else:
                valid[(slice(-1, None), slice(None)) if axis == 0 else (slice(None), slice(-1, None))] = False
            new = np.where(valid, np.minimum(new, neighbour), new)
        if np.array_equal(new, labels):
            break
        labels = new
    _, counts = np.unique(labels[mask], return_counts=True)
    return int(counts.max())
