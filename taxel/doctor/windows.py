"""Window statistics and no-contact detection for the drift check.

Everything here is numpy only and pure. A stream is cut into windows of a fixed number of
samples, restarted at every re-baseline so no window straddles one. A window is *quiet*
when, after removing its own linear trend, every element's spread is within a few times
the stream's noise floor, which excludes press onsets and releases but keeps a slow drift
ramp and a held press. Among the quiet windows, the largest set that lies on one line, the
lower one on a tie, is *no-contact* and that line is the drift; the quiet windows off it
are *contact* (a held press is a plateau above the rest level).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["WindowStats", "no_contact", "window_edges", "window_stats"]

_MAX_PAIRS = 50_000


@dataclass(frozen=True)
class WindowStats:
    """Per-window statistics of a ``(n, n_elements, n_channels)`` stream.

    Attributes:
        starts: First sample of each window, shape ``(W,)``.
        ends: One past the last sample of each window, shape ``(W,)``.
        level: Mean of the stream in each window, ``(W, E, C)``.
        spread: Standard deviation after removing each window's own linear trend, ``(W, E, C)``.
        noise: The noise floor per element and channel: the median spread over windows, ``(E, C)``.
        quiet: Windows whose spread is within ``quiet_factor`` times the noise floor on every element.
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
) -> WindowStats:
    """Level, detrended spread and quietness of each window of ``data`` (``(n, E, C)``)."""
    values = np.asarray(data, dtype=np.float64)
    level = np.empty((len(starts), *values.shape[1:]))
    spread = np.empty_like(level)
    for w, (s, e) in enumerate(zip(starts, ends, strict=True)):
        block = values[s:e]
        t = np.arange(e - s, dtype=np.float64) - (e - s - 1) / 2.0
        mean = block.mean(axis=0)
        slope = np.tensordot(t, block - mean, axes=(0, 0)) / float(t @ t)
        residual = block - mean - t[:, None, None] * slope
        level[w] = mean
        spread[w] = residual.std(axis=0)
    noise = np.median(spread, axis=0) if len(starts) else np.zeros(values.shape[1:])
    quiet = np.all(spread <= quiet_factor * noise + atol, axis=(1, 2)) if len(starts) else np.zeros(0, bool)
    return WindowStats(starts, ends, level, spread, noise, quiet)


def no_contact(
    y: NDArray[np.floating],
    t: NDArray[np.floating],
    quiet: NDArray[np.bool_],
    noise: NDArray[np.floating],
    contact_factor: float = 6.0,
    atol: float = 1e-6,
    max_candidates: int = 2000,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Split the quiet windows into no-contact ones and held presses, and fit the drift line.

    ``y`` is a per-window level, ``(W, E, C)``, ``t`` the window times. Every pair of quiet
    windows proposes a line per element and channel; the line with the most quiet windows
    within ``contact_factor`` times the noise floor on every element wins, the one at the
    lower level on a tie, since a press adds to the rest level. A held press through half
    the windows therefore still loses to the rest windows. The winning set is refitted
    robustly (median of pairwise slopes, median intercept) and the mask is the quiet
    windows on that line. Returns the no-contact mask ``(W,)`` and the line's ``slope`` and
    ``intercept``, ``(E, C)`` each. With fewer than two quiet windows the slope is zero and
    the intercept the quiet level.
    """
    levels = np.asarray(y, dtype=np.float64)
    times = np.asarray(t, dtype=np.float64)
    idx = np.flatnonzero(quiet)
    shape = levels.shape[1:]
    mask = np.zeros(len(levels), bool)
    if idx.size == 0:
        return mask, np.zeros(shape), np.full(shape, np.nan)
    if idx.size == 1:
        mask[idx] = True
        return mask, np.zeros(shape), levels[idx[0]].copy()
    yq, tq = levels[idx], times[idx]
    tolerance = np.maximum(contact_factor * np.asarray(noise, dtype=np.float64), atol)
    i, j = _pairs(idx.size, max_candidates)
    slopes = (yq[j] - yq[i]) / (tq[j] - tq[i])[:, None, None]
    intercepts = yq[i] - slopes * tq[i][:, None, None]
    best: tuple[int, float] | None = None
    inliers = np.ones(idx.size, bool)
    for p in range(len(i)):
        residual = np.abs(yq - (slopes[p] * tq[:, None, None] + intercepts[p]))
        inside = np.all(residual <= tolerance, axis=(1, 2))
        count = int(inside.sum())
        magnitude = float(np.abs(yq[inside]).mean()) if count else np.inf
        key = (count, -magnitude)
        if best is None or key > best:
            best, inliers = key, inside
    slope, intercept = _robust_line(yq[inliers], tq[inliers])
    residual = np.abs(yq - (slope * tq[:, None, None] + intercept))
    mask[idx[np.all(residual <= tolerance, axis=(1, 2))]] = True
    return mask, slope, intercept


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
    slope = np.median((y[j] - y[i]) / (t[j] - t[i])[:, None, None], axis=0)
    intercept = np.median(y - slope * t[:, None, None], axis=0)
    return slope, intercept
