"""Reducing native-rate frames to one dataset-rate frame, by the method declared in ``info.reduction``.

Every function here is pure: given the native samples of a stream (``host_ts``
and ``data``), a target time on the host clock and the window from
``info.reduction``, it returns the dataset-rate value. The writer uses them to
build the main-table copy at record time and a reader can call them on the
sidecar to reproduce it, which is what makes the copy verifiable.

Methods, for a target time ``T`` and a window ``w`` in seconds:

- ``mean``: the mean of the samples with ``T - w < host_ts <= T``.
- ``last``: the newest sample with ``T - w < host_ts <= T``.
- ``nearest``: the sample nearest to ``T`` among those with
  ``T - w/2 <= host_ts <= T + w/2``; a tie takes the earlier one.

``mean`` and ``last`` are causal: they only use samples received by ``T``, so a
frame built at or after its target time is complete and a reader gets the same
answer from the sidecar. ``nearest`` looks half a window ahead, so a sample that
arrives after the frame was built could have been the nearer one; the writer
therefore refuses it for array streams and it is here for readers and for video
features, where the encoder decides what the frame is.

A window with no sample reduces to ``None``. Over an episode, that frame repeats
the previous frame's value; the first frame of an episode falls back to the
newest sample at or before its time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from taxel.schema import REDUCTION_METHODS

__all__ = ["reduce_episode", "reduce_frame"]


def reduce_frame(
    method: str, host_ts: ArrayLike, data: ArrayLike, target: float, window_ms: float
) -> NDArray[np.float32] | None:
    """Reduce the samples in the window around ``target`` to one frame, or ``None`` if there is none.

    ``host_ts`` must be sorted ascending and ``data`` has one leading row per sample.
    The result is float32 with the per-sample shape.
    """
    times = np.asarray(host_ts, dtype=np.float64)
    values = np.asarray(data)
    if times.ndim != 1 or values.shape[:1] != times.shape:
        raise ValueError(f"host_ts {times.shape} and data {values.shape} must have one row per sample")
    window = _window_s(window_ms)
    if method == "mean":
        lo, hi = _causal_window(times, target, window)
        if hi == lo:
            return None
        return values[lo:hi].astype(np.float64).mean(axis=0).astype(np.float32)
    if method == "last":
        lo, hi = _causal_window(times, target, window)
        return None if hi == lo else values[hi - 1].astype(np.float32)
    if method == "nearest":
        lo = int(np.searchsorted(times, target - window / 2, side="left"))
        hi = int(np.searchsorted(times, target + window / 2, side="right"))
        if hi == lo:
            return None
        nearest = lo + int(np.argmin(np.abs(times[lo:hi] - target)))  # argmin takes the first on a tie
        return values[nearest].astype(np.float32)
    raise ValueError(f"reduction method must be one of {REDUCTION_METHODS}, got {method!r}")


def reduce_episode(
    method: str, host_ts: ArrayLike, data: ArrayLike, targets: ArrayLike, window_ms: float
) -> NDArray[np.float32]:
    """Reduce an episode's native stream at each target time, with the hold rule for empty windows.

    A target whose window holds no sample takes the previous frame's value. The
    first target falls back to the newest sample at or before it; if there is none
    either, the stream does not cover the episode and ``ValueError`` is raised.
    Returns float32 ``(len(targets), *shape)``.
    """
    times = np.asarray(host_ts, dtype=np.float64)
    values = np.asarray(data)
    wanted = np.atleast_1d(np.asarray(targets, dtype=np.float64))
    out = np.empty((len(wanted), *values.shape[1:]), dtype=np.float32)
    previous: NDArray[np.float32] | None = None
    for k, target in enumerate(wanted):
        frame = reduce_frame(method, times, values, float(target), window_ms)
        if frame is None:
            if previous is None:
                before = int(np.searchsorted(times, target, side="right"))
                if before == 0:
                    raise ValueError(f"no native sample at or before the first frame time {target}")
                frame = values[before - 1].astype(np.float32)
            else:
                frame = previous
        out[k] = frame
        previous = frame
    return out


def _window_s(window_ms: float) -> float:
    if isinstance(window_ms, bool) or not isinstance(window_ms, (int, float)) or not window_ms > 0:
        raise ValueError(f"window_ms must be a positive number, got {window_ms!r}")
    return window_ms / 1000.0


def _causal_window(times: NDArray[np.float64], target: float, window: float) -> tuple[int, int]:
    """Indices ``[lo, hi)`` of the samples with ``target - window < host_ts <= target``."""
    lo = int(np.searchsorted(times, target - window, side="right"))
    hi = int(np.searchsorted(times, target, side="right"))
    return lo, hi
