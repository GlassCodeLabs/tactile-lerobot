"""``TactileDatasetWriter``: records tactile sensors into a LeRobot dataset in the ``SCHEMA.md`` layout.

The caller creates the dataset through LeRobot's public API with the entries from
:func:`tactile_features`, then runs the writer beside its record loop::

    writer = TactileDatasetWriter(dataset, sensors)
    writer.start_session()
    for each episode:
        writer.start_episode()
        for each dataset frame:
            dataset.add_frame({**robot_frame, **writer.frame(), "task": task})
        writer.end_episode()
        dataset.save_episode()
        writer.idle()  # while the environment is reset
    writer.finalize()
    dataset.finalize()

At every frame the writer drains each sensor, keeps the native frames for the
episode and returns one dataset-rate array per instance, reduced from the native
frames by the method in the feature's ``info.reduction`` at the frame's nominal
time ``origin + k / fps``, where ``origin`` is the host clock when the episode's
first frame was built. At the end of the episode the native frames are written to
``tactile/<instance_id>/episode-{index:06d}.parquet`` and the session record in
``meta.json`` gets the episode range, the episode origin and the sensor's
counters. The dataset is used through ``fps``, ``root`` and ``meta`` only, so
nothing here imports LeRobot.
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from taxel import __version__
from taxel.dataset.features import feature_entries, read_feature_entry
from taxel.dataset.reduce import reduce_frame
from taxel.dataset.sidecar import (
    episode_path,
    meta_path,
    read_session_meta,
    write_episode,
    write_session_meta,
)
from taxel.schema import (
    ARRAY_FAMILIES,
    Block,
    FrameCounts,
    Reduction,
    SchemaError,
    SessionMeta,
    SessionRecord,
    TactileInfo,
    feature_key,
)
from taxel.sensors.base import TactileReading, TactileSensor, host_clock

__all__ = ["TactileDatasetWriter", "tactile_features"]

logger = logging.getLogger(__name__)

SESSION_FIELDS = frozenset({"serial", "firmware", "device", "calibration", "gel", "skin"})
"""Session record fields a caller may supply; the writer fills in the rest."""


def tactile_features(
    sensors: Iterable[TactileSensor], fps: float, method: str = "mean", window_ms: float | None = None
) -> dict[str, dict[str, Any]]:
    """Return the LeRobot feature entries for connected sensors recording at ``fps``.

    Each instance's ``info`` is its sensor's metadata, plus a ``reduction`` block when
    the sensor's declared rate differs from ``fps``: ``method`` and ``window_ms``
    (default ``1000 / fps``, the span of one dataset frame). The per-frame shape is
    read from the sensor's latest frame.

    Raises:
        ValueError: If ``method`` is ``nearest``, which cannot be reproduced from the
            sidecar for array streams; use ``mean`` or ``last``.
        NotImplementedError: For a sensor outside the array families.
    """
    _check_method(method)
    entries: dict[str, dict[str, Any]] = {}
    for sensor in sensors:
        info, shape = _instance(sensor)
        reduction = None
        if info.declared_rate_hz != fps:
            reduction = Reduction(method, round(1000.0 / fps, 3) if window_ms is None else window_ms)
        info = replace(info, reduction=reduction)
        entries.update(feature_entries(sensor.config.instance_id, info, shape))
    return entries


class TactileDatasetWriter:
    """Writes the tactile side of a LeRobot dataset: the per-frame copy, the sidecar, the session record.

    Args:
        dataset: A LeRobot dataset in write mode whose features contain the entries
            from :func:`tactile_features` for these sensors.
        sensors: Connected sensors, one per instance.
        baseline_s: Seconds of no-contact frames averaged into the session's baseline
            at :meth:`start_session`; ``0`` records no baseline.
        session: Values for the session record's ``serial``, ``firmware``, ``device``,
            ``calibration``, ``gel`` and ``skin``, as schema objects or dicts. Anything
            not given is written as ``null``.

    Raises:
        ValueError: If a sensor is not connected.
        SchemaError: If the dataset's feature entry for an instance does not match the
            sensor's metadata, or declares a reduction the writer does not produce.
    """

    def __init__(
        self,
        dataset: Any,
        sensors: Iterable[TactileSensor],
        *,
        baseline_s: float = 2.0,
        session: Mapping[str, Any] | None = None,
    ) -> None:
        self._dataset = dataset
        self._fps = float(dataset.fps)
        self._root = Path(dataset.root)
        self._baseline_s = baseline_s
        self._session_fields = {
            k: v.to_dict() if isinstance(v, Block) else v for k, v in (session or {}).items()
        }
        unknown = sorted(set(self._session_fields) - SESSION_FIELDS)
        if unknown:
            raise SchemaError(f"session fields {unknown} are the writer's; allowed: {sorted(SESSION_FIELDS)}")
        self._streams: dict[str, _Stream] = {}
        features = dataset.meta.features
        for sensor in sensors:
            instance_id = sensor.config.instance_id
            if instance_id in self._streams:
                raise ValueError(f"two sensors share instance_id {instance_id!r}")
            info, shape = _instance(sensor)
            key = feature_key(instance_id)
            if key not in features:
                raise SchemaError(f"dataset has no feature {key!r}; create it with tactile_features()")
            entry = read_feature_entry(features[key])
            if replace(info, reduction=entry.reduction) != entry:
                raise SchemaError(f"feature {key!r} does not match the metadata of {sensor!r}")
            entry_shape = tuple(features[key]["shape"])
            if entry_shape != shape:
                raise SchemaError(f"feature {key!r} has shape {entry_shape}, sensor frames are {shape}")
            if info.declared_rate_hz != self._fps:
                if entry.reduction is None:
                    raise SchemaError(f"feature {key!r} is faster than the dataset but has no reduction")
                _check_method(entry.reduction.method)
                method, window_ms = entry.reduction.method, entry.reduction.window_ms
            else:
                if entry.reduction is not None:
                    raise SchemaError(f"feature {key!r} runs at the dataset fps but declares a reduction")
                method, window_ms = "last", 1000.0 / self._fps
            self._streams[instance_id] = _Stream(sensor, key, shape, method, window_ms, self._root)
        if not self._streams:
            raise ValueError("no sensors given")
        self._session_id: str | None = None
        self._episode: int | None = None
        self._origin = 0.0
        self._n_frames = 0

    # --- Session ----------------------------------------------------------------------------------

    @property
    def instance_ids(self) -> list[str]:
        return list(self._streams)

    @property
    def episode_index(self) -> int | None:
        """Index of the open episode, or ``None`` between episodes."""
        return self._episode

    @property
    def records(self) -> dict[str, SessionRecord | None]:
        """The current session record per instance; ``None`` until the first episode ends."""
        return {instance_id: stream.record for instance_id, stream in self._streams.items()}

    def start_session(self, session_id: str | None = None) -> None:
        """Anchor the clock, capture the baselines and open a session record per instance.

        ``session_id`` defaults to the UTC time, ``YYYYMMDDTHHMMSSZ``. The wall time and
        host clock are read together here, so call this right after the sensors
        connect. Existing records in an instance's ``meta.json`` are kept and the new
        session is appended to them when its first episode ends.
        """
        if self._session_id is not None:
            raise RuntimeError("the session has already started")
        host = host_clock()
        wall = datetime.now(timezone.utc)
        wall_time = wall.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        self._session_id = session_id or wall.strftime("%Y%m%dT%H%M%SZ")
        for stream in self._streams.values():
            stream.open_session(self._session_id, wall_time, host, self._baseline_s, self._session_fields)

    # --- Episodes ---------------------------------------------------------------------------------

    def start_episode(self) -> int:
        """Open the episode LeRobot will save next and return its index."""
        self._require_session()
        if self._episode is not None:
            raise RuntimeError(f"episode {self._episode} is still open")
        self._episode = int(self._dataset.meta.total_episodes)
        self._n_frames = 0
        for stream in self._streams.values():
            stream.drain()
        return self._episode

    def frame(self) -> dict[str, NDArray[np.float32]]:
        """Build the tactile part of the next dataset frame: one array per instance, keyed by feature key.

        Frame ``k`` is reduced at ``origin + k / fps``; the first call sets the origin.
        A call that comes before that time waits for it, so every window is complete.
        """
        if self._episode is None:
            raise RuntimeError("call start_episode() first")
        k = self._n_frames
        if k == 0:
            self._origin = host_clock()
        target = self._origin + k / self._fps
        _wait_until(target)
        out: dict[str, NDArray[np.float32]] = {}
        for stream in self._streams.values():
            stream.drain()
            if k == 0:
                stream.begin_episode(self._origin)
            out[stream.key] = stream.reduce(target, k)
        self._n_frames = k + 1
        return out

    def end_episode(self) -> int:
        """Write the episode's native frames and update every session record; return the episode index."""
        if self._episode is None:
            raise RuntimeError("no episode is open")
        if self._n_frames == 0:
            raise RuntimeError(f"episode {self._episode} has no dataset frame; discard it instead")
        index = self._episode
        for stream in self._streams.values():
            stream.drain()
            stream.end_episode(index, self._origin)
        self._episode = None
        return index

    def discard_episode(self) -> None:
        """Drop the open episode's native frames, as LeRobot's ``clear_episode_buffer`` drops its frames."""
        if self._episode is None:
            raise RuntimeError("no episode is open")
        for stream in self._streams.values():
            stream.discard_episode()
        self._episode = None

    def idle(self) -> None:
        """Drain the sensors between episodes so their buffers do not overflow.

        Call it from the loop that runs while the environment is reset. Only the
        newest window of frames is kept, for the next episode's first frame.
        """
        for stream in self._streams.values():
            stream.drain()
            stream.trim_pending()

    def finalize(self) -> None:
        """Refresh the counters in every session record from the sensors and write every ``meta.json``.

        An episode still open is discarded with a warning. Call this after the sensors
        disconnect, or just before, so the counters are those of the whole session.
        """
        if self._episode is not None:
            logger.warning("finalize() with episode %d open; its native frames are discarded", self._episode)
            self.discard_episode()
        for stream in self._streams.values():
            stream.close_session()

    def _require_session(self) -> None:
        if self._session_id is None:
            raise RuntimeError("call start_session() first")


class _Stream:
    """One instance's buffers, reduction state and session record."""

    def __init__(
        self,
        sensor: TactileSensor,
        key: str,
        shape: tuple[int, ...],
        method: str,
        window_ms: float,
        root: Path,
    ) -> None:
        self.sensor = sensor
        self.key = key
        self.shape = shape
        self.method = method
        self.window_ms = window_ms
        self.root = root
        self.instance_id = sensor.config.instance_id
        self.record: SessionRecord | None = None
        self._fields: dict[str, Any] = {}
        self._existing: list[SessionRecord] = []
        self._origins: list[float] = []
        self._first_episode: int | None = None
        self._pending: list[TactileReading] = []
        self._readings: list[TactileReading] = []
        self._ts: list[float] = []
        self._in_episode = False
        self._previous: NDArray[np.float32] | None = None
        self._newest: TactileReading | None = None

    # --- Session ----------------------------------------------------------------------------------

    def open_session(
        self, session_id: str, wall_time: str, host: float, baseline_s: float, fields: Mapping[str, Any]
    ) -> None:
        if meta_path(self.root, self.instance_id).exists():
            self._existing = list(read_session_meta(self.root, self.instance_id).sessions)
            if any(s.session_id == session_id for s in self._existing):
                raise SchemaError(f"{self.instance_id}: session {session_id!r} already exists in meta.json")
        baseline = self.sensor.capture_baseline(baseline_s).to_dict() if baseline_s > 0 else None
        self._fields = {
            "session_id": session_id,
            "clock": {"wall_time": wall_time, "host_clock": host, "episode_origins": []},
            "adapter": {"name": f"taxel.{self.sensor.config.type}", "version": __version__},
            "calibration": None,
            "baseline": baseline,
            **dict(fields),
        }

    def write_meta(self) -> None:
        if self.record is None:
            return
        meta = SessionMeta(instance_id=self.instance_id, sessions=[*self._existing, self.record])
        write_session_meta(self.root, meta)

    def close_session(self) -> None:
        if self.record is not None:
            self.record = replace(self.record, frames=self._counters())
            self.write_meta()

    def _counters(self) -> FrameCounts:
        return FrameCounts(
            received=self.sensor.frames_received,
            dropped=self.sensor.dropped_frames,
            overflowed=self.sensor.overflowed_frames,
        )

    # --- Buffers ----------------------------------------------------------------------------------

    def drain(self) -> None:
        frames = self.sensor.drain()
        if not frames:
            return
        self._newest = frames[-1]
        if self._in_episode:
            self._readings.extend(frames)
            self._ts.extend(f.host_ts for f in frames)
        else:
            self._pending.extend(frames)

    def trim_pending(self) -> None:
        """Keep the newest window of pending frames, and at least the newest frame."""
        cutoff = host_clock() - self.window_ms / 1000.0
        kept = [f for f in self._pending if f.host_ts > cutoff]
        self._pending = kept or self._pending[-1:]

    def begin_episode(self, origin: float) -> None:
        """Start the episode's native stream with the pending frames the first window needs."""
        cutoff = origin - self.window_ms / 1000.0
        kept = [f for f in self._pending if f.host_ts > cutoff]
        if not any(f.host_ts <= origin for f in kept):
            older = [f for f in self._pending if f.host_ts <= origin]
            fallback = older[-1] if older else self._newest
            if fallback is not None and fallback.host_ts <= origin:
                kept.insert(0, fallback)
        self._pending = []
        self._readings = kept + self._readings
        self._ts = [f.host_ts for f in self._readings]
        self._in_episode = True
        self._previous = None

    def reduce(self, target: float, k: int) -> NDArray[np.float32]:
        """The dataset-rate frame at ``target``; the hold rule applies when the window is empty."""
        window_s = self.window_ms / 1000.0
        lo = max(bisect_right(self._ts, target - window_s) - 1, 0)
        hi = min(bisect_right(self._ts, target) + 1, len(self._ts))
        value = None
        if hi > lo:
            data = np.stack([f.data for f in self._readings[lo:hi]])
            value = reduce_frame(self.method, self._ts[lo:hi], data, target, self.window_ms)
        if value is None:
            if self._previous is not None:
                value = self._previous
            else:
                before = bisect_right(self._ts, target)
                if before == 0:
                    raise RuntimeError(f"{self.instance_id}: no native frame at or before dataset frame {k}")
                value = self._readings[before - 1].data.astype(np.float32)
        self._previous = value
        return value

    def end_episode(self, index: int, origin: float) -> None:
        write_episode(episode_path(self.root, self.instance_id, index), index, self._readings, self.shape)
        self._origins.append(origin)
        if self._first_episode is None:
            self._first_episode = index
        self.record = SessionRecord.from_dict(
            {
                **self._fields,
                "episodes": [self._first_episode, index],
                "clock": {**self._fields["clock"], "episode_origins": list(self._origins)},
                "frames": self._counters().to_dict(),
            }
        )
        self.write_meta()
        self._clear()

    def discard_episode(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self._readings = []
        self._ts = []
        self._in_episode = False
        self._previous = None


def _instance(sensor: TactileSensor) -> tuple[TactileInfo, tuple[int, ...]]:
    if not sensor.is_connected:
        raise ValueError(f"{sensor!r} is not connected")
    info = sensor.metadata()
    if info.family not in ARRAY_FAMILIES:
        raise NotImplementedError(f"{sensor!r}: the writer records array families only, not {info.family}")
    return info, tuple(sensor.read_latest(max_age_ms=float("inf")).data.shape)


def _check_method(method: str) -> None:
    if method == "nearest":
        raise ValueError(
            "'nearest' looks half a window ahead of the frame time and cannot be reproduced from the "
            "sidecar at record time; use 'mean' or 'last' for array streams"
        )
    Reduction(method, 1.0)  # validates the name


def _wait_until(target: float) -> None:
    remaining = target - host_clock()
    if remaining > 0.002:
        time.sleep(remaining - 0.001)
    while host_clock() < target:
        pass
