"""The per-instance sidecar: ``tactile/<instance_id>/meta.json`` and the native-rate episode files.

``meta.json`` is a :class:`~taxel.schema.SessionMeta`. Each
``episode-{index:06d}.parquet`` holds every frame the sensor delivered during one
episode with the five columns of ``SCHEMA.md`` plus ``sample_index``:
``episode_index``, ``sample_index``, ``host_ts``, ``device_ts``, ``seq`` and
``data``. ``data`` is an Arrow fixed-shape tensor column, so pyarrow reads it back
as one ``(n_samples, *shape)`` array. A suffixed stream keeps the same files in
its own directory below the instance's.

pyarrow is imported where it is used, so the alignment helpers and the paths work
with numpy alone.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from taxel.dataset import missing_dependency
from taxel.schema import SchemaError, SessionMeta, feature_key, session_meta_path

if TYPE_CHECKING:
    from taxel.sensors.base import TactileReading

__all__ = [
    "COLUMNS",
    "EpisodeStream",
    "align_nearest",
    "episode_path",
    "frame_times",
    "meta_path",
    "read_episode",
    "read_session_meta",
    "sidecar_dir",
    "write_episode",
    "write_session_meta",
]

SIDECAR_DIR = "tactile"
COLUMNS = ("episode_index", "sample_index", "host_ts", "device_ts", "seq", "data")
_EPISODE_FILE = "episode-{index:06d}.parquet"
_EPISODE_INDEX_META = b"taxel.episode_index"


# --- Paths ---------------------------------------------------------------------------------------


def sidecar_dir(root: str | Path, instance_id: str, suffix: str | None = None) -> Path:
    """``<root>/tactile/<instance_id>``, or the stream's own directory below it for a suffix."""
    feature_key(instance_id, suffix)  # validates the slug and the suffix
    path = Path(root) / SIDECAR_DIR / instance_id
    return path / suffix if suffix else path


def meta_path(root: str | Path, instance_id: str) -> Path:
    """``<root>/tactile/<instance_id>/meta.json``."""
    return Path(root) / session_meta_path(instance_id)


def episode_path(root: str | Path, instance_id: str, episode_index: int, suffix: str | None = None) -> Path:
    """``<root>/tactile/<instance_id>[/<suffix>]/episode-{index:06d}.parquet``."""
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError(f"episode_index must be a non-negative integer, got {episode_index!r}")
    return sidecar_dir(root, instance_id, suffix) / _EPISODE_FILE.format(index=episode_index)


# --- meta.json -----------------------------------------------------------------------------------


def write_session_meta(root: str | Path, meta: SessionMeta) -> Path:
    """Write ``meta.json`` for ``meta.instance_id`` and return its path. Creates the directories."""
    path = meta_path(root, meta.instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n")
    return path


def read_session_meta(root: str | Path, instance_id: str) -> SessionMeta:
    """Read and validate ``meta.json`` for an instance.

    Raises:
        FileNotFoundError: If the instance has no sidecar.
        SchemaError: If the file does not satisfy ``SCHEMA.md``.
    """
    path = meta_path(root, instance_id)
    meta = SessionMeta.from_dict(json.loads(path.read_text()))
    if meta.instance_id != instance_id:
        raise SchemaError(f"{path} is for instance {meta.instance_id!r}, not {instance_id!r}")
    return meta


# --- Episode files ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeStream:
    """The native-rate stream of one instance, or one of its suffixed streams, in one episode.

    Attributes:
        episode_index: The episode every sample belongs to.
        host_ts: Host receive time of each sample, seconds on the host clock, shape ``(n,)``.
        device_ts: The sensor's own timestamps, shape ``(n,)``, or ``None`` when it has no clock.
        seq: The sensor's frame counter, shape ``(n,)``, or ``None`` when it has none.
        data: The samples, float32 ``(n, *shape)``.
    """

    episode_index: int
    host_ts: NDArray[np.float64]
    device_ts: NDArray[np.float64] | None
    seq: NDArray[np.int64] | None
    data: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.host_ts)

    @property
    def shape(self) -> tuple[int, ...]:
        """Per-sample shape."""
        return self.data.shape[1:]


def write_episode(
    path: str | Path, episode_index: int, readings: Sequence[TactileReading], shape: tuple[int, ...]
) -> Path:
    """Write one episode's native frames, oldest first, and return the file path.

    ``shape`` is the per-frame shape, needed so an episode with no frames still
    writes a typed file. Every reading must be float32 of that shape, and ``host_ts``
    must not decrease.

    Raises:
        ValueError: If a reading has the wrong dtype or shape, or ``host_ts`` goes backwards.
    """
    pa, pq = _pyarrow()
    path = Path(path)
    shape = tuple(int(d) for d in shape)
    n = len(readings)
    data = np.empty((n, *shape), dtype=np.float32)
    host_ts = np.empty(n, dtype=np.float64)
    device_ts: list[float | None] = []
    seq: list[int | None] = []
    for i, reading in enumerate(readings):
        if reading.data.dtype != np.float32 or reading.data.shape != shape:
            raise ValueError(
                f"sample {i}: expected float32 {shape}, got {reading.data.dtype} {reading.data.shape}"
            )
        data[i] = reading.data
        host_ts[i] = reading.host_ts
        device_ts.append(reading.device_ts)
        seq.append(reading.seq)
    if n > 1 and np.any(np.diff(host_ts) < 0):
        raise ValueError("host_ts must not decrease within an episode")

    tensor_type = pa.fixed_shape_tensor(pa.float32(), list(shape))
    storage = pa.FixedSizeListArray.from_arrays(pa.array(data.ravel()), int(np.prod(shape)))
    table = pa.table(
        {
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
            "sample_index": pa.array(np.arange(n, dtype=np.int64)),
            "host_ts": pa.array(host_ts),
            "device_ts": pa.array(device_ts, type=pa.float64()),
            "seq": pa.array(seq, type=pa.int64()),
            "data": pa.ExtensionArray.from_storage(tensor_type, storage),
        }
    ).replace_schema_metadata({_EPISODE_INDEX_META: str(episode_index).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    return path


def read_episode(path: str | Path) -> EpisodeStream:
    """Read one episode file back.

    Raises:
        SchemaError: If a column is missing, ``data`` is not a fixed-shape tensor column,
            or ``seq`` is set on some samples and missing on others.
    """
    pa, pq = _pyarrow()
    path = Path(path)
    table = pq.read_table(path)
    missing = [c for c in COLUMNS if c not in table.column_names]
    if missing:
        raise SchemaError(f"{path}: missing columns {missing}")
    data_col = table.column("data").combine_chunks()
    if not isinstance(data_col.type, pa.FixedShapeTensorType):
        raise SchemaError(f"{path}: 'data' must be a fixed-shape tensor column, got {data_col.type}")
    shape = tuple(data_col.type.shape)
    n = len(table)
    if n:
        data = data_col.to_numpy_ndarray().astype(np.float32, copy=False)
        episode_index = int(table.column("episode_index")[0].as_py())
    else:
        data = np.empty((0, *shape), dtype=np.float32)
        meta = (table.schema.metadata or {}).get(_EPISODE_INDEX_META)
        if meta is None:
            raise SchemaError(f"{path}: empty episode file without an episode index")
        episode_index = int(meta.decode())
    host_ts = table.column("host_ts").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    device_col = table.column("device_ts")
    device_ts = None if device_col.null_count == n else device_col.to_numpy(zero_copy_only=False)
    seq_col = table.column("seq")
    if seq_col.null_count == n:
        seq = None
    elif seq_col.null_count:
        raise SchemaError(f"{path}: 'seq' is null on {seq_col.null_count} of {n} samples")
    else:
        seq = seq_col.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return EpisodeStream(episode_index, host_ts, device_ts, seq, data)


# --- Alignment -----------------------------------------------------------------------------------


def frame_times(origin: float, n_frames: int, fps: float) -> NDArray[np.float64]:
    """Host times of ``n_frames`` dataset frames: ``origin + k / fps`` for ``k`` in ``0..n_frames-1``."""
    return origin + np.arange(n_frames, dtype=np.float64) / fps


def align_nearest(host_ts: ArrayLike, targets: ArrayLike) -> NDArray[np.int64]:
    """Return, for each target time, the index of the sample whose ``host_ts`` is nearest to it.

    ``host_ts`` must be sorted ascending. A target exactly between two samples
    takes the earlier one. Targets before the first or after the last sample take
    that sample.

    Raises:
        ValueError: If ``host_ts`` is empty or not sorted.
    """
    times = np.asarray(host_ts, dtype=np.float64)
    wanted = np.atleast_1d(np.asarray(targets, dtype=np.float64))
    if times.ndim != 1 or times.size == 0:
        raise ValueError("host_ts must be a non-empty 1-D array")
    if np.any(np.diff(times) < 0):
        raise ValueError("host_ts must be sorted ascending")
    last = times.size - 1
    right = np.clip(np.searchsorted(times, wanted, side="left"), 0, last)
    left = np.clip(right - 1, 0, last)
    take_right = np.abs(times[right] - wanted) < np.abs(wanted - times[left])
    return np.where(take_right, right, left).astype(np.int64)


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise missing_dependency("pyarrow", e) from None
    return pa, pq
