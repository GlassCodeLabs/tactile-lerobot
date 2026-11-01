"""The per-instance sidecar: ``tactile/<instance_id>/meta.json``, the native-rate episode files and the
baseline stream.

``meta.json`` is a :class:`~taxel.schema.SessionMeta`. Each
``episode-{index:06d}.parquet`` holds every frame the sensor delivered during one
episode with the five columns of ``SCHEMA.md`` plus ``sample_index``:
``episode_index``, ``sample_index``, ``host_ts``, ``device_ts``, ``seq`` and
``data``. ``data`` is an Arrow fixed-shape tensor column, so pyarrow reads it back
as one ``(n_samples, *shape)`` array. A suffixed stream keeps the same files in
its own directory below the instance's.

``baseline/episode-{index:06d}.parquet`` is the episode's baseline stream: a
``capture`` row for every baseline subtracted from the stored frames, from the
native sample it applies to, and a ``sample`` row with the unsubtracted frame
every ``baseline_stream.sample_every`` native samples. :func:`raw_frames` applies
the rule of ``SCHEMA.md`` to get the raw readings back.

pyarrow is imported where it is used, so the alignment helpers and the paths work
with numpy alone.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from taxel.dataset import missing_dependency
from taxel.schema import SchemaError, SessionMeta, feature_key, session_meta_path

if TYPE_CHECKING:
    from taxel.sensors.base import TactileReading

__all__ = [
    "BASELINE_COLUMNS",
    "BASELINE_KINDS",
    "BaselineRow",
    "BaselineRows",
    "COLUMNS",
    "EpisodeStream",
    "align_nearest",
    "baseline_path",
    "episode_path",
    "frame_times",
    "meta_path",
    "raw_frames",
    "read_baseline_episode",
    "read_episode",
    "read_session_meta",
    "sidecar_dir",
    "write_baseline_episode",
    "write_episode",
    "write_session_meta",
]

SIDECAR_DIR = "tactile"
BASELINE_DIR = "baseline"
COLUMNS = ("episode_index", "sample_index", "host_ts", "device_ts", "seq", "data")
BASELINE_COLUMNS = ("episode_index", "sample_index", "kind", "seq", "host_ts", "frame_range", "values")
BASELINE_KINDS = ("capture", "sample")
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
    return sidecar_dir(root, instance_id, suffix) / _episode_file(episode_index)


def baseline_path(root: str | Path, instance_id: str, episode_index: int) -> Path:
    """``<root>/tactile/<instance_id>/baseline/episode-{index:06d}.parquet``."""
    return sidecar_dir(root, instance_id) / BASELINE_DIR / _episode_file(episode_index)


def _episode_file(episode_index: int) -> str:
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError(f"episode_index must be a non-negative integer, got {episode_index!r}")
    return _EPISODE_FILE.format(index=episode_index)


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


# --- Baseline stream -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineRow:
    """One row of the baseline stream.

    A ``capture`` row is a baseline subtracted from the stored frames from ``sample_index``
    on: ``host_ts`` is when it took effect, ``frame_range`` the ``seq`` of the frames it
    was averaged from (or ``None``), ``values`` the baseline in raw terms. A ``sample``
    row is the unsubtracted frame at ``sample_index``, with that frame's ``seq`` and
    ``host_ts`` and no ``frame_range``.
    """

    kind: str
    sample_index: int
    seq: int | None
    host_ts: float
    frame_range: tuple[int, int] | None
    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.kind not in BASELINE_KINDS:
            raise ValueError(f"kind must be one of {BASELINE_KINDS}, got {self.kind!r}")
        if self.kind == "sample" and self.frame_range is not None:
            raise ValueError("a sample row has no frame_range")


@dataclass(frozen=True)
class BaselineRows:
    """The baseline stream of one instance in one episode, as parallel arrays in ``sample_index`` order.

    Attributes:
        episode_index: The episode the rows belong to.
        kind: ``"capture"`` or ``"sample"`` per row, shape ``(n,)``.
        sample_index: The native sample each row refers to, shape ``(n,)``.
        seq: That sample's ``seq``, shape ``(n,)``, or ``None`` when the sensor has none.
        host_ts: Per row, shape ``(n,)``: when a capture took effect, or a sample's receive time.
        frame_range: Per row, ``(first_seq, last_seq)`` or ``None``.
        values: Float32 ``(n, *shape)``: the baseline of a capture row, the raw frame of a sample row.
    """

    episode_index: int
    kind: NDArray[np.str_]
    sample_index: NDArray[np.int64]
    seq: NDArray[np.int64] | None
    host_ts: NDArray[np.float64]
    frame_range: list[tuple[int, int] | None]
    values: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.sample_index)

    @property
    def shape(self) -> tuple[int, ...]:
        """Per-frame shape."""
        return self.values.shape[1:]

    @property
    def captures(self) -> NDArray[np.bool_]:
        """Mask of the capture rows."""
        return self.kind == "capture"

    @property
    def samples(self) -> NDArray[np.bool_]:
        """Mask of the sample rows."""
        return self.kind == "sample"


def write_baseline_episode(
    path: str | Path, episode_index: int, rows: Sequence[BaselineRow], shape: tuple[int, ...]
) -> Path:
    """Write one episode's baseline stream, rows in ``sample_index`` order, and return the file path.

    Every row's ``values`` must be float32 of ``shape``; ``shape`` lets an episode with no
    rows still write a typed file. ``sample_index`` must not decrease, and at one index a
    capture row comes before the sample row, so a reader meets the baseline before the
    frame it applies to.

    Raises:
        ValueError: If a row has the wrong dtype or shape, or the rows are out of order.
    """
    pa, pq = _pyarrow()
    path = Path(path)
    shape = tuple(int(d) for d in shape)
    n = len(rows)
    values = np.empty((n, *shape), dtype=np.float32)
    for i, row in enumerate(rows):
        if row.values.dtype != np.float32 or row.values.shape != shape:
            raise ValueError(f"row {i}: expected float32 {shape}, got {row.values.dtype} {row.values.shape}")
        values[i] = row.values
    for i, (before, after) in enumerate(pairwise(rows)):
        if after.sample_index < before.sample_index:
            raise ValueError(f"row {i + 1}: rows must be in sample_index order")
        if after.sample_index == before.sample_index and (before.kind, after.kind) == ("sample", "capture"):
            raise ValueError(f"row {i + 1}: at one sample_index the capture row comes before the sample row")

    tensor_type = pa.fixed_shape_tensor(pa.float32(), list(shape))
    storage = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), int(np.prod(shape)))
    table = pa.table(
        {
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
            "sample_index": pa.array(np.array([r.sample_index for r in rows], dtype=np.int64)),
            "kind": pa.array([r.kind for r in rows], type=pa.string()),
            "seq": pa.array([r.seq for r in rows], type=pa.int64()),
            "host_ts": pa.array(np.array([r.host_ts for r in rows], dtype=np.float64)),
            "frame_range": pa.array(
                [None if r.frame_range is None else list(r.frame_range) for r in rows],
                type=pa.list_(pa.int64(), 2),
            ),
            "values": pa.ExtensionArray.from_storage(tensor_type, storage),
        }
    ).replace_schema_metadata({_EPISODE_INDEX_META: str(episode_index).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    return path


def read_baseline_episode(path: str | Path) -> BaselineRows:
    """Read one episode's baseline stream back.

    Raises:
        SchemaError: If a column is missing, ``values`` is not a fixed-shape tensor column,
            a ``kind`` is unknown, or ``seq`` is set on some rows and missing on others.
    """
    pa, pq = _pyarrow()
    path = Path(path)
    table = pq.read_table(path)
    missing = [c for c in BASELINE_COLUMNS if c not in table.column_names]
    if missing:
        raise SchemaError(f"{path}: missing columns {missing}")
    values_col = table.column("values").combine_chunks()
    if not isinstance(values_col.type, pa.FixedShapeTensorType):
        raise SchemaError(f"{path}: 'values' must be a fixed-shape tensor column, got {values_col.type}")
    shape = tuple(values_col.type.shape)
    n = len(table)
    if n:
        values = values_col.to_numpy_ndarray().astype(np.float32, copy=False)
        episode_index = int(table.column("episode_index")[0].as_py())
    else:
        values = np.empty((0, *shape), dtype=np.float32)
        meta = (table.schema.metadata or {}).get(_EPISODE_INDEX_META)
        if meta is None:
            raise SchemaError(f"{path}: empty baseline file without an episode index")
        episode_index = int(meta.decode())
    kinds = table.column("kind").to_pylist()
    unknown = sorted({k for k in kinds if k not in BASELINE_KINDS})
    if unknown:
        raise SchemaError(f"{path}: unknown row kinds {unknown}; expected {BASELINE_KINDS}")
    kind = np.array(kinds, dtype=str)
    sample_index = table.column("sample_index").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    seq_col = table.column("seq")
    if seq_col.null_count == n:
        seq = None
    elif seq_col.null_count:
        raise SchemaError(f"{path}: 'seq' is null on {seq_col.null_count} of {n} rows")
    else:
        seq = seq_col.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    host_ts = table.column("host_ts").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    frame_range = [
        None if v is None else (int(v[0]), int(v[1])) for v in table.column("frame_range").to_pylist()
    ]
    return BaselineRows(episode_index, kind, sample_index, seq, host_ts, frame_range, values)


def raw_frames(native: EpisodeStream, baseline: BaselineRows) -> NDArray[np.float32]:
    """Reconstruct the raw frames of an episode: stored data plus the baseline in force at each sample.

    The baseline in force at native sample ``i`` is the ``values`` of the latest capture
    row with ``sample_index <= i``; samples before the first capture row are stored raw.
    The sum is float32, as a sample row's ``values`` are, so the reconstruction at a
    sampled index equals the sample row exactly and the sensor's reading to within
    float32 rounding (``SCHEMA.md``, Baseline stream).

    Raises:
        ValueError: If the two belong to different episodes or shapes, or a capture row
            points outside the native stream.
    """
    if native.episode_index != baseline.episode_index:
        raise ValueError(
            f"episode {native.episode_index} native stream, episode {baseline.episode_index} baseline"
        )
    if native.shape != baseline.shape:
        raise ValueError(f"native frames are {native.shape}, baseline values are {baseline.shape}")
    raw = native.data.astype(np.float32, copy=True)
    starts = baseline.sample_index[baseline.captures]
    values = baseline.values[baseline.captures]
    if starts.size and (int(starts.min()) < 0 or int(starts.max()) >= len(native)):
        raise ValueError(f"a capture row points outside the {len(native)} native samples")
    for i, (start, value) in enumerate(zip(starts, values, strict=True)):
        end = int(starts[i + 1]) if i + 1 < len(starts) else len(native)
        raw[start:end] += value
    return raw


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
