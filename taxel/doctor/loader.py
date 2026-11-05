"""From a LeRobot dataset to the per-episode, per-instance inputs the health checks take.

:class:`DoctorDataset` reads ``meta/info.json``, the episodes table and the data parquet
with pyarrow, the per-instance sidecar through :mod:`taxel.dataset.sidecar`, and video
frames through LeRobot when a check asks for them. A plain LeRobot dataset with tactile
looking features and no sidecar loads too: every input that is not there is named in
``EpisodeInputs.missing`` with the reason, so a check can report ``skipped`` instead of
failing. Metadata goes through the lenient readers, and what they found unknown, missing
or unreadable is carried into :class:`~taxel.schema.InstanceReport`, never raised.
"""

from __future__ import annotations

import io
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from taxel.dataset import DOCTOR_EXTRA, LEROBOT_EXTRA, missing_dependency
from taxel.dataset.sidecar import (
    BaselineRows,
    EpisodeStream,
    baseline_path,
    episode_path,
    read_baseline_episode,
    read_episode,
)
from taxel.schema import (
    FEATURE_KEY_PREFIX,
    InstanceReport,
    Range,
    Reduction,
    SchemaError,
    SessionMeta,
    SessionRecord,
    TactileInfo,
    read_info_lenient,
    read_session_meta_lenient,
)

__all__ = ["DoctorDataset", "EpisodeInputs", "EpisodeRow", "InstanceSource", "resolve_root", "tactile_keys"]

BARE_TACTILE_KEY = FEATURE_KEY_PREFIX.rstrip(".")
"""``observation.tactile`` without an instance id, as some public datasets name their one feature."""

_IMAGE_DTYPES = ("video", "image")


# --- Discovery -----------------------------------------------------------------------------------


def tactile_keys(features: Mapping[str, Any], explicit: Iterable[str] = ()) -> dict[str, str]:
    """Map instance ids to feature keys: every tactile-looking feature, plus ``explicit`` ones.

    A feature looks tactile when its key is ``observation.tactile`` or
    ``observation.tactile.<instance_id>`` (a suffixed stream belongs to its instance and is
    not listed). An explicit entry is an instance id of one of those, or any feature key,
    whose id is then the key's last dotted part after the ``observation.images.`` or
    ``observation.tactile.`` prefix.

    Raises:
        KeyError: If an explicit entry names neither an instance nor a feature.
    """
    found: dict[str, str] = {}
    for key in features:
        if key == BARE_TACTILE_KEY:
            found["tactile"] = key
        elif key.startswith(FEATURE_KEY_PREFIX):
            instance_id, _, suffix = key[len(FEATURE_KEY_PREFIX) :].partition(".")
            if not suffix:
                found[instance_id] = key
    if not explicit:
        return found
    chosen: dict[str, str] = {}
    for name in explicit:
        if name in found:
            chosen[name] = found[name]
        elif name in features:
            chosen[_instance_id_of(name)] = name
        else:
            raise KeyError(f"{name!r} is neither a tactile instance {sorted(found)} nor a feature key")
    return chosen


def _instance_id_of(key: str) -> str:
    for prefix in (FEATURE_KEY_PREFIX, "observation.images."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def resolve_root(dataset: str | Path) -> Path:
    """A local dataset root as is; otherwise a Hub repo id, downloaded into LeRobot's cache.

    The cache is ``$HF_LEROBOT_HOME`` (default ``$HF_HOME/lerobot``, so
    ``~/.cache/huggingface/lerobot``), under the repo id, the directory
    ``LeRobotDataset`` uses too.
    """
    path = Path(dataset).expanduser()
    if path.is_dir():
        return path
    repo_id = str(dataset)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise FileNotFoundError(f"{dataset!r} is not a directory and does not look like a Hub repo id")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise missing_dependency("huggingface_hub", e, DOCTOR_EXTRA) from None
    home = os.environ.get("HF_LEROBOT_HOME")
    base = Path(home) if home else Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "lerobot"
    local = (base.expanduser() / repo_id).resolve()
    snapshot_download(repo_id, repo_type="dataset", local_dir=local)
    return local


# --- Inputs ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeRow:
    """One row of the episodes table: where the episode's frames and videos live."""

    index: int
    length: int
    start: int
    stop: int
    data_file: Path
    videos: dict[str, tuple[Path, float]]
    """Per video key: the file and the timestamp of the episode's first frame inside it."""


@dataclass
class InstanceSource:
    """One tactile instance of the dataset: its feature, its metadata and its sidecar, if any.

    ``kind`` is ``array`` for a float feature of one or two dimensions, ``image`` for a
    video or image feature. ``metadata``, ``error``, ``unknown_keys`` and ``missing_keys``
    are what the lenient readers found; see :class:`~taxel.schema.InstanceReport`.
    """

    instance_id: str
    key: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    info: TactileInfo | None
    metadata: str
    error: str | None
    sessions: SessionMeta | None
    sidecar: bool
    unknown_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)

    @property
    def family(self) -> str:
        return self.info.family.value if self.info is not None else self.kind

    @property
    def declared_rate_hz(self) -> float | None:
        return None if self.info is None else self.info.declared_rate_hz

    @property
    def range(self) -> Range | None:
        return None if self.info is None else self.info.range

    @property
    def reduction(self) -> Reduction | None:
        return None if self.info is None else self.info.reduction

    @property
    def n_channels(self) -> int:
        """Channels of an array instance: the last dimension, or 1 for a one-dimensional feature."""
        return self.shape[1] if len(self.shape) == 2 else 1

    def session_for(self, episode_index: int) -> SessionRecord | None:
        """The session record whose episode range covers ``episode_index``, or ``None``."""
        if self.sessions is None:
            return None
        for record in self.sessions.sessions:
            first, last = record.episodes
            if first <= episode_index <= last:
                return record
        return None

    def report(self) -> InstanceReport:
        return InstanceReport(
            key=self.key,
            family=self.family,
            metadata=self.metadata,
            sidecar=self.sidecar,
            declared_rate_hz=self.declared_rate_hz,
            range=self.range,
            unknown_keys=list(self.unknown_keys),
            missing_keys=list(self.missing_keys),
            error=self.error,
        )


@dataclass
class EpisodeInputs:
    """Everything the checks read for one instance in one episode.

    ``frames`` is the main-table copy as float32 ``(n, n_elements, n_channels)``, a
    one-dimensional feature reshaped to one channel; ``None`` for an image instance,
    whose frames come from :meth:`DoctorDataset.frames`. ``timestamps`` is LeRobot's
    ``timestamp`` column. The sidecar inputs are ``None`` when absent, with the reason in
    ``missing``.
    """

    episode_index: int
    instance: InstanceSource
    fps: float
    frames: NDArray[np.float32] | None
    timestamps: NDArray[np.float64]
    native: EpisodeStream | None
    baseline_rows: BaselineRows | None
    session: SessionRecord | None
    origin: float | None
    missing: list[str]

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)


# --- The dataset -----------------------------------------------------------------------------------


class DoctorDataset:
    """A LeRobot dataset opened for checking, through pyarrow and the sidecar readers.

    Args:
        root: The dataset root, holding ``meta/info.json``.

    Raises:
        FileNotFoundError: If ``meta/info.json`` is missing.
        SchemaError: If ``info.json`` has no ``features`` or no ``fps``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"{self.root} is not a LeRobot dataset: {info_path} not found")
        self.info: dict[str, Any] = json.loads(info_path.read_text())
        if "features" not in self.info or "fps" not in self.info:
            raise SchemaError(f"{info_path} has no 'features' or no 'fps'")
        self.features: dict[str, dict[str, Any]] = self.info["features"]
        self.fps = float(self.info["fps"])
        self._data_path = self.info.get(
            "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        )
        self._video_path = (
            self.info.get("video_path")
            or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        self._episodes = self._read_episodes()
        self._tables: dict[Path, Any] = {}

    # --- Episodes and features ------------------------------------------------------------------

    @property
    def episode_indices(self) -> list[int]:
        return sorted(self._episodes)

    def episode(self, index: int) -> EpisodeRow:
        try:
            return self._episodes[index]
        except KeyError:
            raise KeyError(f"no episode {index}; the dataset has {self.episode_indices}") from None

    @property
    def camera_keys(self) -> list[str]:
        """Video and image features that are not tactile instances, in feature order."""
        tactile = set(tactile_keys(self.features).values())
        return [k for k, ft in self.features.items() if ft.get("dtype") in _IMAGE_DTYPES and k not in tactile]

    def instances(self, explicit: Iterable[str] = ()) -> dict[str, InstanceSource]:
        """The tactile instances to check, by instance id. See :func:`tactile_keys`."""
        return {
            name: self._instance(name, key) for name, key in tactile_keys(self.features, explicit).items()
        }

    def _instance(self, instance_id: str, key: str) -> InstanceSource:
        entry = self.features[key]
        dtype, shape = str(entry.get("dtype")), tuple(int(d) for d in entry.get("shape") or ())
        if dtype in _IMAGE_DTYPES:
            kind = "image"
        elif dtype in ("float32", "float64") and 1 <= len(shape) <= 2:
            kind = "array"
        else:
            raise SchemaError(f"{key}: a tactile instance is a float array or an image, not {dtype} {shape}")
        info, metadata, error, unknown = self._read_info(key, entry)
        sessions, sidecar, missing, session_error = self._read_sessions(instance_id, info)
        if session_error is not None:
            error = session_error if error is None else f"{error}; {session_error}"
        return InstanceSource(
            instance_id, key, kind, dtype, shape, info, metadata, error, sessions, sidecar, unknown, missing
        )

    @staticmethod
    def _read_info(
        key: str, entry: Mapping[str, Any]
    ) -> tuple[TactileInfo | None, str, str | None, list[str]]:
        block = entry.get("info")
        own = {k: v for k, v in (block or {}).items() if not k.startswith("video.")}
        if not own:
            return None, "none", None, []
        try:
            info, unknown = read_info_lenient(block)
        except SchemaError as e:
            return None, "invalid", f"{key}: info block: {e}", []
        return info, "info", None, unknown

    def _read_sessions(
        self, instance_id: str, info: TactileInfo | None
    ) -> tuple[SessionMeta | None, bool, list[str], str | None]:
        relative = info.session_meta if info is not None else f"tactile/{instance_id}/meta.json"
        path = self.root / relative
        if not path.is_file():
            return None, False, [], None
        try:
            meta, unknown, missing = read_session_meta_lenient(json.loads(path.read_text()))
        except (SchemaError, ValueError) as e:
            return None, False, [], f"{relative}: {e}"
        return meta, True, [*unknown, *missing], None

    # --- Main table ---------------------------------------------------------------------------------

    def column(self, key: str, episode_index: int) -> NDArray[Any]:
        """One feature's values for one episode, in frame order, as a numpy array ``(n, *shape)``."""
        row = self.episode(episode_index)
        table = self._table(row.data_file)
        pa = _pyarrow()[0]
        mask = pa.compute.equal(table.column("episode_index"), episode_index)
        part = table.filter(mask)
        if len(part) != row.length:
            raise SchemaError(
                f"episode {episode_index}: the episodes table says {row.length} frames, "
                f"{row.data_file.name} holds {len(part)}"
            )
        col = part.column(key).combine_chunks()
        if isinstance(col, pa.ExtensionArray):
            col = col.storage
        if pa.types.is_struct(col.type):
            return np.array(col.to_pylist(), dtype=object)
        storage = col
        while pa.types.is_list(storage.type) or pa.types.is_fixed_size_list(storage.type):
            storage = storage.flatten()
        shape = tuple(int(d) for d in self.features[key].get("shape") or ())
        values = np.asarray(storage.to_numpy(zero_copy_only=False))
        return values.reshape(len(part), *shape) if shape else values

    def timestamps(self, episode_index: int) -> NDArray[np.float64]:
        """LeRobot's ``timestamp`` column for the episode, as float64 seconds."""
        return np.asarray(self.column("timestamp", episode_index), dtype=np.float64).reshape(-1)

    def episode_inputs(self, instance: InstanceSource, episode_index: int) -> EpisodeInputs:
        """Collect the checks' inputs for one instance in one episode; what is absent goes in ``missing``."""
        missing: list[str] = []
        timestamps = self.timestamps(episode_index)
        frames = None
        if instance.kind == "array":
            raw = np.asarray(self.column(instance.key, episode_index), dtype=np.float32)
            frames = raw.reshape(len(raw), -1, instance.n_channels)
        native = baseline_rows = None
        session = instance.session_for(episode_index)
        origin = None
        if not instance.sidecar:
            missing.append(f"no sidecar: tactile/{instance.instance_id}/meta.json not found")
        elif session is None:
            missing.append(f"no session record covers episode {episode_index}")
        else:
            if session.clock is None:
                missing.append("no clock in the session record")
            else:
                origin = float(session.clock.episode_origins[episode_index - session.episodes[0]])
        if instance.kind == "array" and instance.sidecar:
            native_file = episode_path(self.root, instance.instance_id, episode_index)
            if native_file.is_file():
                native = read_episode(native_file)
            else:
                missing.append(f"no native stream: {native_file.relative_to(self.root)} not found")
            baseline_file = baseline_path(self.root, instance.instance_id, episode_index)
            if baseline_file.is_file():
                baseline_rows = read_baseline_episode(baseline_file)
            else:
                missing.append(f"no baseline stream: {baseline_file.relative_to(self.root)} not found")
        return EpisodeInputs(
            episode_index,
            instance,
            self.fps,
            frames,
            timestamps,
            native,
            baseline_rows,
            session,
            origin,
            missing,
        )

    # --- Video ------------------------------------------------------------------------------------

    def frames(self, key: str, episode_index: int, max_frames: int | None = None) -> NDArray[np.uint8]:
        """Decoded frames of a video or image feature for the episode, uint8 ``(k, height, width, 3)``.

        With ``max_frames``, that many frames evenly spread over the episode; otherwise all.
        Video needs LeRobot's decoder (``taxel[lerobot]``); an image feature is decoded
        from the PNG bytes in the data parquet.

        Raises:
            ImportError: Naming the extra when the decoder is not installed.
        """
        row = self.episode(episode_index)
        dtype = self.features[key].get("dtype")
        if dtype not in _IMAGE_DTYPES:
            raise SchemaError(f"{key} is {dtype!r}, not a video or image feature")
        indices = np.arange(row.length)
        if max_frames is not None and max_frames < row.length:
            indices = np.unique(np.linspace(0, row.length - 1, max_frames).round().astype(int))
        if dtype == "image":
            return self._png_frames(key, episode_index, indices)
        try:
            from lerobot.datasets.video_utils import decode_video_frames
        except ImportError as e:
            raise missing_dependency("LeRobot", e, LEROBOT_EXTRA) from None
        path, from_ts = row.videos[key]
        stamps = [from_ts + int(i) / self.fps for i in indices]
        tensor = decode_video_frames(path, stamps, tolerance_s=1e-4, backend="pyav", return_uint8=True)
        return np.ascontiguousarray(tensor.numpy().transpose(0, 2, 3, 1))

    def _png_frames(self, key: str, episode_index: int, indices: NDArray[np.int64]) -> NDArray[np.uint8]:
        try:
            from PIL import Image
        except ImportError as e:
            raise missing_dependency("Pillow", e, LEROBOT_EXTRA) from None
        rows = self.column(key, episode_index)
        images = [np.asarray(Image.open(io.BytesIO(rows[int(i)]["bytes"])).convert("RGB")) for i in indices]
        return np.stack(images).astype(np.uint8)

    # --- Internals ----------------------------------------------------------------------------------

    def _read_episodes(self) -> dict[int, EpisodeRow]:
        _, pq = _pyarrow()
        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"{self.root}: no episodes table under meta/episodes/")
        video_keys = [k for k, ft in self.features.items() if ft.get("dtype") == "video"]
        rows: dict[int, EpisodeRow] = {}
        for file in files:
            table = pq.read_table(file)
            wanted = [
                "episode_index",
                "length",
                "dataset_from_index",
                "dataset_to_index",
                "data/chunk_index",
                "data/file_index",
            ]
            wanted += [
                f"videos/{k}/{c}" for k in video_keys for c in ("chunk_index", "file_index", "from_timestamp")
            ]
            present = [c for c in wanted if c in table.column_names]
            for record in table.select(present).to_pylist():
                index = int(record["episode_index"])
                data_file = self.root / self._data_path.format(
                    chunk_index=int(record["data/chunk_index"]), file_index=int(record["data/file_index"])
                )
                videos = {}
                for k in video_keys:
                    chunk, file_idx = (
                        record.get(f"videos/{k}/chunk_index"),
                        record.get(f"videos/{k}/file_index"),
                    )
                    if chunk is None or file_idx is None:
                        continue
                    path = self.root / self._video_path.format(
                        video_key=k, chunk_index=int(chunk), file_index=int(file_idx)
                    )
                    videos[k] = (path, float(record.get(f"videos/{k}/from_timestamp") or 0.0))
                rows[index] = EpisodeRow(
                    index,
                    int(record["length"]),
                    int(record["dataset_from_index"]),
                    int(record["dataset_to_index"]),
                    data_file,
                    videos,
                )
        return rows

    def _table(self, path: Path) -> Any:
        if path not in self._tables:
            _, pq = _pyarrow()
            if not path.is_file():
                raise FileNotFoundError(f"{path} is named by the episodes table but does not exist")
            self._tables[path] = pq.read_table(path)
        return self._tables[path]


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.compute  # noqa: F401  registers pa.compute
        import pyarrow.parquet as pq
    except ImportError as e:
        raise missing_dependency("pyarrow", e, DOCTOR_EXTRA) from None
    return pa, pq
