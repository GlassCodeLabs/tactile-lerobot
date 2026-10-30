"""LeRobot feature entries for tactile instances, built from ``SCHEMA.md`` metadata and read back.

A LeRobot dataset describes every column in ``meta/info.json`` with a feature
entry: ``dtype``, ``shape``, ``names`` and an optional ``info`` dict that core
itself uses for video parameters. This module builds the entries of a tactile
instance from its :class:`~taxel.schema.TactileInfo`, so the static metadata
rides in the entry's ``info`` as ``SCHEMA.md`` specifies, validates suffixed
streams against the instance, and reads entries back. It needs numpy only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from taxel.schema import (
    ARRAY_FAMILIES,
    FEATURE_KEY_PREFIX,
    STREAM_SUFFIXES,
    STREAM_WIDTHS,
    Reduction,
    SchemaError,
    StreamInfo,
    TactileInfo,
    feature_key,
)

__all__ = [
    "ARRAY_DTYPE",
    "feature_entries",
    "feature_entry",
    "read_feature_entry",
    "read_stream_entry",
    "stream_entry",
    "tactile_instances",
]

ARRAY_DTYPE = "float32"
"""dtype of every array-family feature and of every suffixed stream."""

_IMAGE_NAMES = ["height", "width", "channels"]


def feature_entry(info: TactileInfo, shape: tuple[int, ...], *, use_videos: bool = True) -> dict[str, Any]:
    """Return the feature entry of an instance's main key, ``observation.tactile.<instance_id>``.

    Array families get a float32 entry of shape ``(n_elements, len(channels))`` with
    ``names`` set to ``None``; channel names live in ``info``. A visuotactile instance
    gets a video entry, or an image entry with ``use_videos=False``, of shape
    ``(height, width, 3)`` with the names LeRobot gives cameras; core adds its own
    ``video.*`` keys to ``info`` when the first episode is encoded.

    Raises:
        SchemaError: If ``shape`` disagrees with ``info`` (wrong rank, width not equal
            to the channel count, or not three image channels).
    """
    shape = _require_shape(shape, "shape")
    if info.family in ARRAY_FAMILIES:
        assert info.channels is not None  # guaranteed by TactileInfo validation
        if len(shape) != 2 or shape[1] != len(info.channels):
            raise SchemaError(
                f"{info.family.value} frames are (n_elements, {len(info.channels)}) for channels "
                f"{info.channels}, got shape {shape}"
            )
        return {"dtype": ARRAY_DTYPE, "shape": shape, "names": None, "info": info.to_dict()}
    if len(shape) != 3 or shape[2] != 3:
        raise SchemaError(f"visuotactile frames are (height, width, 3), got shape {shape}")
    return {
        "dtype": "video" if use_videos else "image",
        "shape": shape,
        "names": list(_IMAGE_NAMES),
        "info": info.to_dict(),
    }


def stream_entry(
    info: TactileInfo, suffix: str, shape: tuple[int, ...], reduction: Reduction | None = None
) -> dict[str, Any]:
    """Return the entry of a suffixed stream, ``observation.tactile.<instance_id>.<suffix>``.

    The stream is validated against the instance through
    :meth:`TactileInfo.check_streams` (visuotactile family, ``derived`` block for
    ``.displacement`` and ``.force``). Its per-frame array is float32 ``(N, 2)``
    for ``.markers`` and ``(N, 3)`` for the other two; when the instance declares
    ``markers``, the ``.markers`` stream must have ``markers.count`` rows.

    Raises:
        SchemaError: If the stream is not allowed beside this instance or ``shape`` is wrong.
    """
    info.check_streams([suffix])
    shape = _require_shape(shape, f".{suffix} shape")
    width = STREAM_WIDTHS[suffix]
    if len(shape) != 2 or shape[1] != width:
        raise SchemaError(f".{suffix} frames are (N, {width}), got shape {shape}")
    if suffix == "markers" and info.markers is not None and shape[0] != info.markers.count:
        raise SchemaError(f".markers has {shape[0]} rows but info.markers.count is {info.markers.count}")
    stream = StreamInfo(session_meta=info.session_meta, reduction=reduction)
    return {"dtype": ARRAY_DTYPE, "shape": shape, "names": None, "info": stream.to_dict()}


def feature_entries(
    instance_id: str,
    info: TactileInfo,
    shape: tuple[int, ...],
    streams: Mapping[str, tuple[tuple[int, ...], Reduction | None]] | None = None,
    *,
    use_videos: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return every entry of one instance keyed by feature key: the main key, then its streams.

    ``streams`` maps a suffix to the stream's per-frame shape and its reduction (``None``
    when its native rate equals the dataset fps). All suffixes are checked together
    against the instance before any entry is built.
    """
    streams = dict(streams or {})
    info.check_streams(streams)
    entries = {feature_key(instance_id): feature_entry(info, shape, use_videos=use_videos)}
    for suffix, (stream_shape, reduction) in streams.items():
        entries[feature_key(instance_id, suffix)] = stream_entry(info, suffix, stream_shape, reduction)
    return entries


def read_feature_entry(entry: Mapping[str, Any]) -> TactileInfo:
    """Read an instance's static metadata back from its feature entry, strictly.

    Raises:
        SchemaError: If the entry has no ``info`` block, ``info`` is not a valid
            ``SCHEMA.md`` block (an unknown key such as a serial number is an error),
            or ``dtype`` and ``shape`` disagree with it.
    """
    info = TactileInfo.from_dict(_info_block(entry))
    dtype, shape = entry.get("dtype"), tuple(entry.get("shape") or ())
    if info.family in ARRAY_FAMILIES:
        if dtype != ARRAY_DTYPE:
            raise SchemaError(f"{info.family.value} features are {ARRAY_DTYPE}, got dtype {dtype!r}")
        feature_entry(info, shape)  # validates the shape against the channels
    elif dtype not in ("video", "image"):
        raise SchemaError(f"visuotactile features are video or image, got dtype {dtype!r}")
    return info


def read_stream_entry(entry: Mapping[str, Any]) -> StreamInfo:
    """Read a suffixed stream's ``info`` block back from its feature entry, strictly."""
    if entry.get("dtype") != ARRAY_DTYPE:
        raise SchemaError(f"suffixed streams are {ARRAY_DTYPE}, got dtype {entry.get('dtype')!r}")
    return StreamInfo.from_dict(_info_block(entry))


def tactile_instances(features: Mapping[str, Any]) -> dict[str, list[str]]:
    """Map each tactile instance in a LeRobot features dict to the suffixes declared beside it.

    Raises:
        SchemaError: If a key under the tactile prefix has an unknown suffix or a
            stream key has no instance key beside it.
    """
    mains: set[str] = set()
    suffixes: dict[str, list[str]] = {}
    for key in features:
        if not key.startswith(FEATURE_KEY_PREFIX):
            continue
        instance_id, _, suffix = key[len(FEATURE_KEY_PREFIX) :].partition(".")
        feature_key(instance_id)  # validates the slug
        if suffix:
            if suffix not in STREAM_SUFFIXES:
                raise SchemaError(f"{key}: unknown stream suffix {suffix!r}; expected {STREAM_SUFFIXES}")
            suffixes.setdefault(instance_id, []).append(suffix)
        else:
            mains.add(instance_id)
    orphans = sorted(set(suffixes) - mains)
    if orphans:
        raise SchemaError(f"stream keys without an instance key: {orphans}")
    return {instance_id: sorted(suffixes.get(instance_id, [])) for instance_id in sorted(mains)}


def _info_block(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    info = entry.get("info")
    if not isinstance(info, Mapping):
        raise SchemaError("feature entry has no 'info' block")
    return info


def _require_shape(shape: Any, path: str) -> tuple[int, ...]:
    try:
        dims = tuple(shape)
    except TypeError:
        raise SchemaError(f"{path} must be a tuple of positive integers, got {shape!r}") from None
    if not dims or any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in dims):
        raise SchemaError(f"{path} must be a tuple of positive integers, got {shape!r}")
    return dims
