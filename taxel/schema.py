"""Dataclasses for the metadata defined in ``SCHEMA.md``.

Two documents are modelled here:

- :class:`TactileInfo`, the static ``info`` block stored in a tactile feature's
  entry in ``meta/info.json``. It holds what is true of a sensor instance in
  every recording session.
- :class:`SessionMeta`, the per-instance sidecar ``tactile/<instance_id>/meta.json``.
  It holds one :class:`SessionRecord` per recording session: serial numbers,
  calibration, baseline, replaceable parts and the episodes the session covers.

Every class validates itself on construction and offers ``to_dict`` /
``from_dict`` that round-trip to the JSON shapes in ``SCHEMA.md``. ``from_dict``
is strict: an unknown key is an error, so a typo in hand-written metadata fails
at load time rather than silently becoming an ignored field. Decoding goes
through :class:`_Reader`, which is the one place a lenient, report-unknown-keys
mode would plug in for third-party datasets.
"""

from __future__ import annotations

import re
import types
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
from itertools import pairwise
from typing import Any, ClassVar, TypeVar, Union, get_args, get_origin, get_type_hints

import numpy as np

__all__ = [
    "ARRAY_FAMILIES",
    "Adapter",
    "Baseline",
    "Block",
    "Calibration",
    "Derived",
    "DeviceGroup",
    "DeviceSession",
    "FEATURE_KEY_PREFIX",
    "Gel",
    "Layout",
    "Markers",
    "Mount",
    "REDUCTION_METHODS",
    "Reduction",
    "STREAM_SUFFIXES",
    "SchemaError",
    "SessionMeta",
    "SessionRecord",
    "Skin",
    "TactileFamily",
    "TactileInfo",
    "feature_key",
    "session_meta_path",
]

FEATURE_KEY_PREFIX = "observation.tactile."
"""Prefix of every tactile feature key: ``observation.tactile.<instance_id>``."""

STREAM_SUFFIXES = ("markers", "displacement", "force")
"""Suffixes a visuotactile instance may declare beside its image key."""

REDUCTION_METHODS = ("nearest", "mean", "last")
"""How a dataset-rate copy was reduced from a faster native stream."""

_SLUG = re.compile(r"^[A-Za-z0-9_-]+$")
_SESSION_META = re.compile(r"^tactile/[A-Za-z0-9_-]+/meta\.json$")


class SchemaError(ValueError):
    """Raised when metadata does not satisfy ``SCHEMA.md``."""


_T = TypeVar("_T", bound="Block")


class Block:
    """Mixin giving every schema dataclass ``to_dict`` and a strict ``from_dict``.

    Subclasses may set ``_label`` (the root of error paths), ``_omit_when_none``
    (keys left out of the dict when ``None``) and ``_key_order`` (the key order
    in ``SCHEMA.md``, when it differs from field order).
    """

    _label: ClassVar[str]
    _omit_when_none: ClassVar[frozenset[str]] = frozenset()
    _key_order: ClassVar[tuple[str, ...] | None] = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form, exactly as ``SCHEMA.md`` shows it."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls: type[_T], data: Mapping[str, Any]) -> _T:
        """Build from parsed JSON. Unknown keys and wrong types raise :class:`SchemaError`."""
        return _from_dict(cls, data, _Reader(), getattr(cls, "_label", cls.__name__))


class TactileFamily(str, Enum):
    """The four sensing families. Adding one is a schema change."""

    TAXEL_ARRAY = "taxel_array"
    VISUOTACTILE = "visuotactile"
    MAGNETIC = "magnetic"
    FORCE_TORQUE = "force_torque"


ARRAY_FAMILIES = frozenset({TactileFamily.TAXEL_ARRAY, TactileFamily.MAGNETIC, TactileFamily.FORCE_TORQUE})
"""Families whose per-frame data is a float32 ``(n_elements, len(channels))`` array."""


def feature_key(instance_id: str, suffix: str | None = None) -> str:
    """Return the feature key for an instance, or for one of its suffixed streams."""
    _require_slug(instance_id, "instance_id")
    if suffix is None:
        return FEATURE_KEY_PREFIX + instance_id
    if suffix not in STREAM_SUFFIXES:
        raise SchemaError(f"unknown stream suffix {suffix!r}; expected one of {STREAM_SUFFIXES}")
    return f"{FEATURE_KEY_PREFIX}{instance_id}.{suffix}"


def session_meta_path(instance_id: str) -> str:
    """Return the sidecar path for an instance, relative to the dataset root."""
    _require_slug(instance_id, "instance_id")
    return f"tactile/{instance_id}/meta.json"


# --- Static metadata: the feature entry's ``info`` block -----------------------------------------


@dataclass
class Layout(Block):
    """Physical position of each element in the sensor's own frame.

    ``positions_mm`` has one ``[x, y, z]`` entry per element, in millimetres.
    ``normals`` is either ``None`` or one unit vector per element.
    """

    frame: str
    positions_mm: list[list[float]]
    normals: list[list[float]] | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.frame, "layout.frame")
        _require_rows(self.positions_mm, 3, "layout.positions_mm")
        if self.normals is not None:
            _require_rows(self.normals, 3, "layout.normals")
            if len(self.normals) != len(self.positions_mm):
                raise SchemaError(
                    f"layout.normals has {len(self.normals)} rows, positions_mm has {len(self.positions_mm)}"
                )


@dataclass
class Reduction(Block):
    """How a dataset-rate copy was made from a faster native stream."""

    method: str
    window_ms: float

    def __post_init__(self) -> None:
        if self.method not in REDUCTION_METHODS:
            raise SchemaError(f"reduction.method must be one of {REDUCTION_METHODS}, got {self.method!r}")
        _require_positive(self.window_ms, "reduction.window_ms")


@dataclass
class Markers(Block):
    """Marker arrangement of a visuotactile sensor."""

    count: int
    grid: list[int]

    def __post_init__(self) -> None:
        _require_positive(self.count, "markers.count")
        if len(self.grid) != 2 or any(not isinstance(g, int) or g <= 0 for g in self.grid):
            raise SchemaError(f"markers.grid must be two positive integers, got {self.grid!r}")


@dataclass
class Derived(Block):
    """The algorithm that produced a ``.displacement`` or ``.force`` stream."""

    method: str
    version: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.method, "derived.method")
        _require_nonempty_str(self.version, "derived.version")


@dataclass
class Mount(Block):
    """Where on the robot the sensor sits. ``pose`` is in the link frame, or ``None``."""

    body: str
    link: str
    pose: list[Any] | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.body, "mount.body")
        _require_nonempty_str(self.link, "mount.link")
        if self.pose is not None and not isinstance(self.pose, list):
            raise SchemaError(f"mount.pose must be a list or null, got {type(self.pose).__name__}")


@dataclass
class DeviceGroup(Block):
    """The physical connection an instance shares with others, such as a serial hub.

    Instances with the same ``clock`` have directly comparable ``device_ts``.
    """

    id: str
    clock: str

    def __post_init__(self) -> None:
        _require_slug(self.id, "device.id")
        _require_slug(self.clock, "device.clock")


@dataclass
class TactileInfo(Block):
    """Static metadata of one sensor instance: the ``info`` block of its feature entry.

    Only facts that hold in every session belong here. LeRobot's ``merge``
    requires feature entries to be identical, so anything that varies between
    sessions goes in :class:`SessionRecord` instead.

    ``family``, ``vendor``, ``model``, ``declared_rate_hz``, ``mount``, ``device``
    and ``session_meta`` are always written. ``units``, ``channels``, ``layout``,
    ``reduction``, ``markers`` and ``derived`` are written only when set.
    """

    family: TactileFamily
    declared_rate_hz: float
    session_meta: str
    vendor: str | None = None
    model: str | None = None
    units: dict[str, str] | None = None
    channels: list[str] | None = None
    layout: Layout | None = None
    reduction: Reduction | None = None
    markers: Markers | None = None
    derived: Derived | None = None
    mount: Mount | None = None
    device: DeviceGroup | None = None

    _label: ClassVar[str] = "info"
    _omit_when_none: ClassVar[frozenset[str]] = frozenset(
        {"units", "channels", "layout", "reduction", "markers", "derived"}
    )
    _key_order: ClassVar[tuple[str, ...] | None] = (
        "family",
        "vendor",
        "model",
        "declared_rate_hz",
        "units",
        "channels",
        "layout",
        "reduction",
        "markers",
        "derived",
        "mount",
        "device",
        "session_meta",
    )

    def __post_init__(self) -> None:
        self.family = _enum(TactileFamily, self.family, "family")
        _require_positive(self.declared_rate_hz, "declared_rate_hz")
        if not isinstance(self.session_meta, str) or not _SESSION_META.match(self.session_meta):
            raise SchemaError(
                f"session_meta must look like 'tactile/<instance_id>/meta.json', got {self.session_meta!r}"
            )
        if self.family in ARRAY_FAMILIES:
            if not self.channels:
                raise SchemaError(f"channels is required for family {self.family.value!r}")
            if any(not isinstance(c, str) or not c for c in self.channels):
                raise SchemaError("channels must be non-empty strings")
            if len(set(self.channels)) != len(self.channels):
                raise SchemaError(f"channels must be unique, got {self.channels!r}")
            if self.family is TactileFamily.FORCE_TORQUE and len(self.channels) != 6:
                raise SchemaError(f"force_torque declares six channels, got {len(self.channels)}")
        if self.units is not None and any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in self.units.items()
        ):
            raise SchemaError("units must map quantity names to unit strings")

    @property
    def n_channels(self) -> int | None:
        """Last dimension of a per-frame array, or ``None`` for an image stream."""
        return None if self.channels is None else len(self.channels)

    def check_streams(self, suffixes: Iterable[str]) -> None:
        """Validate the suffixed streams declared beside this instance's key.

        ``.displacement`` and ``.force`` require a ``derived`` block, and any
        suffixed stream requires the visuotactile family.
        """
        declared = set(suffixes)
        unknown = declared - set(STREAM_SUFFIXES)
        if unknown:
            raise SchemaError(f"unknown stream suffixes {sorted(unknown)}; expected {STREAM_SUFFIXES}")
        if declared and self.family is not TactileFamily.VISUOTACTILE:
            raise SchemaError(
                f"suffixed streams are defined for visuotactile instances only, not {self.family.value!r}"
            )
        if declared & {"displacement", "force"} and self.derived is None:
            raise SchemaError("'.displacement' and '.force' streams require a 'derived' block")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TactileInfo:
        """Build from an ``info`` block. Core's own ``video.*`` keys are ignored."""
        own = {k: v for k, v in data.items() if not k.startswith("video.")}
        return super().from_dict(own)


# --- Per-session metadata: ``tactile/<instance_id>/meta.json`` ---------------------------------


@dataclass
class DeviceSession(Block):
    """Serial and firmware of the shared hub during one session."""

    serial: str | None = None
    firmware: str | None = None


@dataclass
class Adapter(Block):
    """The code that turned the sensor's output into the stored values."""

    name: str
    version: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "adapter.name")
        _require_nonempty_str(self.version, "adapter.version")


@dataclass
class Calibration(Block):
    """How raw readings were converted to physical units. ``method`` is free text for now."""

    version: str | None = None
    method: str | None = None
    date: str | None = None
    reference: str | None = None


@dataclass
class Baseline(Block):
    """What the sensor read with nothing touching it at session start.

    Array families store the reading as ``values``; visuotactile sensors store a
    ``reference_frame`` pointer. Exactly one of the two is set. ``applied`` says
    whether ``values`` were subtracted at record time, which is only valid when
    they are stored, so stored data plus ``values`` reconstructs the raw readings.
    """

    captured_at: str
    duration_s: float
    applied: bool
    values: list[list[float]] | None = None
    reference_frame: str | None = None

    _omit_when_none: ClassVar[frozenset[str]] = frozenset({"values", "reference_frame"})

    def __post_init__(self) -> None:
        _require_nonempty_str(self.captured_at, "baseline.captured_at")
        _require_positive(self.duration_s, "baseline.duration_s")
        if not isinstance(self.applied, bool):
            raise SchemaError(f"baseline.applied must be a boolean, got {self.applied!r}")
        if (self.values is None) == (self.reference_frame is None):
            raise SchemaError("baseline needs exactly one of 'values' or 'reference_frame'")
        if self.values is not None:
            _require_rows(self.values, None, "baseline.values")
        else:
            _require_nonempty_str(self.reference_frame, "baseline.reference_frame")
        if self.applied and self.values is None:
            raise SchemaError("baseline.applied is true but no values are stored")


@dataclass
class Gel(Block):
    """Replaceable gel of a visuotactile sensor."""

    id: str
    installed_at: str | None = None
    cycles_at_install: int | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.id, "gel.id")


@dataclass
class Skin(Block):
    """Replaceable skin of a magnetic or taxel-array sensor."""

    id: str
    installed_at: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.id, "skin.id")


@dataclass
class SessionRecord(Block):
    """One recording session of one instance.

    ``adapter``, ``calibration`` and ``baseline`` have no default: they are
    always written, as ``null`` when unknown, so their absence is a visible fact.
    ``episodes`` is the inclusive range of episode indices the session covers.
    """

    session_id: str
    episodes: tuple[int, int]
    adapter: Adapter | None
    calibration: Calibration | None
    baseline: Baseline | None
    serial: str | None = None
    firmware: str | None = None
    device: DeviceSession | None = None
    gel: Gel | None = None
    skin: Skin | None = None

    _label: ClassVar[str] = "session"
    _key_order: ClassVar[tuple[str, ...] | None] = (
        "session_id",
        "episodes",
        "serial",
        "firmware",
        "device",
        "adapter",
        "calibration",
        "baseline",
        "gel",
        "skin",
    )

    def __post_init__(self) -> None:
        _require_nonempty_str(self.session_id, "session_id")
        self.episodes = tuple(self.episodes)  # type: ignore[assignment]
        if len(self.episodes) != 2 or any(
            not isinstance(e, int) or isinstance(e, bool) for e in self.episodes
        ):
            raise SchemaError(f"episodes must be two integers [first, last], got {self.episodes!r}")
        first, last = self.episodes
        if first < 0 or last < first:
            raise SchemaError(f"episodes must satisfy 0 <= first <= last, got {list(self.episodes)}")


@dataclass
class SessionMeta(Block):
    """The whole sidecar file ``tactile/<instance_id>/meta.json``."""

    instance_id: str
    sessions: list[SessionRecord]

    _label: ClassVar[str] = "meta"

    def __post_init__(self) -> None:
        _require_slug(self.instance_id, "instance_id")
        ranges = sorted(s.episodes for s in self.sessions)
        for (_, prev_last), (first, _) in pairwise(ranges):
            if first <= prev_last:
                raise SchemaError(f"session episode ranges overlap: {ranges}")


# --- Encoding and decoding -----------------------------------------------------------------------


class _Reader:
    """Decoding policy. Strict today; a lenient mode would record unknown keys instead of raising."""

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict
        self.unknown: list[str] = []

    def check_keys(self, data: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
        extra = sorted(set(data) - set(allowed))
        if not extra:
            return
        if self.strict:
            raise SchemaError(f"{path}: unknown keys {extra}")
        self.unknown.extend(f"{path}.{key}" for key in extra)


def _to_dict(obj: Any) -> dict[str, Any]:
    cls = type(obj)
    omit = getattr(cls, "_omit_when_none", frozenset())
    order = getattr(cls, "_key_order", None) or tuple(f.name for f in fields(obj))
    out: dict[str, Any] = {}
    for name in order:
        value = getattr(obj, name)
        if value is None and name in omit:
            continue
        out[name] = _encode(value)
    return out


def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_dict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


def _from_dict(cls: type, data: Any, reader: _Reader, path: str) -> Any:
    if not isinstance(data, Mapping):
        raise SchemaError(f"{path}: expected an object, got {type(data).__name__}")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    reader.check_keys(data, names, path)
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in data:
            if field.default is MISSING and field.default_factory is MISSING:
                raise SchemaError(f"{path}: missing required key {field.name!r}")
            continue
        kwargs[field.name] = _decode(hints[field.name], data[field.name], reader, f"{path}.{field.name}")
    try:
        return cls(**kwargs)
    except SchemaError as e:
        raise SchemaError(f"{path}: {e}") from None


def _decode(hint: Any, value: Any, reader: _Reader, path: str) -> Any:
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        members = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            if len(members) == len(get_args(hint)):
                raise SchemaError(f"{path}: null is not allowed")
            return None
        return _decode(members[0], value, reader, path)
    if is_dataclass(hint):
        return _from_dict(hint, value, reader, path)
    if isinstance(hint, type) and issubclass(hint, Enum):
        return _enum(hint, value, path)
    if origin is list:
        if not isinstance(value, list):
            raise SchemaError(f"{path}: expected a list, got {type(value).__name__}")
        (item,) = get_args(hint)
        return [_decode(item, v, reader, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise SchemaError(f"{path}: expected a list, got {type(value).__name__}")
        items = get_args(hint)
        if len(value) != len(items):
            raise SchemaError(f"{path}: expected {len(items)} items, got {len(value)}")
        return tuple(
            _decode(t, v, reader, f"{path}[{i}]") for i, (t, v) in enumerate(zip(items, value, strict=True))
        )
    if origin is dict:
        if not isinstance(value, Mapping):
            raise SchemaError(f"{path}: expected an object, got {type(value).__name__}")
        key_type, value_type = get_args(hint)
        return {
            _decode(key_type, k, reader, path): _decode(value_type, v, reader, f"{path}.{k}")
            for k, v in value.items()
        }
    if hint is Any:
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise SchemaError(f"{path}: expected a boolean, got {value!r}")
        return value
    if hint is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaError(f"{path}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SchemaError(f"{path}: expected a number, got {value!r}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise SchemaError(f"{path}: expected a string, got {value!r}")
        return value
    raise TypeError(f"{path}: no decoder for type hint {hint!r}")


def _enum(cls: type[Enum], value: Any, path: str) -> Any:
    try:
        return cls(value)
    except ValueError:
        raise SchemaError(f"{path} must be one of {[m.value for m in cls]}, got {value!r}") from None


def _require_nonempty_str(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{path} must be a non-empty string, got {value!r}")


def _require_slug(value: Any, path: str) -> None:
    if not isinstance(value, str) or not _SLUG.match(value):
        raise SchemaError(f"{path} must match [A-Za-z0-9_-]+, got {value!r}")


def _require_positive(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise SchemaError(f"{path} must be a positive number, got {value!r}")


def _require_rows(value: Any, width: int | None, path: str) -> None:
    """Require a non-empty, rectangular list of numeric rows, optionally of a fixed width."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise SchemaError(f"{path} must be a rectangular list of numbers") from None
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise SchemaError(f"{path} must be a non-empty 2-D list of numbers, got shape {array.shape}")
    if width is not None and array.shape[1] != width:
        raise SchemaError(f"{path} rows must have {width} values, got {array.shape[1]}")
