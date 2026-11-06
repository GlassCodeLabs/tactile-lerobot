"""Dataclasses for the metadata defined in ``SCHEMA.md``.

Two documents are modelled here:

- :class:`TactileInfo`, the static ``info`` block stored in a tactile feature's
  entry in ``meta/info.json``. It holds what is true of a sensor instance in
  every recording session.
- :class:`SessionMeta`, the per-instance sidecar ``tactile/<instance_id>/meta.json``.
  It holds one :class:`SessionRecord` per recording session: serial numbers,
  calibration, baseline, replaceable parts and the episodes the session covers.
- :class:`HealthReport`, the health annotations ``taxel doctor`` writes to
  ``tactile/health.json``: one block of checks per episode per instance, with the
  thresholds the run used.

Every class validates itself on construction and offers ``to_dict`` /
``from_dict`` that round-trip to the JSON shapes in ``SCHEMA.md``. ``from_dict``
is strict: an unknown key is an error, so a typo in hand-written metadata fails
at load time rather than silently becoming an ignored field. Decoding goes
through :class:`_Reader`; :func:`read_info_lenient` and
:func:`read_session_meta_lenient` use its lenient mode, which reports unknown keys
and the keys a later schema version added instead of raising, for datasets this
package did not write or wrote under an earlier version.
"""

from __future__ import annotations

import re
import types
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar, TypeVar, Union, get_args, get_origin, get_type_hints

import numpy as np

__all__ = [
    "ARRAY_FAMILIES",
    "Adapter",
    "Axes",
    "Baseline",
    "BaselineStream",
    "Block",
    "Calibration",
    "CHECK_NAMES",
    "Clock",
    "CopyCheck",
    "Derived",
    "DeviceGroup",
    "DeviceSession",
    "DriftReport",
    "FEATURE_KEY_PREFIX",
    "FilesCheck",
    "FrameCounts",
    "FramerateReport",
    "GapHistogram",
    "Gel",
    "GelReport",
    "HealthReport",
    "InstanceChecks",
    "InstanceReport",
    "IntegrityReport",
    "Layout",
    "Markers",
    "Mount",
    "ORIGIN_AT",
    "Origin",
    "REDUCTION_METHODS",
    "Range",
    "RawCheck",
    "RebaselineEvent",
    "Reduction",
    "STREAM_SUFFIXES",
    "STREAM_WIDTHS",
    "SURFACES",
    "SaturationReport",
    "SchemaError",
    "SessionMeta",
    "SessionRecord",
    "SkewReport",
    "Skin",
    "Skipped",
    "StreamInfo",
    "TactileFamily",
    "TactileInfo",
    "Thresholds",
    "feature_key",
    "health_report_path",
    "read_info_lenient",
    "read_session_meta_lenient",
    "session_meta_path",
]

FEATURE_KEY_PREFIX = "observation.tactile."
"""Prefix of every tactile feature key: ``observation.tactile.<instance_id>``."""

STREAM_SUFFIXES = ("markers", "displacement", "force")
"""Suffixes a visuotactile instance may declare beside its image key."""

STREAM_WIDTHS = {"markers": 2, "displacement": 3, "force": 3}
"""Last dimension of a suffixed stream's per-frame array: pixel coordinates, or dx dy dz / fx fy fz."""

REDUCTION_METHODS = ("nearest", "mean", "last")
"""How a dataset-rate copy was reduced from a faster native stream."""

ORIGIN_AT = ("center", "corner", "element", "mount")
"""Where a layout's origin sits on the sensor body."""

SURFACES = ("gel", "die", "skin", "pad", "vendor")
"""Which surface a layout's frame and positions describe."""

HEALTH_REPORT_PATH = "tactile/health.json"
"""Where ``taxel doctor`` writes its report, relative to the dataset root."""

CHECK_NAMES = ("integrity", "drift", "framerate", "saturation", "gel", "skew")
"""The checks a health report carries per episode per instance, in report order."""

_SLUG = re.compile(r"^[A-Za-z0-9_-]+$")
_SESSION_META = re.compile(r"^tactile/[A-Za-z0-9_-]+/meta\.json$")
_WALL_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


class SchemaError(ValueError):
    """Raised when metadata does not satisfy ``SCHEMA.md``."""


_T = TypeVar("_T", bound="Block")


class Block:
    """Mixin giving every schema dataclass ``to_dict`` and a strict ``from_dict``.

    Subclasses may set ``_label`` (the root of error paths), ``_omit_when_none``
    (keys left out of the dict when ``None``), ``_key_order`` (the key order in
    ``SCHEMA.md``, when it differs from field order) and ``_lenient_missing``
    (required keys a lenient read fills with ``None`` when absent, for documents
    written under an earlier schema version).
    """

    _label: ClassVar[str]
    _omit_when_none: ClassVar[frozenset[str]] = frozenset()
    _key_order: ClassVar[tuple[str, ...] | None] = None
    _lenient_missing: ClassVar[frozenset[str]] = frozenset()

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
class Origin(Block):
    """Where (0, 0, 0) of the sensor's frame sits on its body.

    ``at`` is one of :data:`ORIGIN_AT`; ``description`` says which corner or element in words.
    """

    at: str
    description: str

    def __post_init__(self) -> None:
        if self.at not in ORIGIN_AT:
            raise SchemaError(f"origin.at must be one of {ORIGIN_AT}, got {self.at!r}")
        _require_nonempty_str(self.description, "origin.description")


@dataclass
class Axes(Block):
    """Where +x, +y and +z point, in the sensor's own terms. The frame is right-handed."""

    x: str
    y: str
    z: str

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            _require_nonempty_str(getattr(self, name), f"axes.{name}")


@dataclass
class Layout(Block):
    """The sensor's coordinate frame and, for the array families, the position of each element in it.

    ``origin``, ``axes`` and ``surface`` (one of :data:`SURFACES`) make the frame
    explicit, so two sensors' positions are comparable. ``positions_mm`` has one
    ``[x, y, z]`` row per element, in millimetres, and is omitted for an image
    sensor, whose image is the element arrangement. ``normals`` is ``None`` or one
    unit vector per position, and is written only beside ``positions_mm``.
    """

    origin: Origin
    axes: Axes
    surface: str
    positions_mm: list[list[float]] | None = None
    normals: list[list[float]] | None = None

    _omit_when_none: ClassVar[frozenset[str]] = frozenset({"positions_mm"})

    def __post_init__(self) -> None:
        if self.surface not in SURFACES:
            raise SchemaError(f"layout.surface must be one of {SURFACES}, got {self.surface!r}")
        if self.positions_mm is not None:
            _require_rows(self.positions_mm, 3, "layout.positions_mm")
        if self.normals is not None:
            if self.positions_mm is None:
                raise SchemaError("layout.normals is given without positions_mm")
            _require_rows(self.normals, 3, "layout.normals")
            if len(self.normals) != len(self.positions_mm):
                raise SchemaError(
                    f"layout.normals has {len(self.normals)} rows, positions_mm has {len(self.positions_mm)}"
                )

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        if self.positions_mm is None:
            del out["normals"]
        return out


@dataclass
class Range(Block):
    """The rated range of the readings, in ``units``: what a reading cannot exceed.

    ``min`` and ``max`` are numbers for the whole instance or lists with one value per
    channel. A reading at or beyond a limit is saturated. Without a range, a health check
    can only use the observed extremes.
    """

    min: float | list[float]
    max: float | list[float]

    def __post_init__(self) -> None:
        for name in ("min", "max"):
            value = getattr(self, name)
            ok = _is_number(value) or (
                isinstance(value, list) and value and all(_is_number(v) for v in value)
            )
            if not ok:
                raise SchemaError(f"range.{name} must be a number or a list of numbers, got {value!r}")
        lo, hi = np.asarray(self.min, dtype=np.float64), np.asarray(self.max, dtype=np.float64)
        if lo.ndim == 1 and hi.ndim == 1 and lo.shape != hi.shape:
            raise SchemaError(f"range.min has {lo.size} values and range.max {hi.size}")
        if not np.all(lo < hi):
            raise SchemaError("range.min must be below range.max on every channel")

    @property
    def n_channels(self) -> int | None:
        """Length of the per-channel lists, or ``None`` when both limits are scalars."""
        sizes = [len(v) for v in (self.min, self.max) if isinstance(v, list)]
        return max(sizes) if sizes else None

    def limits(self, n_channels: int) -> tuple[np.ndarray, np.ndarray]:
        """The ``(min, max)`` limits as float64 arrays of ``n_channels`` values each."""
        lo = np.broadcast_to(np.asarray(self.min, dtype=np.float64), (n_channels,))
        hi = np.broadcast_to(np.asarray(self.max, dtype=np.float64), (n_channels,))
        return lo.copy(), hi.copy()


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

    ``family``, ``vendor``, ``model``, ``declared_rate_hz``, ``layout``, ``mount``,
    ``device`` and ``session_meta`` are always written. ``units``, ``channels``,
    ``range``, ``reduction``, ``markers`` and ``derived`` are written only when set. ``layout``
    is required for every family; its ``positions_mm`` is required for
    ``taxel_array`` and ``magnetic``, at most one row for ``force_torque``, and
    omitted for ``visuotactile``.
    """

    family: TactileFamily
    declared_rate_hz: float
    session_meta: str
    vendor: str | None = None
    model: str | None = None
    units: dict[str, str] | None = None
    channels: list[str] | None = None
    range: Range | None = None
    layout: Layout | None = None
    reduction: Reduction | None = None
    markers: Markers | None = None
    derived: Derived | None = None
    mount: Mount | None = None
    device: DeviceGroup | None = None

    _label: ClassVar[str] = "info"
    _omit_when_none: ClassVar[frozenset[str]] = frozenset(
        {"units", "channels", "range", "reduction", "markers", "derived"}
    )
    _key_order: ClassVar[tuple[str, ...] | None] = (
        "family",
        "vendor",
        "model",
        "declared_rate_hz",
        "units",
        "channels",
        "range",
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
        _require_session_meta(self.session_meta)
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
        if self.range is not None:
            if self.family not in ARRAY_FAMILIES:
                raise SchemaError(f"range is defined for the array families, not {self.family.value!r}")
            per_channel = self.range.n_channels
            if per_channel is not None and per_channel != len(self.channels or ()):
                raise SchemaError(f"range lists {per_channel} values for {len(self.channels or ())} channels")
        if self.layout is None:
            raise SchemaError("layout is required: origin, axes and surface declare the sensor's frame")
        positions = self.layout.positions_mm
        if self.family in (TactileFamily.TAXEL_ARRAY, TactileFamily.MAGNETIC) and positions is None:
            raise SchemaError(f"layout.positions_mm is required for family {self.family.value!r}")
        if self.family is TactileFamily.FORCE_TORQUE and positions is not None and len(positions) != 1:
            raise SchemaError(f"force_torque is one element; layout.positions_mm has {len(positions)} rows")
        if self.family is TactileFamily.VISUOTACTILE and positions is not None:
            raise SchemaError(
                "layout.positions_mm is omitted for visuotactile; the image is the element arrangement"
            )

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


@dataclass
class StreamInfo(Block):
    """The ``info`` block of a suffixed stream key (``.markers``, ``.displacement``, ``.force``).

    A suffixed stream carries only ``reduction``, when its main-table column is a
    dataset-rate copy of a faster native stream, and ``session_meta``. Everything
    else about the stream is read from the instance's own entry.
    """

    session_meta: str
    reduction: Reduction | None = None

    _label: ClassVar[str] = "info"
    _omit_when_none: ClassVar[frozenset[str]] = frozenset({"reduction"})
    _key_order: ClassVar[tuple[str, ...] | None] = ("reduction", "session_meta")

    def __post_init__(self) -> None:
        _require_session_meta(self.session_meta)


def read_info_lenient(data: Mapping[str, Any]) -> tuple[TactileInfo, list[str]]:
    """Decode an ``info`` block from a dataset this package did not write.

    Unknown keys are returned as dotted paths (``info.serial``) instead of raising,
    so a reader can report them. Wrong types and missing required keys still raise
    :class:`SchemaError`. Core's own ``video.*`` keys are ignored, as in
    :meth:`TactileInfo.from_dict`.
    """
    reader = _Reader(strict=False)
    own = {k: v for k, v in data.items() if not k.startswith("video.")}
    info = _from_dict(TactileInfo, own, reader, "info")
    return info, reader.unknown


# --- Per-session metadata: ``tactile/<instance_id>/meta.json`` ---------------------------------


@dataclass
class Clock(Block):
    """Anchors a session's host clock to the calendar and to the dataset's frames.

    Sidecar ``host_ts`` values come from ``time.perf_counter()``, which has no epoch,
    and LeRobot's own ``timestamp`` column is ``frame_index / fps``, restarting at
    zero in every episode. ``wall_time`` and ``host_clock`` are the UTC time and the
    host clock read at ``connect()``, so any ``host_ts`` maps to the calendar as
    ``wall_time + (host_ts - host_clock)``. ``episode_origins`` holds, for each
    episode in ``episodes`` in order, the host clock value when that episode's first
    dataset frame was built, so frame ``k`` of the episode sits at
    ``origin + k / fps`` on the host clock.
    """

    wall_time: str
    host_clock: float
    episode_origins: list[float]

    def __post_init__(self) -> None:
        if not isinstance(self.wall_time, str) or not _WALL_TIME.match(self.wall_time):
            raise SchemaError(
                "clock.wall_time must be UTC ISO 8601 like '2026-11-21T14:03:05.117Z', "
                f"got {self.wall_time!r}"
            )
        if not _is_number(self.host_clock):
            raise SchemaError(f"clock.host_clock must be a number, got {self.host_clock!r}")
        if not isinstance(self.episode_origins, list) or any(not _is_number(v) for v in self.episode_origins):
            raise SchemaError(f"clock.episode_origins must be numbers, got {self.episode_origins!r}")


@dataclass
class FrameCounts(Block):
    """The sensor's frame counters since ``connect()``, so per session.

    ``received`` frames reached the host; ``dropped`` were lost at the source, seen as
    gaps in ``seq``; ``overflowed`` were discarded on the host because the recorder
    did not drain the sensor in time. They are read from the sensor's counters, not
    reconstructed from the episode files.
    """

    received: int
    dropped: int
    overflowed: int

    def __post_init__(self) -> None:
        for name in ("received", "dropped", "overflowed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaError(f"frames.{name} must be a non-negative integer, got {value!r}")


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
    ``reference_frame`` pointer. Exactly one of the two is set. ``frame_range`` is
    ``(first_seq, last_seq)`` of the native frames averaged, or ``None`` when the
    sensor has no ``seq``; it is required on a strict read and tolerated as absent
    on a lenient one, since 0.2 records do not have it. ``applied`` says whether
    ``values`` were subtracted at record time, which is only valid when they are
    stored; later captures live in the baseline stream, not here.
    """

    captured_at: str
    duration_s: float
    applied: bool
    frame_range: tuple[int, int] | None
    values: list[list[float]] | None = None
    reference_frame: str | None = None

    _omit_when_none: ClassVar[frozenset[str]] = frozenset({"values", "reference_frame"})
    _lenient_missing: ClassVar[frozenset[str]] = frozenset({"frame_range"})

    def __post_init__(self) -> None:
        _require_nonempty_str(self.captured_at, "baseline.captured_at")
        _require_positive(self.duration_s, "baseline.duration_s")
        if not isinstance(self.applied, bool):
            raise SchemaError(f"baseline.applied must be a boolean, got {self.applied!r}")
        if self.frame_range is not None:
            self.frame_range = _seq_range(self.frame_range, "baseline.frame_range")
        if (self.values is None) == (self.reference_frame is None):
            raise SchemaError("baseline needs exactly one of 'values' or 'reference_frame'")
        if self.values is not None:
            _require_rows(self.values, None, "baseline.values")
        else:
            _require_nonempty_str(self.reference_frame, "baseline.reference_frame")
        if self.applied and self.values is None:
            raise SchemaError("baseline.applied is true but no values are stored")


@dataclass
class BaselineStream(Block):
    """Sampling period of the baseline stream: one raw sample every ``sample_every`` native samples."""

    sample_every: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_every, bool)
            or not isinstance(self.sample_every, int)
            or self.sample_every < 1
        ):
            raise SchemaError(
                f"baseline_stream.sample_every must be a positive integer, got {self.sample_every!r}"
            )


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
    ``clock``, ``frames`` and ``baseline_stream`` are written by the recorder; a
    ``clock`` must carry one episode origin per episode in the range.
    """

    session_id: str
    episodes: tuple[int, int]
    adapter: Adapter | None
    calibration: Calibration | None
    baseline: Baseline | None
    baseline_stream: BaselineStream | None = None
    clock: Clock | None = None
    frames: FrameCounts | None = None
    serial: str | None = None
    firmware: str | None = None
    device: DeviceSession | None = None
    gel: Gel | None = None
    skin: Skin | None = None

    _label: ClassVar[str] = "session"
    _key_order: ClassVar[tuple[str, ...] | None] = (
        "session_id",
        "episodes",
        "clock",
        "frames",
        "serial",
        "firmware",
        "device",
        "adapter",
        "calibration",
        "baseline",
        "baseline_stream",
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
        if self.clock is not None and len(self.clock.episode_origins) != last - first + 1:
            raise SchemaError(
                f"clock.episode_origins has {len(self.clock.episode_origins)} values for "
                f"{last - first + 1} episodes {list(self.episodes)}"
            )


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


def read_session_meta_lenient(data: Mapping[str, Any]) -> tuple[SessionMeta, list[str], list[str]]:
    """Decode a ``meta.json`` from a dataset this package did not write, or wrote under an earlier schema.

    Returns the metadata, the unknown keys and the tolerated missing keys, both as
    dotted paths (``meta.sessions[0].baseline.frame_range``). A missing
    ``baseline.frame_range``, added in 0.3, is read as ``None``. Wrong types and
    other missing required keys still raise :class:`SchemaError`.
    """
    reader = _Reader(strict=False)
    meta = _from_dict(SessionMeta, data, reader, "meta")
    return meta, reader.unknown, reader.missing


# --- Health annotations: ``tactile/health.json`` --------------------------------------------------


def health_report_path(root: str | Path) -> Path:
    """``<root>/tactile/health.json``."""
    return Path(root) / HEALTH_REPORT_PATH


@dataclass
class Skipped(Block):
    """A check, or one part of one, that could not run: ``skipped`` says why. It is never a pass."""

    skipped: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.skipped, "skipped")


@dataclass
class CopyCheck(Block):
    """Integrity, part ``copy``: the main-table copy equals the reduction recomputed from the native stream.

    ``first_mismatch`` is the frame index of the first disagreeing frame.
    """

    frames: int
    mismatches: int
    first_mismatch: int | None
    max_abs_diff: float
    flag: bool


@dataclass
class RawCheck(Block):
    """Integrity, part ``raw``: every sample row equals the reconstruction from stored data plus captures.

    ``first_mismatch`` is the ``sample_index`` of the first disagreeing sample. A mismatch
    means the subtractor changed without reporting; it is never counted as drift.
    """

    samples: int
    mismatches: int
    first_mismatch: int | None
    max_abs_diff: float
    flag: bool


@dataclass
class FilesCheck(Block):
    """Integrity, part ``files``: the sidecar is complete and consistent for the episode.

    ``capture_at_zero`` is ``None`` when no baseline is applied. ``counters`` holds when the
    session's ``frames.received`` is at least the native rows and ``frames.dropped`` at least
    the ``seq`` gaps inside the session's files; ``None`` when the record has no counters or a
    file is missing.
    """

    native: bool
    baseline: bool
    capture_at_zero: bool | None
    origins: bool
    counters: bool | None
    flag: bool


@dataclass
class IntegrityReport(Block):
    """The integrity check: three parts, each run or skipped on its own; ``flag`` when any part fails."""

    copy: CopyCheck | Skipped
    raw: RawCheck | Skipped
    files: FilesCheck | Skipped
    flag: bool


@dataclass
class RebaselineEvent(Block):
    """A capture row inside an episode: the baseline changed at ``sample_index``.

    ``delta`` is, per channel, the largest change of the baseline across elements. The
    event is reported, not scored as drift.
    """

    sample_index: int
    host_ts: float
    delta: list[float]


@dataclass
class DriftReport(Block):
    """The drift check: the no-contact level against the baseline in force, per channel.

    ``no_contact_level`` and ``level_change`` come from the episode alone (the quiet-window
    level at the start, mean over elements, and its change over the episode, the element of
    largest change). ``offset`` is that level against the baseline in force at the last
    no-contact window, per channel the element of largest magnitude, or ``skipped`` when there is
    no baseline; ``offset_pct_range`` scales it by ``range`` when known; ``slope_per_s`` is
    the median element's drift rate. ``baseline`` says
    where the baseline came from (``capture``, ``session``, ``none``), ``source`` what the
    raw level was read from (``samples``, ``raw_frames``, ``main_table``). ``frame_range``
    is the session baseline's, ``"not recorded"`` for a record without the key, ``"no seq"``
    for a sensor without one. ``candidates`` is how many pairs of quiet windows were tried
    as drift lines (at most 300, a seeded sample beyond that). ``first_frame_zero`` names
    the case where the level at frame 0 is exactly zero on every channel; it is an
    observation, never a flag. ``confidence`` is ``low`` when there is neither a range nor
    a baseline, so the numbers are a level change in unknown units. The flag compares
    ``offset_pct_range`` with the ``drift`` threshold when the range is known, and the
    native offset (or level change) with ``drift_units`` when the caller set one; without
    either, drift never flags.
    """

    no_contact_level: list[float]
    level_change: list[float]
    offset: list[float] | Skipped
    offset_pct_range: list[float] | None
    slope_per_s: list[float]
    baseline: str
    source: str
    baseline_age_s: float | None
    frame_range: list[int] | str | None
    windows: int
    no_contact_windows: int
    candidates: int
    first_frame_zero: bool
    rebaselines: list[RebaselineEvent]
    confidence: str
    flag: bool

    def __post_init__(self) -> None:
        _require_one_of(self.baseline, ("capture", "session", "none"), "drift.baseline")
        _require_one_of(self.source, ("samples", "raw_frames", "main_table"), "drift.source")
        _require_one_of(self.confidence, ("normal", "low"), "drift.confidence")


@dataclass
class GapHistogram(Block):
    """Gaps between consecutive samples in units of the declared period: ``counts[i]`` gaps fall
    between ``edges_periods[i - 1]`` and ``edges_periods[i]``; the last count is beyond the last edge."""

    edges_periods: list[float]
    counts: list[int]

    def __post_init__(self) -> None:
        if len(self.counts) != len(self.edges_periods) + 1:
            raise SchemaError("gap_histogram.counts needs one more entry than edges_periods")


@dataclass
class FramerateReport(Block):
    """The framerate check. ``source`` is ``sidecar`` (native ``host_ts`` and ``seq``) or
    ``main_table``; ``timestamps`` says whether they were ``measured`` or are LeRobot's nominal
    ``frame_index / fps`` (``nominal``), which says nothing about the sensor. ``frames`` is the
    session's counters when known."""

    effective_hz: float
    declared_hz: float | None
    dropped_pct: float | None
    max_gap_ms: float
    max_gap_threshold_ms: float
    gap_histogram: GapHistogram
    frames: FrameCounts | None
    source: str
    timestamps: str
    flag: bool

    def __post_init__(self) -> None:
        _require_one_of(self.source, ("sidecar", "main_table"), "framerate.source")
        _require_one_of(self.timestamps, ("measured", "nominal"), "framerate.timestamps")


@dataclass
class SaturationReport(Block):
    """The saturation check, per channel: the fraction of samples at or beyond the limit. ``limit`` is
    ``declared`` (from ``range``) or ``observed`` (the extremes seen, ``confidence`` ``low``)."""

    saturated_pct: list[float]
    limit: str
    min: list[float]
    max: list[float]
    confidence: str
    source: str
    flag: bool

    def __post_init__(self) -> None:
        _require_one_of(self.limit, ("declared", "observed"), "saturation.limit")
        _require_one_of(self.confidence, ("declared", "low"), "saturation.confidence")
        _require_one_of(self.source, ("sidecar", "main_table"), "saturation.source")


@dataclass
class GelReport(Block):
    """The gel check. ``kind`` ``pixels``: dead pixels of an image stream, with the largest connected
    cluster; ``kind`` ``elements``: zero-variance elements of an array stream, listed."""

    kind: str
    dead_pct: float
    largest_cluster_px: int | None
    dead_elements: list[int] | None
    frames_used: int
    flag: bool

    def __post_init__(self) -> None:
        _require_one_of(self.kind, ("pixels", "elements"), "gel.kind")


@dataclass
class SkewReport(Block):
    """The skew check: the lag of the tactile stream behind ``reference`` (a camera key or another
    instance) at a contact onset, with the cross-correlation peak as ``confidence``."""

    lag_ms: float
    confidence: float
    reference: str
    onsets: int
    resolution_ms: float
    flag: bool


@dataclass
class InstanceChecks(Block):
    """One episode's checks for one instance, each a result or ``Skipped``."""

    integrity: IntegrityReport | Skipped
    drift: DriftReport | Skipped
    framerate: FramerateReport | Skipped
    saturation: SaturationReport | Skipped
    gel: GelReport | Skipped
    skew: SkewReport | Skipped


@dataclass
class InstanceReport(Block):
    """What the run found out about an instance before checking any episode.

    ``metadata`` is ``info`` when the feature entry carried a readable ``info`` block,
    ``none`` when it had none (a plain LeRobot dataset), ``invalid`` when it failed even the
    lenient read, with the reason in ``error``. ``unknown_keys`` and ``missing_keys`` are
    the lenient readers' findings for the ``info`` block and the session record.
    """

    key: str
    family: str
    metadata: str
    sidecar: bool
    declared_rate_hz: float | None
    range: Range | None
    unknown_keys: list[str]
    missing_keys: list[str]
    error: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.key, "instance.key")
        _require_one_of(self.metadata, ("info", "none", "invalid"), "instance.metadata")


@dataclass
class Thresholds(Block):
    """The thresholds a run flagged against. The defaults are conservative. ``drift`` is a fraction of
    the declared range; ``drift_units`` is in the stream's native units, for a stream without a
    range, and ``None`` means drift does not flag there. ``max_gap_ms`` ``None`` means ten declared
    periods, resolved per instance and recorded in the framerate block. ``confidence`` is the
    cross-correlation peak a skew lag needs before it can be flagged."""

    drift: float = 0.05
    drift_units: float | None = None
    dropped: float = 0.02
    max_gap_ms: float | None = None
    gel: float = 0.05
    cluster_px: int = 25
    skew: float = 50.0
    confidence: float = 0.6
    saturated: float = 0.01
    integrity: bool = True


@dataclass
class HealthReport(Block):
    """The whole of ``tactile/health.json``: the doctor version, the thresholds, one
    :class:`InstanceReport` per instance and, per episode (``episode_{index:06d}``), one
    :class:`InstanceChecks` per instance."""

    doctor_version: str
    generated_at: str
    thresholds: Thresholds
    instances: dict[str, InstanceReport]
    episodes: dict[str, dict[str, InstanceChecks]]

    _label: ClassVar[str] = "health"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.doctor_version, "doctor_version")
        _require_nonempty_str(self.generated_at, "generated_at")
        for name in self.episodes:
            if not re.fullmatch(r"episode_\d{6}", name):
                raise SchemaError(f"episodes keys look like 'episode_000012', got {name!r}")


# --- Encoding and decoding -----------------------------------------------------------------------


class _Reader:
    """Decoding policy. Strict raises on an unknown key; lenient records it, and fills a required key a
    later schema version added (``_lenient_missing``) with ``None``, recording that too."""

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict
        self.unknown: list[str] = []
        self.missing: list[str] = []

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
    if isinstance(value, Block):
        return value.to_dict()
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
                if not reader.strict and field.name in getattr(cls, "_lenient_missing", ()):
                    reader.missing.append(f"{path}.{field.name}")
                    kwargs[field.name] = None
                    continue
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
        if len(members) == 1:
            return _decode(members[0], value, reader, path)
        return _decode_union(members, value, reader, path)
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


def _decode_union(members: list[Any], value: Any, reader: _Reader, path: str) -> Any:
    """Decode a value allowed to be one of several types.

    A mapping goes to the first dataclass member whose required keys it carries, which is
    how a check block is told apart from :class:`Skipped`; any other value goes to the first
    member that accepts it, as in ``float | list[float]``.
    """
    classes = [m for m in members if is_dataclass(m)]
    if isinstance(value, Mapping) and classes:
        # The member whose required keys are all present, else the closest one, so the error names
        # the unknown or missing key rather than the union.
        def closeness(cls: type) -> tuple[int, int]:
            required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
            return len(required & set(value)), -len(required - set(value))

        return _from_dict(max(classes, key=closeness), value, reader, path)
    for member in members:
        if is_dataclass(member):
            continue
        try:
            return _decode(member, value, reader, path)
        except SchemaError:
            continue
    raise SchemaError(f"{path}: {value!r} fits none of the allowed types")


def _enum(cls: type[Enum], value: Any, path: str) -> Any:
    try:
        return cls(value)
    except ValueError:
        raise SchemaError(f"{path} must be one of {[m.value for m in cls]}, got {value!r}") from None


def _require_nonempty_str(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{path} must be a non-empty string, got {value!r}")


def _require_one_of(value: Any, allowed: tuple[str, ...], path: str) -> None:
    if value not in allowed:
        raise SchemaError(f"{path} must be one of {allowed}, got {value!r}")


def _require_slug(value: Any, path: str) -> None:
    if not isinstance(value, str) or not _SLUG.match(value):
        raise SchemaError(f"{path} must match [A-Za-z0-9_-]+, got {value!r}")


def _require_session_meta(value: Any) -> None:
    if not isinstance(value, str) or not _SESSION_META.match(value):
        raise SchemaError(f"session_meta must look like 'tactile/<instance_id>/meta.json', got {value!r}")


def _seq_range(value: Any, path: str) -> tuple[int, int]:
    """Require ``[first, last]`` sequence numbers with ``0 <= first <= last``."""
    items = tuple(value) if isinstance(value, (list, tuple)) else ()
    if len(items) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in items):
        raise SchemaError(f"{path} must be two integers [first_seq, last_seq], got {value!r}")
    first, last = items
    if first < 0 or last < first:
        raise SchemaError(f"{path} must satisfy 0 <= first_seq <= last_seq, got {list(items)}")
    return first, last


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
