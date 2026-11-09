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
package did not write or wrote under an earlier version. The lenient ``info`` reader
also reads a 0.3 channel list as channel objects whose ``source`` is ``unknown``,
a value the strict reader rejects.
"""

from __future__ import annotations

import re
import types
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
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
    "CHANNEL_SOURCES",
    "CORE_INFO_KEYS",
    "CORE_INFO_PREFIXES",
    "Calibration",
    "CHECK_NAMES",
    "Channel",
    "ChannelReport",
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
    "HeldGroup",
    "HeldReport",
    "INDEXING_KINDS",
    "Indexing",
    "InstanceChecks",
    "InstanceReport",
    "IntegrityReport",
    "Layout",
    "Markers",
    "Mount",
    "ORIGIN_AT",
    "Origin",
    "PRESSURE_UNITS",
    "Processed",
    "REDUCTION_METHODS",
    "Range",
    "RawCheck",
    "RebaselineEvent",
    "Reduction",
    "Region",
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
    "TRANSPORT_KINDS",
    "TactileInfo",
    "Thresholds",
    "Transport",
    "UNCALIBRATED_UNITS",
    "UNKNOWN_SOURCE",
    "check_device_indices",
    "check_session_calibration",
    "feature_key",
    "health_report_path",
    "read_info_lenient",
    "read_session_meta_lenient",
    "session_meta_path",
    "without_core_keys",
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

CHANNEL_SOURCES = ("measured", "derived", "unavailable")
"""What a channel's column is: a value the sensor reported, one computed from another stream, or none."""

UNKNOWN_SOURCE = "unknown"
"""The ``source`` the lenient reader gives a channel read from 0.3 metadata, which never said; the strict
reader rejects it."""

UNCALIBRATED_UNITS = ("counts", "a.u.")
"""Units of a channel that is measured but not calibrated: raw integers from an analog-to-digital converter,
or a value the vendor scaled. Either requires the session's ``calibration.method`` to be ``uncalibrated``."""

PRESSURE_UNITS = ("Pa", "kPa", "N/mm2", "N/mm²", "psi")
"""Units that make a channel a pressure, which requires element area in its ``layout``."""

INDEXING_KINDS = ("grid", "regions")
"""Forms of ``layout.indexing``: one regular grid, or several."""

TRANSPORT_KINDS = ("usb", "ble", "serial", "ethernet", "can", "other")
"""Links a session record's ``transport.kind`` names."""

CORE_INFO_PREFIXES = ("video.", "audio.")
"""Prefixes of the keys LeRobot core writes into a video feature's ``info`` itself."""

CORE_INFO_KEYS = frozenset({"has_audio", "is_depth_map"})
"""The other keys LeRobot core writes into a video feature's ``info``; 0.6.1 writes both on every one."""

_CLOCK_DTYPES = ("float32", "float64", "int32", "int64")

HEALTH_REPORT_PATH = "tactile/health.json"
"""Where ``taxel doctor`` writes its report, relative to the dataset root."""

CHECK_NAMES = ("integrity", "drift", "framerate", "saturation", "gel", "skew", "held")
"""The checks a health report carries per episode per instance, in report order."""

_SLUG = re.compile(r"^[A-Za-z0-9_-]+$")
_SESSION_META = re.compile(r"^tactile/[A-Za-z0-9_-]+/meta\.json$")
_WALL_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


class SchemaError(ValueError):
    """Raised when metadata does not satisfy ``SCHEMA.md``."""


_LENIENT: ContextVar[bool] = ContextVar("taxel_schema_lenient", default=False)
"""Set while the lenient reader builds a block, so that validation accepts what only it produces: a
channel whose ``source`` is ``unknown`` or that has no unit, and a missing ``device.index``."""


_T = TypeVar("_T", bound="Block")


class Block:
    """Mixin giving every schema dataclass ``to_dict`` and a strict ``from_dict``.

    Subclasses may set ``_label`` (the root of error paths), ``_omit_when_none``
    (keys left out of the dict when ``None``), ``_key_order`` (the key order in
    ``SCHEMA.md``, when it differs from field order) and ``_lenient_missing``
    (keys a later schema version made required: a lenient read fills them with
    ``None`` when absent and reports them, for documents written under an earlier
    version). Without ``_key_order`` a ``flag`` key is
    written last, after any field a later version added with a default.
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
class Channel(Block):
    """One column of an array entry's last dimension: its name, its unit, and what it is.

    ``source`` is one of :data:`CHANNEL_SOURCES`: ``measured``, a value the sensor reported;
    ``derived``, a value computed from another stream, whose method the instance's ``derived`` block
    names; or ``unavailable``, a column with no measurement, stored as 0.0, which only converted data
    that keeps a fixed channel set declares. ``unit`` is a free string, physical where the sensor is
    calibrated and ``counts`` or ``a.u.`` (:data:`UNCALIBRATED_UNITS`) where it is not. Only the
    lenient reader, reading 0.3 metadata, gives a channel the ``unknown`` source or no unit.
    """

    name: str
    unit: str | None
    source: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "channel.name")
        lenient = _LENIENT.get()
        if self.unit is not None or not lenient:
            _require_nonempty_str(self.unit, f"channel {self.name!r}: unit")
        sources = (*CHANNEL_SOURCES, UNKNOWN_SOURCE) if lenient else CHANNEL_SOURCES
        _require_one_of(self.source, sources, f"channel {self.name!r}: source")

    @property
    def is_pressure(self) -> bool:
        """Whether the unit is a pressure (:data:`PRESSURE_UNITS`), which needs element area."""
        return self.unit in PRESSURE_UNITS


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
class Region(Block):
    """One regular grid of elements in an instance's flat element order, part of ``layout.indexing``.

    ``shape`` is ``[rows, cols]`` and ``pitch_mm`` ``[dy, dx]``, the centre-to-centre distance between
    neighbouring elements down a column and along a row. Elements are row-major: element ``(r, c)`` is
    element ``first_element + r * cols + c`` of the feature and sits ``c * dx`` along +x and ``r * dy``
    along +y from the region's first element. ``area_mm2`` is each element's area, required when a
    channel is a pressure.
    """

    name: str
    shape: list[int]
    pitch_mm: list[float]
    first_element: int
    area_mm2: float | None = None

    _omit_when_none: ClassVar[frozenset[str]] = frozenset({"area_mm2"})

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "region.name")
        _require_grid(self.shape, self.pitch_mm, self.area_mm2, f"region {self.name!r}")
        first = self.first_element
        if isinstance(first, bool) or not isinstance(first, int) or first < 0:
            raise SchemaError(
                f"region {self.name!r}: first_element must be a non-negative integer, got {first!r}"
            )

    @property
    def n_elements(self) -> int:
        return self.shape[0] * self.shape[1]


@dataclass
class Indexing(Block):
    """Element geometry for a source that publishes a grid, not each element's position: ``layout.indexing``.

    ``kind`` ``grid`` is one regular grid over every element, with ``shape``, ``pitch_mm`` and an
    optional ``area_mm2`` as in :class:`Region`; ``regions`` is several :class:`Region` blocks, which
    cover the instance's elements exactly once. Where one region lies relative to another is not part
    of it; ``positions_mm`` says that.
    """

    kind: str
    shape: list[int] | None = None
    pitch_mm: list[float] | None = None
    area_mm2: float | None = None
    regions: list[Region] | None = None

    _omit_when_none: ClassVar[frozenset[str]] = frozenset({"shape", "pitch_mm", "area_mm2", "regions"})

    def __post_init__(self) -> None:
        _require_one_of(self.kind, INDEXING_KINDS, "indexing.kind")
        if self.kind == "grid":
            if self.regions is not None:
                raise SchemaError("indexing: a grid has shape and pitch_mm, not regions")
            _require_grid(self.shape, self.pitch_mm, self.area_mm2, "indexing")
            return
        if self.shape is not None or self.pitch_mm is not None or self.area_mm2 is not None:
            raise SchemaError("indexing: each region carries its own shape, pitch_mm and area_mm2")
        if not self.regions:
            raise SchemaError("indexing: kind 'regions' needs at least one region")
        names = [r.name for r in self.regions]
        if len(set(names)) != len(names):
            raise SchemaError(f"indexing: region names must be unique, got {names!r}")
        expected = 0
        for region in sorted(self.regions, key=lambda r: r.first_element):
            if region.first_element != expected:
                raise SchemaError(
                    f"indexing: the regions must cover the elements once; region {region.name!r} starts at "
                    f"element {region.first_element}, expected {expected}"
                )
            expected += region.n_elements

    @property
    def parts(self) -> list[Region]:
        """The regions, with a grid as one region that starts at element 0."""
        if self.kind == "grid":
            assert self.shape is not None and self.pitch_mm is not None
            return [Region("grid", list(self.shape), list(self.pitch_mm), 0, self.area_mm2)]
        return list(self.regions or ())

    @property
    def n_elements(self) -> int:
        return sum(r.n_elements for r in self.parts)


@dataclass
class Layout(Block):
    """The sensor's coordinate frame and, for the array families, where its elements are.

    ``origin``, ``axes`` and ``surface`` (one of :data:`SURFACES`) make the frame explicit, so two
    sensors' positions are comparable. ``positions_mm`` has one ``[x, y, z]`` row per element, in
    millimetres; it is a point set that assumes no grid, and is omitted for an image sensor, whose
    image is the element arrangement. ``normals`` is ``None`` or one unit vector per position, and is
    written only beside ``positions_mm``. ``indexing`` is the fallback for a source that publishes a
    grid instead of positions. Element area, required for a pressure channel, is ``area_mm2`` in
    ``indexing``, or ``element_area_mm2`` (one number, or one per element) beside ``positions_mm``
    alone.
    """

    origin: Origin
    axes: Axes
    surface: str
    positions_mm: list[list[float]] | None = None
    normals: list[list[float]] | None = None
    indexing: Indexing | None = None
    element_area_mm2: float | list[float] | None = None

    _omit_when_none: ClassVar[frozenset[str]] = frozenset({"positions_mm", "indexing", "element_area_mm2"})

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
        positions, indexing = self.positions_mm, self.indexing
        if positions is not None and indexing is not None and len(positions) != indexing.n_elements:
            raise SchemaError(
                f"layout.positions_mm has {len(positions)} rows, "
                f"layout.indexing {indexing.n_elements} elements"
            )
        area = self.element_area_mm2
        if area is not None:
            if self.indexing is not None:
                raise SchemaError(
                    "layout.element_area_mm2 is for positions_mm alone; "
                    "with indexing, area_mm2 goes on each region"
                )
            if self.positions_mm is None:
                raise SchemaError("layout.element_area_mm2 is given without positions_mm")
            values = area if isinstance(area, list) else [area]
            if not values or any(not _is_number(v) or not v > 0 for v in values):
                raise SchemaError(
                    f"layout.element_area_mm2 must be positive, one number or one per element, got {area!r}"
                )
            if isinstance(area, list) and len(area) != len(self.positions_mm):
                raise SchemaError(
                    f"layout.element_area_mm2 has {len(area)} values for {len(self.positions_mm)} positions"
                )

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        if self.positions_mm is None:
            del out["normals"]
        return out

    @property
    def n_elements(self) -> int | None:
        """Elements the layout describes: rows of ``positions_mm``, else elements of ``indexing``."""
        if self.positions_mm is not None:
            return len(self.positions_mm)
        return None if self.indexing is None else self.indexing.n_elements

    @property
    def has_area(self) -> bool:
        """Whether every element has an area, as a pressure channel requires."""
        if self.indexing is not None:
            return all(r.area_mm2 is not None for r in self.indexing.parts)
        return self.element_area_mm2 is not None


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
    """The algorithm that produced an instance's derived channels: the columns of its suffixed streams,
    which are all derived, and any channel with ``source`` ``derived``."""

    method: str
    version: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.method, "derived.method")
        _require_nonempty_str(self.version, "derived.version")


@dataclass
class Processed(Block):
    """The processing that made a visuotactile image from raw gel frames, such as a difference image or a
    deformation map. A raw gel image has none, and a check that assumes one, such as dead pixels, does not
    apply."""

    method: str
    version: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.method, "processed.method")
        _require_nonempty_str(self.version, "processed.version")


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

    Instances with the same ``clock`` have directly comparable ``device_ts``. ``index`` is the
    instance's 0-based position on the device, unique among the instances with the same ``id`` and
    fixed for the dataset's life. :class:`TactileInfo` and a sensor config require it; a lenient read
    of 0.3 metadata, which had none, leaves it ``None`` and reports it.
    """

    id: str
    clock: str
    index: int | None = None

    _lenient_missing: ClassVar[frozenset[str]] = frozenset({"index"})

    def __post_init__(self) -> None:
        _require_slug(self.id, "device.id")
        _require_slug(self.clock, "device.clock")
        index = self.index
        if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
            raise SchemaError(f"device.index must be a non-negative integer, got {index!r}")

    def require_index(self) -> None:
        """Raise the fix when ``index`` is missing: it is set in the sensor config."""
        if self.index is None:
            raise SchemaError(
                f"device.index is missing: set device.index in the sensor config, the instance's 0-based "
                f"position on device {self.id!r}"
            )


@dataclass
class TactileInfo(Block):
    """Static metadata of one sensor instance: the ``info`` block of its feature entry.

    Only facts that hold in every session belong here. LeRobot's ``merge``
    requires feature entries to be identical, so anything that varies between
    sessions goes in :class:`SessionRecord` instead.

    ``family``, ``vendor``, ``model``, ``declared_rate_hz``, ``layout``, ``mount``,
    ``device`` and ``session_meta`` are always written, the other keys only when set.
    ``channels`` is required for the array families, one :class:`Channel` per column,
    and absent for ``visuotactile``, whose suffixed streams declare their own. ``layout``
    is required for every family: ``taxel_array`` and ``magnetic`` need ``positions_mm``
    or ``indexing`` in it, ``force_torque`` has at most one position, and
    ``visuotactile`` none. A ``derived`` channel needs the ``derived`` block, a pressure
    channel needs element area, and ``processed`` is for ``visuotactile`` only.
    ``clock_column`` names a main-table feature of real per-frame times, checked against
    the dataset's features by :meth:`check_clock_column`.
    """

    family: TactileFamily
    declared_rate_hz: float
    session_meta: str
    vendor: str | None = None
    model: str | None = None
    channels: list[Channel] | None = None
    range: Range | None = None
    layout: Layout | None = None
    reduction: Reduction | None = None
    markers: Markers | None = None
    derived: Derived | None = None
    processed: Processed | None = None
    clock_column: str | None = None
    mount: Mount | None = None
    device: DeviceGroup | None = None

    _label: ClassVar[str] = "info"
    _omit_when_none: ClassVar[frozenset[str]] = frozenset(
        {"channels", "range", "reduction", "markers", "derived", "processed", "clock_column"}
    )
    _key_order: ClassVar[tuple[str, ...] | None] = (
        "family",
        "vendor",
        "model",
        "declared_rate_hz",
        "channels",
        "range",
        "layout",
        "reduction",
        "markers",
        "derived",
        "processed",
        "clock_column",
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
            _require_channels(self.channels, "channels")
            if self.family is TactileFamily.FORCE_TORQUE and len(self.channels) != 6:
                raise SchemaError(f"force_torque declares six channels, got {len(self.channels)}")
        elif self.channels is not None:
            raise SchemaError(
                "a visuotactile image entry has no channels; its suffixed streams declare theirs"
            )
        if self.processed is not None and self.family is not TactileFamily.VISUOTACTILE:
            raise SchemaError(f"processed is defined for visuotactile images, not {self.family.value!r}")
        derived = [c.name for c in self.channels or () if c.source == "derived"]
        if derived and self.derived is None:
            raise SchemaError(
                f"channels {derived} are derived, which requires a 'derived' block naming the method"
            )
        if self.clock_column is not None:
            _require_nonempty_str(self.clock_column, "clock_column")
        if self.device is not None and not _LENIENT.get():
            self.device.require_index()
        if self.range is not None:
            if self.family not in ARRAY_FAMILIES:
                raise SchemaError(f"range is defined for the array families, not {self.family.value!r}")
            per_channel = self.range.n_channels
            if per_channel is not None and per_channel != len(self.channels or ()):
                raise SchemaError(f"range lists {per_channel} values for {len(self.channels or ())} channels")
        if self.layout is None:
            raise SchemaError("layout is required: origin, axes and surface declare the sensor's frame")
        layout = self.layout
        positions = layout.positions_mm
        if self.family in (TactileFamily.TAXEL_ARRAY, TactileFamily.MAGNETIC):
            if positions is None and layout.indexing is None:
                raise SchemaError(
                    f"layout.positions_mm or layout.indexing is required for family {self.family.value!r}"
                )
        elif layout.indexing is not None or layout.element_area_mm2 is not None:
            raise SchemaError(
                f"layout.indexing and element area are defined for taxel_array and magnetic, "
                f"not {self.family.value!r}"
            )
        if self.family is TactileFamily.FORCE_TORQUE and positions is not None and len(positions) != 1:
            raise SchemaError(f"force_torque is one element; layout.positions_mm has {len(positions)} rows")
        if self.family is TactileFamily.VISUOTACTILE and positions is not None:
            raise SchemaError(
                "layout.positions_mm is omitted for visuotactile; the image is the element arrangement"
            )
        pressures = [c.name for c in self.channels or () if c.is_pressure]
        if pressures and not layout.has_area:
            raise SchemaError(
                f"channels {pressures} are pressures, which require element area: area_mm2 on each region "
                "of layout.indexing, or layout.element_area_mm2 beside positions_mm"
            )

    @property
    def n_channels(self) -> int | None:
        """Last dimension of a per-frame array, or ``None`` for an image stream."""
        return None if self.channels is None else len(self.channels)

    def check_streams(self, suffixes: Iterable[str]) -> None:
        """Validate the suffixed streams declared beside this instance's key.

        Any suffixed stream requires the visuotactile family and, since every one is
        computed from the image, a ``derived`` block naming the method.
        """
        declared = set(suffixes)
        unknown = declared - set(STREAM_SUFFIXES)
        if unknown:
            raise SchemaError(f"unknown stream suffixes {sorted(unknown)}; expected {STREAM_SUFFIXES}")
        if declared and self.family is not TactileFamily.VISUOTACTILE:
            raise SchemaError(
                f"suffixed streams are defined for visuotactile instances only, not {self.family.value!r}"
            )
        if declared and self.derived is None:
            raise SchemaError("suffixed streams are derived from the image and require a 'derived' block")

    def check_clock_column(self, features: Mapping[str, Mapping[str, Any]]) -> None:
        """Check that ``clock_column`` names a feature of the dataset holding one number per frame.

        Raises:
            SchemaError: If it names no feature in ``features``, or one that is not a single
                number per frame.
        """
        if self.clock_column is None:
            return
        entry = features.get(self.clock_column)
        if entry is None:
            raise SchemaError(f"clock_column {self.clock_column!r} is not a feature of the dataset")
        dtype, shape = entry.get("dtype"), list(entry.get("shape") or [])
        if dtype not in _CLOCK_DTYPES or shape not in ([], [1]):
            raise SchemaError(
                f"clock_column {self.clock_column!r} is {dtype} {shape}, not one number per frame"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TactileInfo:
        """Build from an ``info`` block, strictly. LeRobot's own keys are set aside (see
        :func:`without_core_keys`). A 0.3 channel list of names is an error; :func:`read_info_lenient`
        reads it."""
        own = without_core_keys(data)
        channels = own.get("channels")
        if isinstance(channels, list) and any(isinstance(c, str) for c in channels):
            raise SchemaError(
                "info.channels: each channel is an object with name, unit and source since 0.4; "
                "a list of names is the 0.3 form, which read_info_lenient reads"
            )
        return super().from_dict(own)


@dataclass
class StreamInfo(Block):
    """The ``info`` block of a suffixed stream key (``.markers``, ``.displacement``, ``.force``).

    A suffixed stream carries its ``channels``, every one ``derived``, since the stream is
    computed from the image by the method the instance's ``derived`` block names;
    ``reduction``, when its main-table column is a dataset-rate copy of a faster native
    stream; and ``session_meta``. Everything else about the stream is read from the
    instance's own entry.
    """

    session_meta: str
    channels: list[Channel]
    reduction: Reduction | None = None

    _label: ClassVar[str] = "info"
    _omit_when_none: ClassVar[frozenset[str]] = frozenset({"reduction"})
    _key_order: ClassVar[tuple[str, ...] | None] = ("channels", "reduction", "session_meta")

    def __post_init__(self) -> None:
        _require_session_meta(self.session_meta)
        if not self.channels:
            raise SchemaError("a suffixed stream declares its channels")
        _require_channels(self.channels, "channels")
        measured = [c.name for c in self.channels if c.source != "derived"]
        if measured:
            raise SchemaError(
                f"a suffixed stream is derived from the image; channels {measured} must be 'derived'"
            )


def without_core_keys(data: Mapping[str, Any]) -> dict[str, Any]:
    """``data`` without the keys LeRobot core writes into a feature's ``info`` itself.

    Those are everything under :data:`CORE_INFO_PREFIXES` and the keys in :data:`CORE_INFO_KEYS`;
    LeRobot 0.6.1 writes ``has_audio`` and ``is_depth_map`` on every video feature.
    """
    return {k: v for k, v in data.items() if not k.startswith(CORE_INFO_PREFIXES) and k not in CORE_INFO_KEYS}


def read_info_lenient(data: Mapping[str, Any]) -> tuple[TactileInfo, list[str], list[str], bool]:
    """Decode an ``info`` block from a dataset this package did not write, or wrote under an earlier schema.

    Returns the info, the unknown keys and the tolerated missing keys, both as dotted paths
    (``info.serial``, ``info.device.index``), and whether a 0.3 channel declaration was
    upgraded. A 0.3 array entry lists ``channels`` as names with units per quantity in a
    ``units`` block; each name becomes a :class:`Channel` with ``source`` ``unknown``, never
    ``measured`` (see :func:`_upgrade_channels`). A missing ``device.index``, added in 0.4, is
    read as ``None``. LeRobot's own keys are set aside, as in :meth:`TactileInfo.from_dict`.
    Wrong types and other missing required keys still raise :class:`SchemaError`.
    """
    reader = _Reader(strict=False)
    own, upgraded = _upgrade_channels(without_core_keys(data))
    info = _from_dict(TactileInfo, own, reader, "info")
    return info, reader.unknown, reader.missing, upgraded


_QUANTITY_PREFIXES = {"force": ("f",), "torque": ("m", "t")}
"""How a 0.3 ``units`` block that names several quantities is matched to channel names."""


def _upgrade_channels(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """A 0.3 channel list, names plus a ``units`` block keyed by quantity, as 0.4 channel objects.

    Each channel gets ``source`` ``unknown``. Its unit is the block's when the block names one
    quantity, the quantity its name starts with when it names several (``f`` for force, ``m`` or
    ``t`` for torque), and ``None`` without a block or a match. Any other block is returned as is.
    """
    channels = data.get("channels")
    if not isinstance(channels, list) or not channels or not all(isinstance(c, str) for c in channels):
        return data, False
    units = data.get("units")
    upgraded = {k: v for k, v in data.items() if k != "units"}
    upgraded["channels"] = [
        {"name": name, "unit": _unit_of(name, units), "source": UNKNOWN_SOURCE} for name in channels
    ]
    return upgraded, True


def _unit_of(name: str, units: Any) -> str | None:
    if not isinstance(units, Mapping) or not units:
        return None
    if len(units) == 1:
        matches = list(units.values())
    else:
        matches = [
            u for quantity, u in units.items() if name.startswith(_QUANTITY_PREFIXES.get(quantity, ()))
        ]
    return matches[0] if len(matches) == 1 and isinstance(matches[0], str) and matches[0] else None


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
class Transport(Block):
    """The link an instance was read over in one session, and the rate the sensor was configured to on it.

    ``kind`` is one of :data:`TRANSPORT_KINDS`. ``rate_hz`` is the configured rate, not a measured one,
    or ``None`` when unknown; a sensor's rate can depend on its link. ``info.declared_rate_hz`` stays
    the instance's nominal native rate, the same in every session.
    """

    kind: str
    rate_hz: float | None

    def __post_init__(self) -> None:
        _require_one_of(self.kind, TRANSPORT_KINDS, "transport.kind")
        if self.rate_hz is not None:
            _require_positive(self.rate_hz, "transport.rate_hz")


@dataclass
class SessionRecord(Block):
    """One recording session of one instance.

    ``adapter``, ``calibration`` and ``baseline`` have no default: they are
    always written, as ``null`` when unknown, so their absence is a visible fact.
    ``episodes`` is the inclusive range of episode indices the session covers.
    ``clock``, ``frames`` and ``baseline_stream`` are written by the recorder; a
    ``clock`` must carry one episode origin per episode in the range. ``transport`` is
    the link the session read the sensor over, ``None`` when unknown.
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
    transport: Transport | None = None
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
        "transport",
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


def check_session_calibration(info: TactileInfo, record: SessionRecord) -> None:
    """Check a session record against its instance's channels: a channel in ``counts`` or ``a.u.``
    (:data:`UNCALIBRATED_UNITS`) requires ``calibration.method`` ``uncalibrated``.

    Raises:
        SchemaError: If an uncalibrated unit meets another calibration method, or none.
    """
    units = sorted({c.unit for c in info.channels or () if c.unit in UNCALIBRATED_UNITS})
    method = None if record.calibration is None else record.calibration.method
    if units and method != "uncalibrated":
        raise SchemaError(
            f"session {record.session_id!r}: channels in {units} require calibration.method "
            f"'uncalibrated', got {method!r}"
        )


def check_device_indices(infos: Mapping[str, TactileInfo]) -> None:
    """Check that the instances on each device, keyed here by instance id, have distinct ``device.index``.

    Raises:
        SchemaError: If two instances with the same ``device.id`` share an index.
    """
    seen: dict[tuple[str, int | None], str] = {}
    for instance_id, info in infos.items():
        if info.device is None:
            continue
        slot = (info.device.id, info.device.index)
        if slot in seen:
            raise SchemaError(
                f"{seen[slot]!r} and {instance_id!r} share device.index {info.device.index} "
                f"on device {info.device.id!r}"
            )
        seen[slot] = instance_id


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

    ``delta`` is, per channel, the largest change of the baseline across elements, ``None`` at an
    ``unavailable`` channel. The event is reported, not scored as drift.
    """

    sample_index: int
    host_ts: float
    delta: list[float | None]


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
    either, drift never flags. ``nan_pct`` is, per element, the fraction of frames in which it
    reads NaN; an element over half is left out of the check, and the rest is measured on the
    finite frames. Every per-channel list is ``None`` at a channel declared ``unavailable``, which
    the check leaves out.
    """

    no_contact_level: list[float | None]
    level_change: list[float | None]
    offset: list[float | None] | Skipped
    offset_pct_range: list[float | None] | None
    slope_per_s: list[float | None]
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
    nan_pct: list[float] | None = None

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
    ``frame_index / fps`` (``nominal``), which says nothing about the sensor. ``clock`` names the
    main-table column the times were read from when it is not ``timestamp``. ``stalled`` counts
    the steps of zero length, a clock that did not advance, apart from the drops. ``frames`` is
    the session's counters when known. ``configured_hz`` is the rate the session's ``transport``
    says the sensor was set to, reported beside ``effective_hz`` and never flagged."""

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
    stalled: int | None = None
    clock: str | None = None
    configured_hz: float | None = None

    _key_order: ClassVar[tuple[str, ...] | None] = (
        "effective_hz",
        "configured_hz",
        "declared_hz",
        "dropped_pct",
        "max_gap_ms",
        "max_gap_threshold_ms",
        "gap_histogram",
        "frames",
        "source",
        "timestamps",
        "stalled",
        "clock",
        "flag",
    )

    def __post_init__(self) -> None:
        _require_one_of(self.source, ("sidecar", "main_table"), "framerate.source")
        _require_one_of(self.timestamps, ("measured", "nominal"), "framerate.timestamps")


@dataclass
class SaturationReport(Block):
    """The saturation check, per channel: the fraction of samples at or beyond the limit, and at the
    lower (``at_min_pct``) and the upper (``at_max_pct``) limit separately. ``limit`` is ``declared``
    (from ``range``) or ``observed`` (the extremes seen, ``confidence`` ``low``). Only a declared
    limit flags: a stream rests at an observed extreme as readily as it clips there. ``nan_pct``
    is per element; the fractions are of the finite samples of the elements NaN in at most half
    the frames. Every per-channel list is ``None`` at a channel declared ``unavailable``."""

    saturated_pct: list[float | None]
    limit: str
    min: list[float | None]
    max: list[float | None]
    confidence: str
    source: str
    flag: bool
    at_min_pct: list[float | None] | None = None
    at_max_pct: list[float | None] | None = None
    nan_pct: list[float] | None = None

    _key_order: ClassVar[tuple[str, ...] | None] = (
        "saturated_pct",
        "at_min_pct",
        "at_max_pct",
        "limit",
        "min",
        "max",
        "confidence",
        "source",
        "nan_pct",
        "flag",
    )

    def __post_init__(self) -> None:
        _require_one_of(self.limit, ("declared", "observed"), "saturation.limit")
        _require_one_of(self.confidence, ("declared", "low"), "saturation.confidence")
        _require_one_of(self.source, ("sidecar", "main_table"), "saturation.source")


@dataclass
class GelReport(Block):
    """The gel check.

    ``kind`` ``pixels``: dead pixels of an image stream, with the largest connected cluster in
    pixels and as a fraction of the image; ``median_variance`` is the episode's median pixel
    variance in counts² and ``rest_variance`` the instance's rest flicker the check was judged
    on, the quietest checked episode's median. ``kind`` ``elements``: ``constant_elements`` are
    the elements constant in this episode, an observation, and ``dead_elements`` those constant
    in every checked episode of the instance, which is what flags; it is ``null`` until two
    episodes were checked, since one cannot tell a dead element from one nothing touched.
    ``episodes_checked`` is how many episodes that decision, or the rest flicker, was taken
    over. ``nan_pct`` is per element, on an array stream.
    """

    kind: str
    dead_pct: float
    largest_cluster_px: int | None
    dead_elements: list[int] | None
    frames_used: int
    flag: bool
    largest_cluster_pct: float | None = None
    constant_elements: list[int] | None = None
    median_variance: float | None = None
    rest_variance: float | None = None
    episodes_checked: int = 1
    nan_pct: list[float] | None = None

    _key_order: ClassVar[tuple[str, ...] | None] = (
        "kind",
        "dead_pct",
        "largest_cluster_px",
        "largest_cluster_pct",
        "dead_elements",
        "constant_elements",
        "frames_used",
        "median_variance",
        "rest_variance",
        "episodes_checked",
        "nan_pct",
        "flag",
    )

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
class HeldGroup(Block):
    """Held frames of one group of elements, a fingertip's six channels say: the measures of
    :class:`HeldReport` over the group's elements alone."""

    held_pct: float
    longest_run: int
    single_holds: int


@dataclass
class HeldReport(Block):
    """The held check on an array stream: frames identical to the frame before.

    ``held_pct`` is the fraction of compared frames equal to the previous frame on every element,
    and ``frames`` how many were compared: a pair in which either frame holds a NaN is left out,
    and so is an element NaN in more than half the frames. ``longest_run`` is the most consecutive
    held frames and ``single_holds`` the runs of exactly one. ``expected_pct`` is what a correct
    stream holds anyway: 0 on a native stream, ``1 - rate / fps`` on a main-table copy of a slower
    sensor. ``groups`` has the same measures per group of elements when the feature's names share
    stems (``left_thumb_f6_0`` to ``_5``), else ``null``; a frame held on one group and not on the
    others is a held group, not a held frame. ``source`` is ``sidecar`` or ``main_table`` and
    ``nan_pct`` is per element. The flag needs a declared rate: ``held_pct`` above
    ``expected_pct`` by more than the ``held`` threshold.
    """

    held_pct: float
    frames: int
    longest_run: int
    single_holds: int
    expected_pct: float
    groups: dict[str, HeldGroup] | None
    nan_pct: list[float]
    source: str
    flag: bool

    def __post_init__(self) -> None:
        _require_one_of(self.source, ("sidecar", "main_table"), "held.source")


@dataclass
class InstanceChecks(Block):
    """One episode's checks for one instance, each a result or ``Skipped``. ``held`` is absent from
    a report written before the check existed."""

    integrity: IntegrityReport | Skipped
    drift: DriftReport | Skipped
    framerate: FramerateReport | Skipped
    saturation: SaturationReport | Skipped
    gel: GelReport | Skipped
    skew: SkewReport | Skipped
    held: HeldReport | Skipped | None = None

    _omit_when_none: ClassVar[frozenset[str]] = frozenset({"held"})


@dataclass
class ChannelReport(Block):
    """A channel as a run read it, in an :class:`InstanceReport`: its name, unit and source. Unlike
    :class:`Channel` it also holds what a lenient read of 0.3 metadata gives, ``source`` ``unknown``
    and no unit, so a report on such a dataset reads back."""

    name: str
    unit: str | None
    source: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.name, "channel.name")
        _require_one_of(self.source, (*CHANNEL_SOURCES, UNKNOWN_SOURCE), f"channel {self.name!r}: source")


@dataclass
class InstanceReport(Block):
    """What the run found out about an instance before checking any episode.

    ``metadata`` is ``info`` when the feature entry carried a readable ``info`` block,
    ``none`` when it had none (a plain LeRobot dataset), ``invalid`` when it failed even the
    lenient read, with the reason in ``error``. ``channels`` are the instance's channels as
    read, ``None`` without an ``info`` block or for an image, and ``upgraded`` says the
    lenient reader turned a 0.3 channel list into them. ``unknown_keys`` and ``missing_keys``
    are the lenient readers' findings for the ``info`` block and the session record.
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
    channels: list[ChannelReport] | None = None
    upgraded: bool = False

    _key_order: ClassVar[tuple[str, ...] | None] = (
        "key",
        "family",
        "metadata",
        "sidecar",
        "declared_rate_hz",
        "range",
        "channels",
        "upgraded",
        "unknown_keys",
        "missing_keys",
        "error",
    )

    def __post_init__(self) -> None:
        _require_nonempty_str(self.key, "instance.key")
        _require_one_of(self.metadata, ("info", "none", "invalid"), "instance.metadata")


@dataclass
class Thresholds(Block):
    """The thresholds a run flagged against. The defaults are conservative. ``drift`` is a fraction of
    the declared range; ``drift_units`` is in the stream's native units, for a stream without a
    range, and ``None`` means drift does not flag there. ``max_gap_ms`` ``None`` means ten declared
    periods, resolved per instance and recorded in the framerate block. ``cluster_pct`` is the
    largest dead cluster as a fraction of the image; ``cluster_px`` the same in pixels, which
    replaces the fraction when set. ``confidence`` is the cross-correlation peak a skew lag needs
    before it can be flagged. ``held`` is how far the held fraction may exceed the expected one."""

    drift: float = 0.05
    drift_units: float | None = None
    dropped: float = 0.02
    max_gap_ms: float | None = None
    gel: float = 0.05
    cluster_px: int | None = None
    cluster_pct: float = 0.005
    skew: float = 50.0
    confidence: float = 0.6
    saturated: float = 0.01
    held: float = 0.05
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
    order = getattr(cls, "_key_order", None) or _flag_last(tuple(f.name for f in fields(obj)))
    out: dict[str, Any] = {}
    for name in order:
        value = getattr(obj, name)
        if value is None and name in omit:
            continue
        out[name] = _encode(value)
    return out


def _flag_last(names: tuple[str, ...]) -> tuple[str, ...]:
    """Field order with ``flag`` moved last: a check block's verdict follows the fields added later."""
    return (*(n for n in names if n != "flag"), "flag") if "flag" in names else names


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
            if not reader.strict and field.name in getattr(cls, "_lenient_missing", ()):
                reader.missing.append(f"{path}.{field.name}")
                kwargs[field.name] = None
                continue
            if field.default is MISSING and field.default_factory is MISSING:
                raise SchemaError(f"{path}: missing required key {field.name!r}")
            continue
        kwargs[field.name] = _decode(hints[field.name], data[field.name], reader, f"{path}.{field.name}")
    token = _LENIENT.set(not reader.strict)
    try:
        return cls(**kwargs)
    except SchemaError as e:
        raise SchemaError(f"{path}: {e}") from None
    finally:
        _LENIENT.reset(token)


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


def _require_channels(channels: Any, path: str) -> None:
    """Require a list of :class:`Channel` objects with unique names."""
    if not isinstance(channels, list) or any(not isinstance(c, Channel) for c in channels):
        raise SchemaError(
            f"{path} must be Channel objects (name, unit, source); a list of names is the 0.3 form"
        )
    names = [c.name for c in channels]
    if len(set(names)) != len(names):
        raise SchemaError(f"{path} must have unique names, got {names!r}")


def _require_grid(shape: Any, pitch_mm: Any, area_mm2: Any, path: str) -> None:
    """Require a grid's ``[rows, cols]``, its ``[dy, dx]`` pitch in mm and an optional positive area."""
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape)
    ):
        raise SchemaError(f"{path}: shape must be two positive integers [rows, cols], got {shape!r}")
    if (
        not isinstance(pitch_mm, (list, tuple))
        or len(pitch_mm) != 2
        or any(not _is_number(p) or not p > 0 for p in pitch_mm)
    ):
        raise SchemaError(f"{path}: pitch_mm must be two positive numbers [dy, dx], got {pitch_mm!r}")
    if area_mm2 is not None:
        _require_positive(area_mm2, f"{path}: area_mm2")


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
