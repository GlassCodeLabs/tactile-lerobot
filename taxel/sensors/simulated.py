"""Deterministic simulated sensors with injectable faults and a scripted stimulus, for tests and for
validating health checks.

Frames are a pure function of the seed and the frame's sequence number, so two sensors
built from the same config produce identical data, and enabling one fault does not change
the others. Timing is real: frames are paced at ``declared_rate_hz`` on the host clock.
:class:`SimulatedTactile` is an array sensor and :class:`SimulatedVisuotactile` an image
sensor; both take the same press schedule, so a check that needs one contact visible in
two streams at a known time has it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from taxel.schema import (
    ARRAY_FAMILIES,
    Axes,
    Layout,
    Markers,
    Origin,
    Range,
    SchemaError,
    TactileFamily,
    TactileInfo,
    session_meta_path,
)
from taxel.sensors.base import (
    AppliedBaseline,
    TactileReading,
    TactileSensor,
    TactileSensorConfig,
    host_clock,
)

__all__ = [
    "Press",
    "SimulatedTactile",
    "SimulatedTactileConfig",
    "SimulatedVisuotactile",
    "SimulatedVisuotactileConfig",
]

Press = tuple[float, float, float]
"""A press: ``(start_s, duration_s, amplitude)``, start and duration in sensor time from ``connect()``."""

_UNITS = {
    TactileFamily.TAXEL_ARRAY: {"force": "N"},
    TactileFamily.FORCE_TORQUE: {"force": "N", "torque": "N.m"},
    TactileFamily.MAGNETIC: {"field": "uT"},
}


def _check_presses(schedule: Sequence[Sequence[float]] | None) -> list[Press] | None:
    """Validate a contact schedule and return it as tuples, or ``None`` when it is empty."""
    if not schedule:
        return None
    presses: list[Press] = []
    for i, press in enumerate(schedule):
        items = tuple(press) if isinstance(press, (list, tuple)) else ()
        if len(items) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in items):
            raise ValueError(f"contact_schedule[{i}] must be (start_s, duration_s, amplitude), got {press!r}")
        start, duration, amplitude = (float(v) for v in items)
        if start < 0 or duration <= 0:
            raise ValueError(f"contact_schedule[{i}]: start_s must be >= 0 and duration_s > 0, got {press!r}")
        presses.append((start, duration, amplitude))
    return presses


def _check_intervals(
    schedule: Sequence[Sequence[float]] | None, name: str
) -> list[tuple[float, float]] | None:
    """Validate ``(start_s, duration_s)`` intervals and return them as tuples, or ``None`` when empty."""
    if not schedule:
        return None
    intervals: list[tuple[float, float]] = []
    for i, interval in enumerate(schedule):
        items = tuple(interval) if isinstance(interval, (list, tuple)) else ()
        if len(items) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in items):
            raise ValueError(f"{name}[{i}] must be (start_s, duration_s), got {interval!r}")
        start, duration = float(items[0]), float(items[1])
        if start < 0 or duration <= 0:
            raise ValueError(f"{name}[{i}]: start_s must be >= 0 and duration_s > 0, got {interval!r}")
        intervals.append((start, duration))
    return intervals


def _interval_frames(schedule: Sequence[tuple[float, float]] | None, rate_hz: float) -> list[tuple[int, int]]:
    """Intervals in frames, ``(first, end)`` covering ``first <= seq < end``, rounded as presses are."""
    return [
        (first, end) for first, end, _ in _press_frames([(s, d, 0.0) for s, d in schedule or ()], rate_hz)
    ]


def _press_frames(schedule: Sequence[Press] | None, rate_hz: float) -> list[tuple[int, int, float]]:
    """The schedule in frames: ``(first, end, amplitude)`` per press, covering ``first <= seq < end``.

    Times are rounded to whole frames once, here, so a press boundary never depends on
    floating-point sums; a press lasts at least one frame.
    """
    frames = []
    for start, duration, amplitude in schedule or ():
        first = round(start * rate_hz)
        frames.append((first, max(first + 1, round((start + duration) * rate_hz)), amplitude))
    return frames


def _press_amplitude(presses: Sequence[tuple[int, int, float]], seq: int) -> float:
    """Sum of the amplitudes of the presses active at frame ``seq``."""
    return float(sum(a for first, end, a in presses if first <= seq < end))


def _wait_for_due(due: float, timeout_s: float) -> None:
    """Sleep until ``due`` on the host clock; ``TimeoutError`` when it is more than ``timeout_s`` away."""
    wait = due - host_clock()
    if wait > timeout_s:
        time.sleep(timeout_s)
        raise TimeoutError("next simulated frame is not due yet")
    if wait > 0:
        time.sleep(wait)


# --- Array sensor --------------------------------------------------------------------------------


@TactileSensorConfig.register_subclass("simulated")
@dataclass(kw_only=True)
class SimulatedTactileConfig(TactileSensorConfig):
    """Configuration of the simulated array sensor. Every fault is off by default.

    Attributes:
        n_elements: Number of sensing elements. Frames are ``(n_elements, len(channels))``.
        channels: Channel names, in the order of the last frame dimension.
        seed: Seed for the rest pattern, the noise and the drop decisions.
        rest_std: Spread of the fixed per-element no-contact pattern (the true baseline).
        noise_std: Spread of the per-frame Gaussian noise added to the rest pattern.
        contact_schedule: Stimulus. Presses as ``(start_s, duration_s, amplitude)`` in sensor time
            from ``connect()``; while a press is on, ``amplitude`` is added to every channel of
            ``contact_elements``. Overlapping presses add up.
        contact_elements: Elements a press acts on; default the first quarter of the array.
        drift_per_s: Fault. Linear ramp added per channel, in units per second of sensor time.
        dead_elements: Fault. Indices of elements pinned to ``dead_value`` in every frame.
        dead_value: The constant dead elements read.
        drop_probability: Fault. Probability that any given frame is dropped at the source:
            its ``seq`` is skipped and the base class counts it in ``dropped_frames``.
        clock_offset_s: Fault. Constant offset of ``device_ts`` from ``host_ts``.
        saturation_max: Fault. Values are clamped to ``[-saturation_max, saturation_max]``.
        declare_range: Whether ``metadata()`` declares ``[-saturation_max, saturation_max]`` as the
            sensor's ``range`` when a clamp is set; off, a check must infer the limit.
        rebaseline_at_s: Fault. Sensor times at which the backend re-baselines on its own, as a
            firmware re-zero would: from the first frame at or past each time it subtracts the
            rest pattern plus the drift so far, so stored frames return to zero while the sensor
            has not. Each re-baseline is stamped on the readings like a host baseline and is
            forgotten when the sensor reconnects.
        nan_schedule: Fault. ``(start_s, duration_s)`` intervals of sensor time in which
            ``nan_elements`` read NaN, the way a dataset marks a reading it lost or blanked.
        nan_elements: The elements NaN in those intervals; default every element.
        hold_schedule: Fault. ``(start_s, duration_s)`` intervals in which the sensor sends the
            frame from just before the interval again on ``hold_elements``, the way a reader with
            no new frame passes the last one on; ``seq`` and the timestamps keep advancing. An
            interval must start after the first frame.
        hold_elements: The elements held; default every element, a whole-frame hold.
    """

    family: TactileFamily = TactileFamily.TAXEL_ARRAY
    declared_rate_hz: float = 100.0
    n_elements: int = 16
    channels: list[str] = field(default_factory=lambda: ["fx", "fy", "fz"])
    seed: int = 0
    rest_std: float = 0.05
    noise_std: float = 0.01
    contact_schedule: list[Press] | None = None
    contact_elements: list[int] | None = None
    drift_per_s: list[float] | None = None
    dead_elements: list[int] | None = None
    dead_value: float = 0.0
    drop_probability: float = 0.0
    clock_offset_s: float = 0.0
    saturation_max: float | None = None
    declare_range: bool = True
    rebaseline_at_s: list[float] | None = None
    nan_schedule: list[tuple[float, float]] | None = None
    nan_elements: list[int] | None = None
    hold_schedule: list[tuple[float, float]] | None = None
    hold_elements: list[int] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family not in ARRAY_FAMILIES:
            raise SchemaError(f"the simulated sensor produces array families only, not {self.family.value!r}")
        if self.n_elements < 1:
            raise ValueError(f"n_elements must be at least 1, got {self.n_elements}")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError(f"channels must be non-empty and unique, got {self.channels!r}")
        if self.family is TactileFamily.FORCE_TORQUE and (self.n_elements, len(self.channels)) != (1, 6):
            raise SchemaError("force_torque frames are (1, 6): one element, six channels")
        if self.rest_std < 0 or self.noise_std < 0:
            raise ValueError("rest_std and noise_std must be non-negative")
        self.contact_schedule = _check_presses(self.contact_schedule)
        if self.contact_elements is not None and (
            not self.contact_elements or any(not 0 <= i < self.n_elements for i in self.contact_elements)
        ):
            raise ValueError(f"contact_elements must be a non-empty subset of range({self.n_elements})")
        if self.drift_per_s is not None and len(self.drift_per_s) != len(self.channels):
            raise ValueError(f"drift_per_s needs one value per channel ({len(self.channels)})")
        if self.dead_elements is not None and any(not 0 <= i < self.n_elements for i in self.dead_elements):
            raise ValueError(f"dead_elements must index into range({self.n_elements})")
        if not 0.0 <= self.drop_probability < 1.0:
            raise ValueError(f"drop_probability must be in [0, 1), got {self.drop_probability}")
        if self.saturation_max is not None and not self.saturation_max > 0:
            raise ValueError(f"saturation_max must be positive, got {self.saturation_max}")
        times = self.rebaseline_at_s
        if times is not None and (any(t < 0 for t in times) or list(times) != sorted(times)):
            raise ValueError(f"rebaseline_at_s must be non-negative and increasing, got {times!r}")
        self.nan_schedule = _check_intervals(self.nan_schedule, "nan_schedule")
        self.hold_schedule = _check_intervals(self.hold_schedule, "hold_schedule")
        if any(first < 1 for first, _ in _interval_frames(self.hold_schedule, self.declared_rate_hz)):
            raise ValueError(
                "hold_schedule: a hold repeats the frame before it, so it cannot start at frame 0"
            )
        for name in ("nan_elements", "hold_elements"):
            chosen = getattr(self, name)
            if chosen is not None and (not chosen or any(not 0 <= i < self.n_elements for i in chosen)):
                raise ValueError(f"{name} must be a non-empty subset of range({self.n_elements})")

    @property
    def shape(self) -> tuple[int, int]:
        """Per-frame shape, ``(n_elements, len(channels))``."""
        return (self.n_elements, len(self.channels))

    @property
    def pressed_elements(self) -> list[int]:
        """The elements a press acts on: ``contact_elements``, or the first quarter of the array."""
        if self.contact_elements is not None:
            return list(self.contact_elements)
        return list(range(max(1, self.n_elements // 4)))


class SimulatedTactile(TactileSensor):
    """Simulated array sensor. See :class:`SimulatedTactileConfig` for its faults and its stimulus.

    Frame ``k`` (its ``seq``) is ``rest + noise_k + drift * k / rate``, plus the press amplitude
    on the pressed elements, clamped to the saturation limit, with dead elements overwritten
    last. The true baseline is :attr:`rest`; a scheduled re-baseline absorbs :meth:`drifted_rest`.
    """

    config_class = SimulatedTactileConfig

    def __init__(self, config: SimulatedTactileConfig) -> None:
        super().__init__(config)
        self.config: SimulatedTactileConfig = config
        rng = np.random.default_rng([config.seed, 0])
        self._rest: NDArray[np.float64] = rng.normal(0.0, config.rest_std, config.shape)
        self._presses = _press_frames(config.contact_schedule, config.declared_rate_hz)
        self._nans = _interval_frames(config.nan_schedule, config.declared_rate_hz)
        self._holds = _interval_frames(config.hold_schedule, config.declared_rate_hz)
        self._rebaseline_seqs = [round(t * config.declared_rate_hz) for t in config.rebaseline_at_s or ()]
        self._t0 = 0.0
        self._seq = -1
        self._next_rebaseline = 0
        self._own_baseline: AppliedBaseline | None = None

    @property
    def rest(self) -> NDArray[np.float32]:
        """The no-contact pattern every frame is built on: ground truth for a captured baseline."""
        return self._rest.astype(np.float32)

    def drifted_rest(self, seq: int) -> NDArray[np.float32]:
        """The rest pattern plus the drift accumulated by frame ``seq``: what a re-baseline there absorbs."""
        cfg = self.config
        values = self._rest
        if cfg.drift_per_s is not None:
            values = values + np.asarray(cfg.drift_per_s) * (seq / cfg.declared_rate_hz)
        return values.astype(np.float32)

    def press_amplitude(self, seq: int) -> float:
        """Amplitude added to the pressed elements in frame ``seq``; ``0.0`` outside every press.

        A press scheduled at ``start_s`` for ``duration_s`` covers the frames from
        ``round(start_s * rate)`` up to, not including, ``round((start_s + duration_s) * rate)``.
        """
        return _press_amplitude(self._presses, seq)

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        return [{"id": "simulated", "type": "simulated"}]

    def metadata(self) -> TactileInfo:
        cfg = self.config
        pitch_mm = 1.5
        positions = [[pitch_mm * (i % 8), pitch_mm * (i // 8), 0.0] for i in range(cfg.n_elements)]
        return TactileInfo(
            family=cfg.family,
            vendor="taxel",
            model="simulated",
            declared_rate_hz=cfg.declared_rate_hz,
            units=dict(_UNITS[cfg.family]),
            channels=list(cfg.channels),
            range=(
                Range(min=-cfg.saturation_max, max=cfg.saturation_max)
                if cfg.saturation_max is not None and cfg.declare_range
                else None
            ),
            layout=Layout(
                origin=Origin(at="corner", description="element 0, the corner of the grid"),
                axes=Axes(
                    x="along a row of the 8-wide grid, from element 0 toward element 7",
                    y="down the grid columns, from element 0 toward element 8",
                    z="out of the sensing surface, toward the contacting object",
                ),
                surface="pad",
                positions_mm=positions,
                normals=[[0.0, 0.0, 1.0]] * cfg.n_elements,
            ),
            mount=cfg.mount,
            device=cfg.device,
            session_meta=session_meta_path(cfg.instance_id),
        )

    # --- Backend hooks ------------------------------------------------------------------------

    def _open(self) -> None:
        self._t0 = host_clock()
        self._seq = -1
        self._next_rebaseline = 0
        # A re-baseline the device applied itself does not survive its restart; a host baseline persists.
        if self._own_baseline is not None and self.applied_baseline is self._own_baseline:
            self.clear_baseline()
        self._own_baseline = None

    def _close(self) -> None:
        pass

    def _read_frame(self) -> TactileReading:
        cfg = self.config
        period = 1.0 / cfg.declared_rate_hz
        while True:
            _wait_for_due(self._t0 + (self._seq + 2) * period, cfg.timeout_ms / 1000.0)
            self._seq += 1
            if cfg.drop_probability > 0 and self._is_dropped(self._seq):
                continue
            self._rebaseline_if_due(self._seq)
            data = self._frame(self._seq)
            host_ts = host_clock()
            return TactileReading(
                data, host_ts=host_ts, device_ts=host_ts + cfg.clock_offset_s, seq=self._seq
            )

    def _rebaseline_if_due(self, seq: int) -> None:
        """Re-baseline at the first produced frame at or past each scheduled time, like a firmware re-zero."""
        seqs = self._rebaseline_seqs
        due = False
        while self._next_rebaseline < len(seqs) and seq >= seqs[self._next_rebaseline]:
            self._next_rebaseline += 1
            due = True
        if due:
            self._own_baseline = self.apply_baseline(self.drifted_rest(seq), (seq, seq))

    def _is_dropped(self, seq: int) -> bool:
        """Whether frame ``seq`` is dropped at the source. A pure function of the seed and ``seq``."""
        return bool(np.random.default_rng([self.config.seed, 2, seq]).random() < self.config.drop_probability)

    def _frame(self, seq: int) -> NDArray[np.float32]:
        """Frame ``seq`` with the stimulus and every fault applied; pure in the seed and ``seq``.

        A hold replaces the frame, or its ``hold_elements``, with the frame before the hold; NaN
        is written last, over everything else.
        """
        cfg = self.config
        data = self._live_frame(seq)
        for first, end in self._holds:
            if first <= seq < end:
                held = self._live_frame(first - 1)
                if cfg.hold_elements is None:
                    data = held
                else:
                    data[cfg.hold_elements] = held[cfg.hold_elements]
        for first, end in self._nans:
            if first <= seq < end:
                data[slice(None) if cfg.nan_elements is None else cfg.nan_elements] = np.nan
        return data

    def _live_frame(self, seq: int) -> NDArray[np.float32]:
        """Frame ``seq`` with the stimulus and the faults that act on every frame, before holds and NaN."""
        cfg = self.config
        noise = np.random.default_rng([cfg.seed, 1, seq]).normal(0.0, cfg.noise_std, cfg.shape)
        data = self._rest + noise
        if cfg.drift_per_s is not None:
            data = data + np.asarray(cfg.drift_per_s) * (seq / cfg.declared_rate_hz)
        amplitude = self.press_amplitude(seq)
        if amplitude:
            data[cfg.pressed_elements, :] += amplitude
        if cfg.saturation_max is not None:
            data = np.clip(data, -cfg.saturation_max, cfg.saturation_max)
        if cfg.dead_elements:
            data[cfg.dead_elements, :] = cfg.dead_value
        return data.astype(np.float32)


# --- Image sensor --------------------------------------------------------------------------------


@TactileSensorConfig.register_subclass("simulated_visuotactile")
@dataclass(kw_only=True)
class SimulatedVisuotactileConfig(TactileSensorConfig):
    """Configuration of the simulated image sensor. The press schedule is its stimulus; the dead
    pixels and the transport delay are faults, off by default.

    Attributes:
        height: Image height; frames are uint8 ``(height, width, 3)``.
        width: Image width.
        marker_grid: Rows and columns of the dark marker dots drawn on the gel.
        seed: Seed for the per-frame noise.
        noise_std: Spread of the Gaussian noise added to every pixel, in 8-bit counts. The default
            is the order of a real camera's noise plus lighting jitter, and well above what video
            encoding leaves on a constant region, so a dead pixel stays detectable after encoding.
        contact_schedule: Stimulus. Presses as ``(start_s, duration_s, amplitude)`` in sensor time
            from ``connect()``; a press brightens a round patch at ``contact_center`` by
            ``amplitude`` counts at its peak.
        contact_center: Centre of the patch as fractions of the height and the width.
        contact_radius: Standard deviation of the patch's Gaussian profile, as a fraction of half the width.
        dead_pixels: Fault. ``(row, col, height, width)`` of a block pinned to ``dead_value`` in every frame.
        dead_value: The constant the dead block reads.
        transport_delay_s: Fault. The frame delivered at host time ``t`` shows the sensor at
            ``t - transport_delay_s`` (rounded to whole frames), and ``device_ts`` is
            ``host_ts - transport_delay_s``, so a press appears in the images that much later than
            in an array stream on the same schedule. Unlike the array sensor's ``clock_offset_s``,
            it moves the content, not only the timestamp.
    """

    family: TactileFamily = TactileFamily.VISUOTACTILE
    declared_rate_hz: float = 30.0
    height: int = 48
    width: int = 64
    marker_grid: tuple[int, int] = (4, 6)
    seed: int = 0
    noise_std: float = 6.0
    contact_schedule: list[Press] | None = None
    contact_center: tuple[float, float] = (0.5, 0.5)
    contact_radius: float = 0.2
    dead_pixels: tuple[int, int, int, int] | None = None
    dead_value: int = 0
    transport_delay_s: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family is not TactileFamily.VISUOTACTILE:
            raise SchemaError(f"the simulated image sensor is visuotactile, not {self.family.value!r}")
        if self.height < 8 or self.width < 8:
            raise ValueError(f"height and width must be at least 8, got {self.height}x{self.width}")
        grid = tuple(self.marker_grid)
        if len(grid) != 2 or any(not isinstance(g, int) or g < 1 for g in grid):
            raise ValueError(f"marker_grid must be two positive integers, got {self.marker_grid!r}")
        if grid[0] > self.height or grid[1] > self.width:
            raise ValueError(f"marker_grid {grid} does not fit a {self.height}x{self.width} image")
        self.marker_grid = (int(grid[0]), int(grid[1]))
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        self.contact_schedule = _check_presses(self.contact_schedule)
        center = tuple(self.contact_center)
        if len(center) != 2 or any(not 0.0 <= c <= 1.0 for c in center):
            raise ValueError(f"contact_center must be two fractions in [0, 1], got {self.contact_center!r}")
        self.contact_center = (float(center[0]), float(center[1]))
        if not self.contact_radius > 0:
            raise ValueError(f"contact_radius must be positive, got {self.contact_radius}")
        if self.dead_pixels is not None:
            block = tuple(self.dead_pixels)
            if len(block) != 4 or any(not isinstance(v, int) for v in block):
                raise ValueError(f"dead_pixels must be (row, col, height, width), got {self.dead_pixels!r}")
            row, col, h, w = block
            if row < 0 or col < 0 or h < 1 or w < 1 or row + h > self.height or col + w > self.width:
                raise ValueError(f"dead_pixels {block} does not fit a {self.height}x{self.width} image")
            self.dead_pixels = (row, col, h, w)
        if not 0 <= self.dead_value <= 255:
            raise ValueError(f"dead_value must be an 8-bit value, got {self.dead_value}")
        if self.transport_delay_s < 0:
            raise ValueError(f"transport_delay_s must be non-negative, got {self.transport_delay_s}")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Per-frame shape, ``(height, width, 3)``."""
        return (self.height, self.width, 3)


class SimulatedVisuotactile(TactileSensor):
    """Simulated image sensor. See :class:`SimulatedVisuotactileConfig` for its stimulus and faults.

    Frame ``k`` is a fixed gel image (a colour gradient with a grid of dark markers), plus a
    bright patch while a press is on, plus noise, with the dead block overwritten last. The
    image is uint8 ``(height, width, 3)``; the no-contact image without noise is :attr:`background`.
    Baselines are not subtracted from images: :meth:`apply_baseline` raises.
    """

    config_class = SimulatedVisuotactileConfig

    def __init__(self, config: SimulatedVisuotactileConfig) -> None:
        super().__init__(config)
        self.config: SimulatedVisuotactileConfig = config
        h, w = config.height, config.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        y, x = yy / (h - 1), xx / (w - 1)
        background = np.stack([110 + 40 * x, 120 + 30 * y, 150 - 30 * x], axis=-1)
        rows, cols = config.marker_grid
        radius = max(1.0, min(h / rows, w / cols) / 6)
        for i in range(rows):
            for j in range(cols):
                cy, cx = (i + 0.5) * h / rows, (j + 0.5) * w / cols
                background[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] -= 70
        self._background: NDArray[np.float64] = np.rint(background)
        cy, cx = config.contact_center[0] * (h - 1), config.contact_center[1] * (w - 1)
        sigma = config.contact_radius * w / 2
        self._blob: NDArray[np.float64] = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        self._presses = _press_frames(config.contact_schedule, config.declared_rate_hz)
        self._delay = round(config.transport_delay_s * config.declared_rate_hz)
        self._t0 = 0.0
        self._seq = -1

    @property
    def background(self) -> NDArray[np.uint8]:
        """The no-contact image without noise or dead pixels: ground truth for a reference frame."""
        return np.clip(self._background, 0, 255).astype(np.uint8)

    def press_amplitude(self, seq: int) -> float:
        """Brightness added at the patch centre in frame ``seq``: the press active at the frame it
        shows, ``seq - round(transport_delay_s * rate)``; ``0.0`` outside every press."""
        return _press_amplitude(self._presses, seq - self._delay)

    def apply_baseline(
        self, values: ArrayLike, frame_range: tuple[int, int] | None = None
    ) -> AppliedBaseline:
        raise NotImplementedError(
            "baseline subtraction is defined for the array families; an image baseline is a reference frame"
        )

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        return [{"id": "simulated_visuotactile", "type": "simulated_visuotactile"}]

    def metadata(self) -> TactileInfo:
        cfg = self.config
        rows, cols = cfg.marker_grid
        return TactileInfo(
            family=TactileFamily.VISUOTACTILE,
            vendor="taxel",
            model="simulated_visuotactile",
            declared_rate_hz=cfg.declared_rate_hz,
            layout=Layout(
                origin=Origin(at="corner", description="the top-left pixel of the image"),
                axes=Axes(
                    x="along an image row, toward the right",
                    y="down an image column",
                    z="out of the gel, toward the contacting object",
                ),
                surface="gel",
            ),
            markers=Markers(count=rows * cols, grid=[rows, cols]),
            mount=cfg.mount,
            device=cfg.device,
            session_meta=session_meta_path(cfg.instance_id),
        )

    # --- Backend hooks ------------------------------------------------------------------------

    def _open(self) -> None:
        self._t0 = host_clock()
        self._seq = -1

    def _close(self) -> None:
        pass

    def _read_frame(self) -> TactileReading:
        cfg = self.config
        period = 1.0 / cfg.declared_rate_hz
        _wait_for_due(self._t0 + (self._seq + 2) * period, cfg.timeout_ms / 1000.0)
        self._seq += 1
        image = self._frame(self._seq)
        host_ts = host_clock()
        return TactileReading(
            image, host_ts=host_ts, device_ts=host_ts - cfg.transport_delay_s, seq=self._seq
        )

    def _frame(self, seq: int) -> NDArray[np.uint8]:
        """Frame ``seq`` with the stimulus and every fault applied; pure in the seed and ``seq``."""
        cfg = self.config
        image = self._background.copy()
        amplitude = self.press_amplitude(seq)
        if amplitude:
            image += amplitude * self._blob[..., None]
        if cfg.noise_std > 0:
            image += np.random.default_rng([cfg.seed, 1, seq]).normal(0.0, cfg.noise_std, image.shape)
        out = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        if cfg.dead_pixels is not None:
            row, col, h, w = cfg.dead_pixels
            out[row : row + h, col : col + w] = cfg.dead_value
        return out
