"""A deterministic simulated sensor with injectable faults, for tests and for validating health checks.

Frames are a pure function of the seed and the frame's sequence number, so two sensors
built from the same config produce identical data, and enabling one fault does not change
the others. Timing is real: frames are paced at ``declared_rate_hz`` on the host clock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from taxel.schema import (
    ARRAY_FAMILIES,
    Layout,
    SchemaError,
    TactileFamily,
    TactileInfo,
    session_meta_path,
)
from taxel.sensors.base import TactileReading, TactileSensor, TactileSensorConfig, host_clock

__all__ = ["SimulatedTactile", "SimulatedTactileConfig"]

_UNITS = {
    TactileFamily.TAXEL_ARRAY: {"force": "N"},
    TactileFamily.FORCE_TORQUE: {"force": "N", "torque": "N.m"},
    TactileFamily.MAGNETIC: {"field": "uT"},
}


@TactileSensorConfig.register_subclass("simulated")
@dataclass(kw_only=True)
class SimulatedTactileConfig(TactileSensorConfig):
    """Configuration of the simulated sensor. Every fault is off by default.

    Attributes:
        n_elements: Number of sensing elements. Frames are ``(n_elements, len(channels))``.
        channels: Channel names, in the order of the last frame dimension.
        seed: Seed for the rest pattern, the noise and the drop decisions.
        rest_std: Spread of the fixed per-element no-contact pattern (the true baseline).
        noise_std: Spread of the per-frame Gaussian noise added to the rest pattern.
        drift_per_s: Fault. Linear ramp added per channel, in units per second of sensor time.
        dead_elements: Fault. Indices of elements pinned to ``dead_value`` in every frame.
        dead_value: The constant dead elements read.
        drop_probability: Fault. Probability that any given frame is dropped at the source:
            its ``seq`` is skipped and the base class counts it in ``dropped_frames``.
        clock_offset_s: Fault. Constant offset of ``device_ts`` from ``host_ts``.
        saturation_max: Fault. Values are clamped to ``[-saturation_max, saturation_max]``.
    """

    family: TactileFamily = TactileFamily.TAXEL_ARRAY
    declared_rate_hz: float = 100.0
    n_elements: int = 16
    channels: list[str] = field(default_factory=lambda: ["fx", "fy", "fz"])
    seed: int = 0
    rest_std: float = 0.05
    noise_std: float = 0.01
    drift_per_s: list[float] | None = None
    dead_elements: list[int] | None = None
    dead_value: float = 0.0
    drop_probability: float = 0.0
    clock_offset_s: float = 0.0
    saturation_max: float | None = None

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
        if self.drift_per_s is not None and len(self.drift_per_s) != len(self.channels):
            raise ValueError(f"drift_per_s needs one value per channel ({len(self.channels)})")
        if self.dead_elements is not None and any(not 0 <= i < self.n_elements for i in self.dead_elements):
            raise ValueError(f"dead_elements must index into range({self.n_elements})")
        if not 0.0 <= self.drop_probability < 1.0:
            raise ValueError(f"drop_probability must be in [0, 1), got {self.drop_probability}")
        if self.saturation_max is not None and not self.saturation_max > 0:
            raise ValueError(f"saturation_max must be positive, got {self.saturation_max}")

    @property
    def shape(self) -> tuple[int, int]:
        """Per-frame shape, ``(n_elements, len(channels))``."""
        return (self.n_elements, len(self.channels))


class SimulatedTactile(TactileSensor):
    """Simulated array sensor. See :class:`SimulatedTactileConfig` for the faults it can inject.

    Frame ``k`` (its ``seq``) is ``rest + noise_k + drift * k / rate``, clamped to the
    saturation limit, with dead elements overwritten last. The true baseline is :attr:`rest`.
    """

    config_class = SimulatedTactileConfig

    def __init__(self, config: SimulatedTactileConfig) -> None:
        super().__init__(config)
        self.config: SimulatedTactileConfig = config
        rng = np.random.default_rng([config.seed, 0])
        self._rest: NDArray[np.float64] = rng.normal(0.0, config.rest_std, config.shape)
        self._t0 = 0.0
        self._seq = -1

    @property
    def rest(self) -> NDArray[np.float32]:
        """The no-contact pattern every frame is built on: ground truth for a captured baseline."""
        return self._rest.astype(np.float32)

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
            layout=Layout(frame="sensor", positions_mm=positions, normals=[[0.0, 0.0, 1.0]] * cfg.n_elements),
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
        timeout_s = cfg.timeout_ms / 1000.0
        while True:
            due = self._t0 + (self._seq + 2) * period
            wait = due - host_clock()
            if wait > timeout_s:
                time.sleep(timeout_s)
                raise TimeoutError("next simulated frame is not due yet")
            if wait > 0:
                time.sleep(wait)
            self._seq += 1
            if cfg.drop_probability > 0 and self._is_dropped(self._seq):
                continue
            data = self._frame(self._seq)
            host_ts = host_clock()
            return TactileReading(
                data, host_ts=host_ts, device_ts=host_ts + cfg.clock_offset_s, seq=self._seq
            )

    def _is_dropped(self, seq: int) -> bool:
        """Whether frame ``seq`` is dropped at the source. A pure function of the seed and ``seq``."""
        return bool(np.random.default_rng([self.config.seed, 2, seq]).random() < self.config.drop_probability)

    def _frame(self, seq: int) -> NDArray[np.float32]:
        """Frame ``seq`` with every configured fault applied. A pure function of the seed and ``seq``."""
        cfg = self.config
        noise = np.random.default_rng([cfg.seed, 1, seq]).normal(0.0, cfg.noise_std, cfg.shape)
        data = self._rest + noise
        if cfg.drift_per_s is not None:
            data = data + np.asarray(cfg.drift_per_s) * (seq / cfg.declared_rate_hz)
        if cfg.saturation_max is not None:
            data = np.clip(data, -cfg.saturation_max, cfg.saturation_max)
        if cfg.dead_elements:
            data[cfg.dead_elements, :] = cfg.dead_value
        return data.astype(np.float32)
