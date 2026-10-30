"""The ``TactileSensor`` contract: config, readings, registry and the shared read loop.

The public method surface (``connect``, ``read``, ``async_read``, ``read_latest``,
``disconnect``, ``is_connected``, ``find_sensors``) and the config registry mirror
LeRobot's tactile sensor proposal (huggingface/lerobot#4469), which in turn mirrors
LeRobot's cameras. On top of that, every reading carries timestamps and a sequence
number, the sensor can describe itself (:meth:`TactileSensor.metadata`), capture a
no-contact baseline, and count the frames the device dropped.

A backend implements three hooks, ``_open``, ``_read_frame`` and ``_close``, plus
``metadata`` and ``find_sensors``. The base class owns the background thread, the
frame buffer, timeouts and dropped-frame accounting.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, TypeVar

import numpy as np
from numpy.typing import NDArray

from taxel.schema import Baseline, DeviceGroup, Mount, SchemaError, TactileFamily, TactileInfo, feature_key

__all__ = [
    "TactileReading",
    "TactileSensor",
    "TactileSensorConfig",
    "host_clock",
    "make_tactile_sensor",
]

logger = logging.getLogger(__name__)

_C = TypeVar("_C", bound="TactileSensorConfig")


def host_clock() -> float:
    """Host time in seconds for ``host_ts``: ``time.perf_counter()``, the clock LeRobot cameras stamp
    frames with, so tactile and video timestamps are directly comparable within one process.

    It is monotonic and has no epoch. Anchoring it to wall-clock time is the recorder's job.
    """
    return time.perf_counter()


@dataclass(kw_only=True)
class TactileSensorConfig:
    """Base configuration shared by every backend.

    Attributes:
        instance_id: Short slug naming this sensor instance, stable across sessions
            (``left_fingertip``). It becomes the feature key suffix and the sidecar directory.
        family: The sensing family from ``SCHEMA.md``. Backends normally fix this.
        declared_rate_hz: The rate the sensor is expected to deliver frames at.
        port: Device path or serial port, when the backend needs one.
        timeout_ms: How long reads wait for a frame before raising ``TimeoutError``.
        buffer_size: How many native-rate frames :meth:`TactileSensor.drain` can hold before
            the oldest are discarded (and counted in ``overflowed_frames``).
        mount: Where on the robot the sensor sits, copied into the static metadata.
        device: Grouping with other instances on one hub, copied into the static metadata.
    """

    instance_id: str
    family: TactileFamily
    declared_rate_hz: float
    port: str | None = None
    timeout_ms: float = 1000.0
    buffer_size: int = 4096
    mount: Mount | None = None
    device: DeviceGroup | None = None

    _choice_registry: ClassVar[dict[str, type[TactileSensorConfig]]] = {}

    def __post_init__(self) -> None:
        feature_key(self.instance_id)  # validates the slug
        try:
            self.family = TactileFamily(self.family)
        except ValueError:
            raise SchemaError(
                f"family must be one of {[f.value for f in TactileFamily]}, got {self.family!r}"
            ) from None
        if not self.declared_rate_hz > 0:
            raise ValueError(f"declared_rate_hz must be positive, got {self.declared_rate_hz}")
        if not self.timeout_ms > 0:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be at least 1, got {self.buffer_size}")

    # --- Registry, with the names draccus's ChoiceRegistry uses -------------------------------

    @classmethod
    def register_subclass(
        cls, name: str, choice_type: type[_C] | None = None
    ) -> type[_C] | Callable[[type[_C]], type[_C]]:
        """Register a config class under ``name``; usable as a decorator or a plain call."""
        if choice_type is None:
            return lambda subcls: cls.register_subclass(name, subcls)
        if not (isinstance(choice_type, type) and issubclass(choice_type, TactileSensorConfig)):
            raise TypeError(f"{choice_type!r} is not a TactileSensorConfig subclass")
        existing = cls._choice_registry.get(name)
        if existing is not None and existing is not choice_type:
            raise ValueError(f"cannot register {choice_type} as {name!r}: {existing} already is")
        cls._choice_registry[name] = choice_type
        return choice_type

    @classmethod
    def get_choice_class(cls, name: str) -> type[TactileSensorConfig]:
        """Return the config class registered under ``name``."""
        try:
            return cls._choice_registry[name]
        except KeyError:
            raise KeyError(f"no sensor type {name!r}; known: {sorted(cls._choice_registry)}") from None

    @classmethod
    def get_known_choices(cls) -> dict[str, type[TactileSensorConfig]]:
        """Return a copy of the name-to-class registry."""
        return dict(cls._choice_registry)

    @classmethod
    def get_choice_name(cls, subcls: type) -> str:
        """Return the name a config class was registered under."""
        for name, choice_type in cls._choice_registry.items():
            if choice_type is subcls:
                return name
        raise ValueError(f"{subcls} is not a registered TactileSensorConfig")

    @property
    def type(self) -> str:
        """The registered name of this config's class."""
        return self.get_choice_name(type(self))


@dataclass(frozen=True, slots=True)
class TactileReading:
    """One frame from a sensor, with the timing fields of the native-rate sidecar.

    Attributes:
        data: The frame. Float32 ``(n_elements, len(channels))`` for the array families;
            uint8 ``(H, W, 3)`` for a visuotactile image.
        host_ts: Host receive time from :func:`host_clock`, seconds.
        device_ts: The sensor's own timestamp in seconds, or ``None`` if it has no clock.
        seq: The sensor's own frame counter, or ``None``. A jump means the device dropped frames.
    """

    data: NDArray[Any]
    host_ts: float
    device_ts: float | None = None
    seq: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be a numpy array, got {type(self.data).__name__}")
        if isinstance(self.host_ts, bool) or not isinstance(self.host_ts, (int, float)):
            raise TypeError(f"host_ts must be a float, got {self.host_ts!r}")
        if self.device_ts is not None and not isinstance(self.device_ts, (int, float)):
            raise TypeError(f"device_ts must be a float or None, got {self.device_ts!r}")
        if self.seq is not None and (isinstance(self.seq, bool) or not isinstance(self.seq, int)):
            raise TypeError(f"seq must be an int or None, got {self.seq!r}")


class TactileSensor(abc.ABC):
    """Base class for every sensor backend.

    Lifecycle: :meth:`connect` opens the device and starts a background thread that
    calls :meth:`_read_frame` continuously; :meth:`disconnect` stops it. While
    connected, the three read methods offer different guarantees:

    - :meth:`read` blocks until a frame captured *after* the call arrives.
    - :meth:`async_read` returns the newest frame not yet returned by ``read`` or
      ``async_read``, waiting for one if needed.
    - :meth:`read_latest` returns the newest frame immediately, even if it was
      already returned, as long as it is not older than ``max_age_ms``.

    :meth:`drain` returns every frame received since the previous ``drain`` (or
    ``connect``), which is how a recorder captures the native-rate stream in full.

    Subclasses set ``config_class`` so :func:`make_tactile_sensor` can find them, and
    implement :meth:`_open`, :meth:`_read_frame`, :meth:`_close`, :meth:`metadata` and
    :meth:`find_sensors`.
    """

    config_class: ClassVar[type[TactileSensorConfig]]
    _sensor_classes: ClassVar[dict[type[TactileSensorConfig], type[TactileSensor]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        config_class = cls.__dict__.get("config_class")
        if config_class is not None:
            TactileSensor._sensor_classes[config_class] = cls

    def __init__(self, config: TactileSensorConfig) -> None:
        self.config = config
        self._connected = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: TactileReading | None = None
        self._received = 0
        self._consumed = 0
        self._buffer: deque[TactileReading] = deque(maxlen=config.buffer_size)
        self._last_seq: int | None = None
        self._dropped = 0
        self._overflowed = 0
        self._last_error: BaseException | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(instance_id={self.config.instance_id!r}, connected={self._connected})"

    def __enter__(self) -> TactileSensor:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    def __del__(self) -> None:
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # noqa: BLE001 - best effort during garbage collection
            pass

    # --- Backend hooks ------------------------------------------------------------------------

    @abc.abstractmethod
    def _open(self) -> None:
        """Open the device so that :meth:`_read_frame` can be called. Raise on failure."""

    @abc.abstractmethod
    def _read_frame(self) -> TactileReading:
        """Block until the next frame arrives and return it, stamped with :func:`host_clock`.

        Called only from the background thread. Must return or raise ``TimeoutError``
        within ``config.timeout_ms`` so that :meth:`disconnect` can stop the thread.
        Backends that detect gaps by a means other than ``seq`` call :meth:`_count_dropped`.
        """

    @abc.abstractmethod
    def _close(self) -> None:
        """Release the device. Called after the background thread has stopped."""

    @abc.abstractmethod
    def metadata(self) -> TactileInfo:
        """Return the static ``info`` block for this instance, valid per ``SCHEMA.md``."""

    @staticmethod
    @abc.abstractmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Describe the devices of this type visible on the host, one dict per device."""

    # --- Lifecycle ----------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True between a successful :meth:`connect` and :meth:`disconnect`."""
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        """Open the device and start the read thread.

        Resets ``dropped_frames``, ``overflowed_frames`` and the buffer. With ``warmup``,
        waits up to ``config.timeout_ms`` for the first frame and raises ``TimeoutError``
        (after disconnecting) if none arrives.
        """
        if self._connected:
            raise ConnectionError(f"{self} is already connected")
        self._open()
        with self._cond:
            self._latest = None
            self._received = self._consumed = 0
            self._buffer.clear()
            self._last_seq = None
            self._dropped = self._overflowed = 0
            self._last_error = None
        self._stop.clear()
        self._connected = True
        self._thread = threading.Thread(
            target=self._run, name=f"taxel-{self.config.instance_id}", daemon=True
        )
        self._thread.start()
        if warmup:
            try:
                self._wait_for(lambda: self._received > 0, self.config.timeout_ms, "first frame")
            except TimeoutError:
                self.disconnect()
                raise

    def disconnect(self) -> None:
        """Stop the read thread and release the device. Safe to call when not connected."""
        if not self._connected:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.config.timeout_ms / 1000.0 + 1.0)
            if self._thread.is_alive():
                logger.warning("%s: read thread did not stop; _read_frame must honour timeout_ms", self)
            self._thread = None
        try:
            self._close()
        finally:
            self._connected = False

    # --- Reading ------------------------------------------------------------------------------

    def read(self) -> TactileReading:
        """Return the first frame captured after this call, waiting up to ``config.timeout_ms``.

        Raises:
            ConnectionError: If not connected.
            TimeoutError: If no new frame arrives in time.
        """
        self._require_connected()
        with self._cond:
            start = self._received
        self._wait_for(lambda: self._received > start, self.config.timeout_ms, "a new frame")
        with self._cond:
            self._consumed = self._received
            return self._latest  # type: ignore[return-value]

    def async_read(self, timeout_ms: float | None = None) -> TactileReading:
        """Return the newest frame not yet returned by ``read`` or ``async_read``.

        Returns immediately if one is buffered, otherwise waits up to ``timeout_ms``
        (default ``config.timeout_ms``). Frames that arrived between calls are skipped,
        not queued; use :meth:`drain` to get every frame.

        Raises:
            ConnectionError: If not connected.
            TimeoutError: If no unconsumed frame arrives in time.
        """
        self._require_connected()
        timeout = self.config.timeout_ms if timeout_ms is None else timeout_ms
        self._wait_for(lambda: self._received > self._consumed, timeout, "an unconsumed frame")
        with self._cond:
            self._consumed = self._received
            return self._latest  # type: ignore[return-value]

    def read_latest(self, max_age_ms: float = 500.0) -> TactileReading:
        """Return the newest frame immediately, without waiting and without consuming it.

        Raises:
            ConnectionError: If not connected.
            RuntimeError: If no frame has been received yet.
            TimeoutError: If the newest frame is older than ``max_age_ms``.
        """
        self._require_connected()
        with self._cond:
            latest = self._latest
        if latest is None:
            raise RuntimeError(f"{self} has not received any frame yet")
        age_ms = (host_clock() - latest.host_ts) * 1000.0
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self}: latest frame is {age_ms:.0f} ms old (max {max_age_ms:.0f} ms)")
        return latest

    def drain(self) -> list[TactileReading]:
        """Return every frame received since the previous ``drain`` or ``connect``, oldest first.

        The buffer holds ``config.buffer_size`` frames; when it overflows the oldest are
        discarded and ``overflowed_frames`` grows. Draining does not affect ``async_read``.
        """
        self._require_connected()
        with self._cond:
            frames = list(self._buffer)
            self._buffer.clear()
        return frames

    # --- Health and provenance ------------------------------------------------------------------

    @property
    def dropped_frames(self) -> int:
        """Frames the device dropped before they reached the host, detected from ``seq`` gaps or
        reported by the backend. Reset by :meth:`connect`."""
        return self._dropped

    @property
    def overflowed_frames(self) -> int:
        """Frames discarded on the host because :meth:`drain` was not called often enough.
        Reset by :meth:`connect`."""
        return self._overflowed

    @property
    def frames_received(self) -> int:
        """Frames received since :meth:`connect`."""
        return self._received

    def capture_baseline(self, seconds: float) -> Baseline:
        """Average the frames received over ``seconds`` and return them as a :class:`Baseline`.

        Call with nothing touching the sensor. The result has ``applied=False``: nothing is
        subtracted from later readings. Backends whose frames are images override this to
        return a ``reference_frame`` instead.

        Raises:
            ConnectionError: If not connected.
            TimeoutError: If no frame arrives during the window.
        """
        self._require_connected()
        if seconds <= 0:
            raise ValueError(f"seconds must be positive, got {seconds}")
        captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        deadline = host_clock() + seconds
        total: NDArray[np.float64] | None = None
        count = 0
        while True:
            remaining_ms = (deadline - host_clock()) * 1000.0
            if remaining_ms <= 0:
                break
            try:
                frame = self.async_read(timeout_ms=remaining_ms)
            except TimeoutError:
                break
            if frame.data.ndim != 2:
                raise NotImplementedError(f"{self}: override capture_baseline for {frame.data.ndim}-D frames")
            total = frame.data.astype(np.float64) if total is None else total + frame.data
            count += 1
        if total is None:
            raise TimeoutError(f"{self}: no frame received in {seconds} s")
        mean = (total / count).astype(np.float32)
        return Baseline(captured_at=captured_at, duration_s=seconds, applied=False, values=mean.tolist())

    # --- Internals ------------------------------------------------------------------------------

    def _count_dropped(self, n: int) -> None:
        """Record ``n`` frames dropped at the source. For backends with a gap signal other than ``seq``."""
        with self._cond:
            self._dropped += n

    def _require_connected(self) -> None:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected; call connect() first")

    def _wait_for(self, predicate: Callable[[], bool], timeout_ms: float, what: str) -> None:
        with self._cond:
            if not self._cond.wait_for(predicate, timeout=timeout_ms / 1000.0):
                detail = f"; last backend error: {self._last_error!r}" if self._last_error else ""
                raise TimeoutError(f"{self}: no {what} within {timeout_ms:.0f} ms{detail}")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._read_frame()
            except TimeoutError:
                continue
            except Exception as e:  # noqa: BLE001 - keep the thread alive, surface the error on read
                if self._stop.is_set():
                    break
                self._last_error = e
                logger.warning("%s: _read_frame failed: %r", self, e)
                continue
            with self._cond:
                if frame.seq is not None:
                    if self._last_seq is not None and frame.seq > self._last_seq + 1:
                        self._dropped += frame.seq - self._last_seq - 1
                    self._last_seq = frame.seq
                if len(self._buffer) == self._buffer.maxlen:
                    self._overflowed += 1
                self._buffer.append(frame)
                self._latest = frame
                self._received += 1
                self._cond.notify_all()


def make_tactile_sensor(config: TactileSensorConfig) -> TactileSensor:
    """Instantiate the backend whose ``config_class`` matches ``type(config)``."""
    try:
        sensor_class = TactileSensor._sensor_classes[type(config)]
    except KeyError:
        known = sorted(c.__name__ for c in TactileSensor._sensor_classes)
        raise KeyError(
            f"no TactileSensor declares config_class {type(config).__name__}; known: {known}"
        ) from None
    return sensor_class(config)
