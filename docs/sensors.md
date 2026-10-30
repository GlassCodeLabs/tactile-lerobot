# Sensor backends

A backend is a `TactileSensor` subclass paired with a `TactileSensorConfig` subclass. The base class owns the background read thread, the frame buffer, timeouts and dropped-frame accounting; a backend implements the device-specific parts. `taxel/sensors/simulated.py` is the reference backend and is about 190 lines including its five injectable faults; a real adapter with a simple wire format is shorter.

The method surface (`connect`, `read`, `async_read`, `read_latest`, `disconnect`, `is_connected`, `find_sensors`) and the config registry follow LeRobot's tactile proposal, [huggingface/lerobot#4469](https://github.com/huggingface/lerobot/pull/4469), which itself follows LeRobot's cameras. What this package adds is timing on every reading, the sensor's own description of itself, a no-contact baseline, and native-rate buffering.

```python
from taxel.sensors import SimulatedTactile, SimulatedTactileConfig

config = SimulatedTactileConfig(instance_id="left_fingertip", n_elements=52, declared_rate_hz=91)
with SimulatedTactile(config) as sensor:
    frame = sensor.read()  # TactileReading: (52, 3) float32 + host_ts, device_ts, seq
    baseline = sensor.capture_baseline(2.0)
    frames = sensor.drain()  # every frame received since connect, at native rate
info = sensor.metadata()  # the static `info` block from SCHEMA.md
```

## The contract

`connect(warmup=True)` opens the device, resets every counter and the buffer, and starts the read thread. With `warmup` it waits up to `timeout_ms` for the first frame and raises `TimeoutError` after disconnecting if none arrives. Calling it twice raises `ConnectionError`. `disconnect()` stops the thread, releases the device, and is safe to call at any time. Both run through the context manager.

While connected, three methods return a `TactileReading` with different guarantees:

| Method | Returns | Waits | Consumes |
| --- | --- | --- | --- |
| `read()` | the first frame captured after the call | up to `timeout_ms` | yes |
| `async_read(timeout_ms=None)` | the newest frame not yet returned by `read` or `async_read` | only if none is pending | yes |
| `read_latest(max_age_ms=500)` | the newest frame, even if already returned | never | no |

"Consumes" means a frame is returned at most once across `read` and `async_read`. Frames that arrive between two `async_read` calls are skipped, not queued: `async_read` is for control loops that want the freshest frame. `read_latest` raises `RuntimeError` before the first frame and `TimeoutError` when the newest frame is older than `max_age_ms`. All three raise `ConnectionError` when not connected.

`drain()` is the fourth read and the one a recorder uses. It returns every frame received since the previous `drain()` or `connect()`, oldest first, from a buffer of `buffer_size` frames (default 4096). It works after `disconnect()` so the tail of a stream can be flushed. When the buffer overflows, the oldest frames are discarded and `overflowed_frames` grows; that is a host-side loss and is kept separate from `dropped_frames`.

## Readings

`TactileReading` is a frozen dataclass with the four columns of the native-rate sidecar in `SCHEMA.md`:

- `data`: the frame. Float32 `(n_elements, len(channels))` for the array families, uint8 `(H, W, 3)` for a visuotactile image.
- `host_ts`: host receive time from `host_clock()`, in seconds.
- `device_ts`: the sensor's own timestamp in seconds, or `None` when it has no clock.
- `seq`: the sensor's own frame counter, or `None`. A jump means the device dropped frames before they reached the host.

`host_clock()` is `time.perf_counter()`, the clock LeRobot's cameras stamp frames with, so tactile and video timestamps are comparable within one process. It is monotonic and has no epoch: two values from different processes or different runs cannot be compared, and a value on its own cannot be placed on a calendar. To make a sidecar's timestamps meaningful later, the session record must store the wall-clock time and the `host_clock()` value taken at `connect()`. Writing that pair is the dataset writer's responsibility, not the backend's.

## Counters

- `dropped_frames`: frames lost at the source. The base class detects them from gaps in `seq`; a backend with another gap signal calls `_count_dropped(n)`. Gap detection assumes `seq` only grows, so a backend whose device counter wraps (an 8-bit or 16-bit counter, say) must unwrap it into a monotonically increasing integer before building the reading.
- `overflowed_frames`: frames discarded on the host because `drain()` was not called often enough.
- `frames_received`: frames received since `connect()`.

All three reset on `connect()`. A later framerate check reads `dropped_frames` rather than reconstructing drops from timestamps.

## Baseline

`capture_baseline(seconds)` averages the frames received over `seconds` and returns a `Baseline` record from `taxel.schema` with `values` set to that mean, `captured_at` in UTC and `applied=False`. It subtracts nothing from later readings; whether and how a baseline is applied is recorded in the session record, never done silently. Call it with nothing touching the sensor. Backends whose frames are images override it to return a `reference_frame` pointer instead, which is what `SCHEMA.md` specifies for visuotactile sensors.

## Metadata

`metadata()` returns a `TactileInfo`, the static `info` block of the instance's feature entry, built from what the backend knows (family, vendor, model, channels, layout) and what the config supplies (`declared_rate_hz`, `mount`, `device`, `instance_id`). It validates itself on construction, so a backend that declares `taxel_array` without `channels` fails at `metadata()`, not at write time.

## Registry

Config classes register under a short name:

```python
@TactileSensorConfig.register_subclass("simulated")
@dataclass(kw_only=True)
class SimulatedTactileConfig(TactileSensorConfig):
    n_elements: int = 16
    channels: list[str] = field(default_factory=lambda: ["fx", "fy", "fz"])
    seed: int = 0
```

`TactileSensorConfig.get_choice_class(name)`, `get_known_choices()`, `get_choice_name(cls)` and the `type` property use the names draccus's `ChoiceRegistry` uses, so bridging to LeRobot's registry is a few lines, without this package importing draccus. The sensor class declares `config_class`, and `make_tactile_sensor(config)` instantiates the backend whose `config_class` matches the config's type.

## Writing a backend

Subclass both classes and implement five things. `_open`, `_read_frame` and `_close` are called by the base class; the first and last from the thread that calls `connect()` and `disconnect()`, `_read_frame` only from the read thread.

```python
from dataclasses import dataclass
from typing import Any

import numpy as np

from taxel.schema import TactileFamily, TactileInfo, session_meta_path
from taxel.sensors import TactileReading, TactileSensor, TactileSensorConfig, host_clock


@TactileSensorConfig.register_subclass("example")
@dataclass(kw_only=True)
class ExampleConfig(TactileSensorConfig):
    family: TactileFamily = TactileFamily.TAXEL_ARRAY
    declared_rate_hz: float = 100.0
    port: str = "/dev/ttyACM0"


class ExampleSensor(TactileSensor):
    config_class = ExampleConfig

    def _open(self) -> None:
        self._link = open_device(self.config.port, timeout_s=self.config.timeout_ms / 1000)

    def _read_frame(self) -> TactileReading:
        packet = self._link.next_packet()  # raises TimeoutError within timeout_ms
        host_ts = host_clock()  # stamp at receipt, before any decoding
        data = decode(packet).astype(np.float32)  # (n_elements, len(channels))
        return TactileReading(data, host_ts=host_ts, device_ts=packet.t, seq=packet.seq)

    def _close(self) -> None:
        self._link.close()

    def metadata(self) -> TactileInfo:
        return TactileInfo(
            family=self.config.family,
            vendor="Example",
            model="EX-1",
            declared_rate_hz=self.config.declared_rate_hz,
            units={"force": "N"},
            channels=["fx", "fy", "fz"],
            mount=self.config.mount,
            device=self.config.device,
            session_meta=session_meta_path(self.config.instance_id),
        )

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        return [{"id": p, "type": "example"} for p in list_ports()]
```

Rules the base class relies on:

- `_read_frame` must return or raise `TimeoutError` within `config.timeout_ms`. The read thread treats `TimeoutError` as "nothing yet" and keeps going; `disconnect()` waits roughly two timeouts for the thread and then logs a warning if it is still blocked.
- Take `host_ts` as early as possible after the bytes arrive, before decoding, so it measures arrival and not parsing.
- Give `seq` as a monotonically increasing integer or `None`. Unwrap a wrapping device counter.
- Any other exception from `_read_frame` is logged and remembered, the thread continues, and the next read that times out quotes it. A device that fails on every call therefore logs at the sensor's rate; an adapter should rate-limit that warning, for instance by raising once and then `TimeoutError` until the fault clears.
- Store nothing in the reading that is not from the device. Baseline subtraction, scaling and resampling belong to the recorder and are recorded in metadata.

Test a backend the way `tests/test_base.py` tests the base class: a fake device that yields scripted frames, no hardware. Use `SimulatedTactile` when a test needs a realistic stream with known faults: drift, dead elements, dropped frames, a clock offset and saturation are each switched on through its config, and every frame it produces is a pure function of the seed and `seq`, so expected values can be recomputed in a test.
