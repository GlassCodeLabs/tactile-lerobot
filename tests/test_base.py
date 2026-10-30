"""Contract tests for ``taxel.sensors.base`` using a minimal fake backend."""

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from taxel.schema import Baseline, SchemaError, TactileFamily, TactileInfo, session_meta_path
from taxel.sensors.base import (
    TactileReading,
    TactileSensor,
    TactileSensorConfig,
    host_clock,
    make_tactile_sensor,
)


@dataclass(kw_only=True)
class FakeConfig(TactileSensorConfig):
    family: TactileFamily = TactileFamily.TAXEL_ARRAY
    declared_rate_hz: float = 200.0
    period_s: float = 0.005
    seqs: tuple[int, ...] | None = None  # sequence numbers to emit; None means no seq


class FakeSensor(TactileSensor):
    """Emits constant-valued frames at ``period_s`` from a background thread."""

    config_class = FakeConfig

    def __init__(self, config: FakeConfig) -> None:
        super().__init__(config)
        self.opened = 0
        self.closed = 0
        self._seq_iter: Iterator[int] | None = None
        self._n = 0

    def _open(self) -> None:
        self.opened += 1
        self._n = 0
        self._seq_iter = iter(self.config.seqs) if self.config.seqs is not None else None

    def _read_frame(self) -> TactileReading:
        time.sleep(self.config.period_s)
        seq = None
        if self._seq_iter is not None:
            try:
                seq = next(self._seq_iter)
            except StopIteration:
                raise TimeoutError("sequence exhausted") from None
        value = float(self._n)
        self._n += 1
        return TactileReading(np.full((4, 3), value, dtype=np.float32), host_clock(), 0.5 + value, seq)

    def _close(self) -> None:
        self.closed += 1

    def metadata(self) -> TactileInfo:
        return TactileInfo(
            family=self.config.family,
            declared_rate_hz=self.config.declared_rate_hz,
            session_meta=session_meta_path(self.config.instance_id),
            channels=["fx", "fy", "fz"],
        )

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        return [{"id": "fake_0", "type": "fake"}]


@pytest.fixture
def registry() -> Iterator[dict[str, type[TactileSensorConfig]]]:
    saved = dict(TactileSensorConfig._choice_registry)
    TactileSensorConfig._choice_registry.clear()
    try:
        yield TactileSensorConfig._choice_registry
    finally:
        TactileSensorConfig._choice_registry.clear()
        TactileSensorConfig._choice_registry.update(saved)


def fake(**overrides: Any) -> FakeSensor:
    return FakeSensor(FakeConfig(**{"instance_id": "left_fingertip", "timeout_ms": 500.0, **overrides}))


class TestRegistry:
    def test_decorator_registers_and_names(self, registry: dict) -> None:
        @TactileSensorConfig.register_subclass("fake")
        @dataclass(kw_only=True)
        class Registered(FakeConfig):
            pass

        assert TactileSensorConfig.get_choice_class("fake") is Registered
        assert TactileSensorConfig.get_known_choices() == {"fake": Registered}
        assert TactileSensorConfig.get_choice_name(Registered) == "fake"
        assert Registered(instance_id="x").type == "fake"

    def test_plain_call_registers(self, registry: dict) -> None:
        assert TactileSensorConfig.register_subclass("fake", FakeConfig) is FakeConfig
        assert TactileSensorConfig.register_subclass("fake", FakeConfig) is FakeConfig  # idempotent

    def test_duplicate_name_with_other_class_rejected(self, registry: dict) -> None:
        TactileSensorConfig.register_subclass("fake", FakeConfig)

        @dataclass(kw_only=True)
        class Other(FakeConfig):
            pass

        with pytest.raises(ValueError, match="already is"):
            TactileSensorConfig.register_subclass("fake", Other)

    def test_non_config_rejected(self, registry: dict) -> None:
        with pytest.raises(TypeError, match="not a TactileSensorConfig subclass"):
            TactileSensorConfig.register_subclass("fake", dict)

    def test_unknown_lookups(self, registry: dict) -> None:
        with pytest.raises(KeyError, match="no sensor type 'nope'"):
            TactileSensorConfig.get_choice_class("nope")
        with pytest.raises(ValueError, match="not a registered"):
            FakeConfig(instance_id="x").type  # noqa: B018

    def test_make_tactile_sensor_uses_config_class(self) -> None:
        sensor = make_tactile_sensor(FakeConfig(instance_id="palm_0"))
        assert isinstance(sensor, FakeSensor)
        assert sensor.config.instance_id == "palm_0"

    def test_make_tactile_sensor_unknown_config(self) -> None:
        @dataclass(kw_only=True)
        class Orphan(FakeConfig):
            pass

        with pytest.raises(KeyError, match="no TactileSensor declares config_class Orphan"):
            make_tactile_sensor(Orphan(instance_id="x"))


class TestConfig:
    def test_family_string_is_coerced(self) -> None:
        assert FakeConfig(instance_id="x", family="magnetic").family is TactileFamily.MAGNETIC

    def test_bad_family(self) -> None:
        with pytest.raises(SchemaError, match="family must be one of"):
            FakeConfig(instance_id="x", family="optical")

    def test_bad_instance_id(self) -> None:
        with pytest.raises(SchemaError, match="instance_id"):
            FakeConfig(instance_id="left/fingertip")

    @pytest.mark.parametrize("field", ["declared_rate_hz", "timeout_ms", "buffer_size"])
    def test_positive_numbers(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            FakeConfig(instance_id="x", **{field: 0})


class TestReading:
    def test_fields_and_frozen(self) -> None:
        r = TactileReading(np.zeros((2, 3), np.float32), host_ts=1.5, device_ts=None, seq=7)
        assert r.data.shape == (2, 3) and r.host_ts == 1.5 and r.device_ts is None and r.seq == 7
        with pytest.raises(AttributeError):
            r.seq = 8  # type: ignore[misc]

    def test_type_checks(self) -> None:
        with pytest.raises(TypeError, match="numpy array"):
            TactileReading([[0.0]], 1.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="host_ts"):
            TactileReading(np.zeros(1), None)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="seq"):
            TactileReading(np.zeros(1), 1.0, seq=1.5)  # type: ignore[arg-type]


class TestContract:
    def test_incomplete_backend_cannot_be_instantiated(self) -> None:
        class Partial(TactileSensor):
            def _open(self) -> None: ...
            def _close(self) -> None: ...
            def metadata(self) -> TactileInfo: ...  # type: ignore[empty-body]

        with pytest.raises(TypeError, match="abstract"):
            Partial(FakeConfig(instance_id="x"))  # type: ignore[abstract]

    def test_connect_disconnect_and_hooks(self) -> None:
        sensor = fake()
        assert not sensor.is_connected
        sensor.connect()
        assert sensor.is_connected and sensor.opened == 1 and sensor.frames_received >= 1
        with pytest.raises(ConnectionError, match="already connected"):
            sensor.connect()
        sensor.disconnect()
        assert not sensor.is_connected and sensor.closed == 1
        sensor.disconnect()  # idempotent
        assert sensor.closed == 1

    def test_context_manager(self) -> None:
        with fake() as sensor:
            assert sensor.is_connected
        assert not sensor.is_connected

    def test_reads_require_connection(self) -> None:
        sensor = fake()
        for call in (sensor.read, sensor.async_read, sensor.read_latest, sensor.drain):
            with pytest.raises(ConnectionError, match="not connected"):
                call()
        with pytest.raises(ConnectionError):
            sensor.capture_baseline(0.1)

    def test_warmup_timeout_disconnects(self) -> None:
        sensor = fake(seqs=(), timeout_ms=100.0)  # exhausted at once: never a frame
        with pytest.raises(TimeoutError, match="first frame"):
            sensor.connect()
        assert not sensor.is_connected and sensor.closed == 1

    def test_connect_without_warmup(self) -> None:
        sensor = fake(seqs=(), timeout_ms=100.0)
        sensor.connect(warmup=False)
        assert sensor.is_connected and sensor.frames_received == 0
        sensor.disconnect()

    def test_read_returns_a_frame_captured_after_the_call(self) -> None:
        with fake() as sensor:
            before = sensor.frames_received
            frame = sensor.read()
            assert float(frame.data[0, 0]) >= before
            assert frame.device_ts == pytest.approx(0.5 + float(frame.data[0, 0]))

    def test_async_read_never_repeats_and_skips_to_newest(self) -> None:
        with fake() as sensor:
            a = sensor.async_read()
            b = sensor.async_read()
            assert float(b.data[0, 0]) > float(a.data[0, 0])
            time.sleep(0.05)  # several frames arrive; the next call returns the newest, not the next
            c = sensor.async_read()
            assert float(c.data[0, 0]) - float(b.data[0, 0]) > 1

    def test_read_latest_peeks_and_repeats(self) -> None:
        with fake(period_s=0.05) as sensor:
            a = sensor.read_latest()
            b = sensor.read_latest()
            assert a is b
            with pytest.raises(TimeoutError, match="old"):
                sensor.read_latest(max_age_ms=0.0)

    def test_read_latest_before_any_frame(self) -> None:
        sensor = fake(seqs=(), timeout_ms=100.0)
        sensor.connect(warmup=False)
        with pytest.raises(RuntimeError, match="not received any frame"):
            sensor.read_latest()
        sensor.disconnect()

    def test_timeout_reports_backend_error(self) -> None:
        class Broken(FakeSensor):
            def _read_frame(self) -> TactileReading:
                time.sleep(0.005)
                raise OSError("bus fault")

        sensor = Broken(FakeConfig(instance_id="x", timeout_ms=100.0))
        with pytest.raises(TimeoutError, match=r"bus fault"):
            sensor.connect()
        assert not sensor.is_connected

    def test_drain_returns_every_frame_once_in_order(self) -> None:
        with fake() as sensor:
            time.sleep(0.05)
            first = sensor.drain()
            second = sensor.drain()
            values = [float(f.data[0, 0]) for f in first]
            assert values == list(range(len(values))) and len(values) >= 2
            assert all(float(f.data[0, 0]) > values[-1] for f in second)
            assert sensor.overflowed_frames == 0

    def test_drain_overflow_is_counted_and_keeps_newest(self) -> None:
        with fake(buffer_size=3) as sensor:
            time.sleep(0.06)
            frames = sensor.drain()
            assert len(frames) == 3
            assert sensor.overflowed_frames >= 1
            assert float(frames[-1].data[0, 0]) == sensor.frames_received - 1

    def test_seq_gaps_count_dropped_frames(self) -> None:
        with fake(seqs=(0, 1, 2, 5, 6, 10)) as sensor:
            time.sleep(0.1)
            assert sensor.dropped_frames == 2 + 3
            assert [f.seq for f in sensor.drain()] == [0, 1, 2, 5, 6, 10]

    def test_counters_reset_on_connect(self) -> None:
        sensor = fake(seqs=(0, 3), buffer_size=1)
        sensor.connect()
        time.sleep(0.1)
        assert sensor.dropped_frames == 2 and sensor.overflowed_frames >= 1
        sensor.disconnect()
        sensor.connect(warmup=False)
        assert sensor.dropped_frames == 0 and sensor.overflowed_frames == 0 and sensor.frames_received == 0
        sensor.disconnect()

    def test_backend_can_report_drops_directly(self) -> None:
        with fake() as sensor:
            sensor._count_dropped(4)
            assert sensor.dropped_frames == 4

    def test_capture_baseline_is_the_mean(self) -> None:
        with fake() as sensor:
            baseline = sensor.capture_baseline(0.05)
        assert isinstance(baseline, Baseline)
        assert baseline.applied is False and baseline.duration_s == 0.05
        assert baseline.captured_at.endswith("Z")
        values = np.asarray(baseline.values, dtype=np.float32)
        assert values.shape == (4, 3)
        assert values.min() == values.max()  # every element has the same value
        assert values[0, 0] > 0  # an average over several frames, not the first frame

    def test_capture_baseline_needs_frames(self) -> None:
        sensor = fake(seqs=(), timeout_ms=100.0)
        sensor.connect(warmup=False)
        with pytest.raises(TimeoutError, match="no frame received"):
            sensor.capture_baseline(0.05)
        sensor.disconnect()

    def test_metadata_is_valid_schema(self) -> None:
        info = fake().metadata()
        assert TactileInfo.from_dict(info.to_dict()) == info
        assert info.session_meta == "tactile/left_fingertip/meta.json"

    def test_find_sensors(self) -> None:
        assert FakeSensor.find_sensors()[0]["type"] == "fake"

    def test_thread_stops_on_disconnect(self) -> None:
        sensor = fake()
        sensor.connect()
        name = f"taxel-{sensor.config.instance_id}"
        assert any(t.name == name for t in threading.enumerate())
        sensor.disconnect()
        assert not any(t.name == name for t in threading.enumerate())
