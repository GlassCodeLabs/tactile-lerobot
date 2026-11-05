"""Tests for the simulated backend: determinism, rate, and each injectable fault."""

import time
from typing import Any

import numpy as np
import pytest

from taxel.schema import SchemaError, TactileFamily, TactileInfo
from taxel.sensors import (
    SimulatedTactile,
    SimulatedTactileConfig,
    TactileReading,
    TactileSensorConfig,
    make_tactile_sensor,
)

RATE = 500.0


def config(**overrides: Any) -> SimulatedTactileConfig:
    base: dict[str, Any] = {"instance_id": "left_fingertip", "declared_rate_hz": RATE, "timeout_ms": 500.0}
    return SimulatedTactileConfig(**{**base, **overrides})


def collect(cfg: SimulatedTactileConfig, seconds: float) -> tuple[SimulatedTactile, list[TactileReading]]:
    """Connect, run for ``seconds``, disconnect, and return the sensor plus every frame received."""
    sensor = SimulatedTactile(cfg)
    sensor.connect()
    time.sleep(seconds)
    sensor.disconnect()
    frames = sensor.drain()
    assert frames, "no frames received"
    return sensor, frames


def by_seq(frames: list[TactileReading]) -> dict[int, np.ndarray]:
    return {f.seq: f.data for f in frames if f.seq is not None}


class TestRegistryAndShape:
    def test_registered_and_discoverable(self) -> None:
        assert config().type == "simulated"
        assert TactileSensorConfig.get_choice_class("simulated") is SimulatedTactileConfig
        assert isinstance(make_tactile_sensor(config()), SimulatedTactile)
        assert SimulatedTactile.find_sensors()[0]["type"] == "simulated"

    def test_frame_shape_and_dtype(self) -> None:
        cfg = config(n_elements=52, channels=["fx", "fy", "fz"])
        with SimulatedTactile(cfg) as sensor:
            frame = sensor.read()
        assert frame.data.shape == (52, 3) and frame.data.dtype == np.float32
        assert cfg.shape == (52, 3)

    def test_force_torque_shape(self) -> None:
        cfg = config(family="force_torque", n_elements=1, channels=["fx", "fy", "fz", "mx", "my", "mz"])
        with SimulatedTactile(cfg) as sensor:
            assert sensor.read().data.shape == (1, 6)
        with pytest.raises(SchemaError, match=r"\(1, 6\)"):
            config(family="force_torque")

    def test_metadata_is_valid_and_matches_config(self) -> None:
        cfg = config(family="magnetic", n_elements=5, channels=["bx", "by", "bz"])
        info = SimulatedTactile(cfg).metadata()
        assert TactileInfo.from_dict(info.to_dict()) == info
        assert info.family is TactileFamily.MAGNETIC
        assert info.units == {"field": "uT"}
        assert info.channels == ["bx", "by", "bz"]
        assert info.layout is not None and len(info.layout.positions_mm) == 5
        assert info.layout.origin.at == "corner" and info.layout.surface == "pad"
        assert info.session_meta == "tactile/left_fingertip/meta.json"
        assert info.declared_rate_hz == RATE

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"family": "visuotactile"}, "array families only"),
            ({"n_elements": 0}, "n_elements"),
            ({"channels": ["fx", "fx"]}, "unique"),
            ({"drift_per_s": [1.0]}, "one value per channel"),
            ({"dead_elements": [16]}, "dead_elements"),
            ({"drop_probability": 1.0}, "drop_probability"),
            ({"saturation_max": 0.0}, "saturation_max"),
            ({"noise_std": -1.0}, "non-negative"),
            ({"contact_schedule": [(0.0, 0.0, 1.0)]}, "duration_s > 0"),
            ({"contact_schedule": [(-1.0, 1.0, 1.0)]}, "start_s must be >= 0"),
            ({"contact_schedule": [(0.0, 1.0)]}, "contact_schedule\\[0\\]"),
            ({"contact_elements": [16]}, "contact_elements"),
            ({"contact_elements": []}, "contact_elements"),
            ({"rebaseline_at_s": [1.0, 0.5]}, "increasing"),
            ({"rebaseline_at_s": [-1.0]}, "non-negative"),
        ],
    )
    def test_config_validation(self, overrides: dict, match: str) -> None:
        with pytest.raises((ValueError, SchemaError), match=match):
            config(**overrides)

    def test_metadata_declares_the_clamp_as_the_range(self) -> None:
        assert SimulatedTactile(config()).metadata().range is None
        declared = SimulatedTactile(config(saturation_max=2.0)).metadata().range
        assert declared is not None and (declared.min, declared.max) == (-2.0, 2.0)
        assert SimulatedTactile(config(saturation_max=2.0, declare_range=False)).metadata().range is None
        info = SimulatedTactile(config(saturation_max=2.0)).metadata()
        assert TactileInfo.from_dict(info.to_dict()) == info


class TestDeterminism:
    def test_same_seed_same_frames(self) -> None:
        _, a = collect(config(seed=7), 0.05)
        _, b = collect(config(seed=7), 0.05)
        shared = by_seq(a).keys() & by_seq(b).keys()
        assert len(shared) >= 5
        for seq in shared:
            np.testing.assert_array_equal(by_seq(a)[seq], by_seq(b)[seq])

    def test_different_seed_different_frames(self) -> None:
        _, a = collect(config(seed=1), 0.02)
        _, b = collect(config(seed=2), 0.02)
        assert not np.array_equal(a[0].data, b[0].data)

    def test_reconnect_restarts_the_sequence(self) -> None:
        sensor = SimulatedTactile(config(seed=3))
        sensor.connect()
        first = sensor.read()
        sensor.disconnect()
        sensor.connect()
        again = sensor.drain()[0]
        sensor.disconnect()
        assert again.seq == 0
        np.testing.assert_array_equal(first.data, sensor._frame(first.seq))
        np.testing.assert_array_equal(again.data, sensor._frame(0))


class TestRateAndSequence:
    def test_rate_seq_and_counters_without_faults(self) -> None:
        sensor, frames = collect(config(), 0.5)
        n = len(frames)
        assert 0.6 * RATE * 0.5 < n < 1.3 * RATE * 0.5, n
        seqs = [f.seq for f in frames]
        assert seqs == list(range(seqs[0], seqs[0] + n))
        assert sensor.dropped_frames == 0 and sensor.overflowed_frames == 0
        periods = np.diff([f.host_ts for f in frames])
        assert abs(float(np.median(periods)) - 1 / RATE) < 0.5 / RATE

    def test_device_ts_equals_host_ts_without_offset(self) -> None:
        _, frames = collect(config(), 0.02)
        assert all(f.device_ts == f.host_ts for f in frames)

    def test_slow_sensor_honours_timeout(self) -> None:
        cfg = config(declared_rate_hz=0.5, timeout_ms=100.0)  # a frame every 2 s
        sensor = SimulatedTactile(cfg)
        with pytest.raises(TimeoutError, match="first frame"):
            sensor.connect()
        assert not sensor.is_connected


class TestFaults:
    def test_drift_is_a_linear_ramp_per_channel(self) -> None:
        clean = SimulatedTactile(config(seed=5))
        drifting = SimulatedTactile(config(seed=5, drift_per_s=[0.0, 0.0, 2.0]))
        for seq in (0, 10, 250):
            delta = drifting._frame(seq) - clean._frame(seq)
            expected = np.zeros(clean.config.shape, np.float32)
            expected[:, 2] = 2.0 * seq / RATE
            np.testing.assert_allclose(delta, expected, atol=1e-5)
            # the noise streams advanced identically, so the pair stays comparable at every seq

    def test_drift_absent_when_off(self) -> None:
        a, b = SimulatedTactile(config(seed=5)), SimulatedTactile(config(seed=5))
        np.testing.assert_array_equal(a._frame(100), b._frame(100))
        np.testing.assert_array_equal(a._frame(100), a._frame(100))  # pure in seq
        assert not np.allclose(a._frame(100), a._frame(0))  # noise differs between frames

    def test_dead_elements_are_pinned(self) -> None:
        _, frames = collect(config(dead_elements=[2, 5], dead_value=-1.0), 0.05)
        stacked = np.stack([f.data for f in frames])
        assert np.all(stacked[:, [2, 5], :] == -1.0)
        assert np.any(stacked[:, 0, :] != -1.0)
        _, clean = collect(config(), 0.02)
        assert not np.all(clean[0].data[[2, 5], :] == -1.0)

    def test_dropped_frames_leave_seq_gaps_and_are_counted(self) -> None:
        sensor, frames = collect(config(drop_probability=0.3, seed=11), 0.5)
        seqs = [f.seq for f in frames]
        assert len(seqs) < seqs[-1] - seqs[0] + 1, "no frame was dropped"
        assert sensor.dropped_frames == seqs[-1] - seqs[0] + 1 - len(seqs)
        assert sensor.dropped_frames > 0

    def test_dropped_frames_do_not_change_surviving_data(self) -> None:
        _, lossy = collect(config(drop_probability=0.3, seed=11), 0.1)
        _, clean = collect(config(seed=11), 0.1)
        clean_by_seq = by_seq(clean)
        compared = 0
        for f in lossy:
            if f.seq in clean_by_seq:
                np.testing.assert_array_equal(f.data, clean_by_seq[f.seq])
                compared += 1
        assert compared >= 5

    def test_no_drops_when_off(self) -> None:
        sensor, frames = collect(config(seed=11), 0.1)
        assert sensor.dropped_frames == 0
        seqs = [f.seq for f in frames]
        assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))

    def test_clock_offset_is_constant(self) -> None:
        _, frames = collect(config(clock_offset_s=-12.5), 0.02)
        assert all(f.device_ts == f.host_ts - 12.5 for f in frames)

    def test_saturation_clamps(self) -> None:
        cfg = config(saturation_max=0.02, rest_std=0.1, noise_std=0.0)
        sensor = SimulatedTactile(cfg)
        frame = sensor._frame(0)
        assert float(np.abs(frame).max()) <= 0.02
        assert np.any(np.abs(frame) == np.float32(0.02))
        assert float(np.abs(sensor.rest).max()) > 0.02  # the underlying signal does exceed the limit
        clean = SimulatedTactile(config(rest_std=0.1, noise_std=0.0))
        assert float(np.abs(clean._frame(0)).max()) > 0.02


class TestBaseline:
    def test_capture_baseline_returns_the_rest_pattern_without_noise(self) -> None:
        with SimulatedTactile(config(noise_std=0.0)) as sensor:
            baseline = sensor.capture_baseline(0.05)
            np.testing.assert_array_equal(np.asarray(baseline.values, np.float32), sensor.rest)
        assert baseline.applied is False and baseline.duration_s == 0.05

    def test_capture_baseline_approaches_the_rest_pattern_with_noise(self) -> None:
        with SimulatedTactile(config(noise_std=0.01)) as sensor:
            baseline = sensor.capture_baseline(0.2)
            error = np.abs(np.asarray(baseline.values, np.float32) - sensor.rest)
        assert float(error.max()) < 0.01  # ~100 samples: standard error 0.001, generous bound


class TestContactSchedule:
    def test_press_adds_its_amplitude_on_the_pressed_elements_while_it_is_on(self) -> None:
        clean = SimulatedTactile(config(seed=5))
        pressed = SimulatedTactile(
            config(seed=5, contact_schedule=[(0.1, 0.2, 2.0), (0.2, 0.05, 1.0)], contact_elements=[1, 3])
        )
        # seq / 500 Hz: the first press covers [0.1, 0.3) s, the second [0.2, 0.25) s, and they add up.
        expected = {0: 0.0, 49: 0.0, 50: 2.0, 100: 3.0, 124: 3.0, 125: 2.0, 149: 2.0, 150: 0.0}
        for seq, amplitude in expected.items():
            assert pressed.press_amplitude(seq) == amplitude
            delta = pressed._frame(seq) - clean._frame(seq)
            wanted = np.zeros(clean.config.shape, np.float32)
            wanted[[1, 3], :] = amplitude
            np.testing.assert_allclose(delta, wanted, atol=1e-6)

    def test_default_pressed_elements_are_the_first_quarter(self) -> None:
        assert config(n_elements=16).pressed_elements == [0, 1, 2, 3]
        assert config(n_elements=2).pressed_elements == [0]
        assert config(n_elements=16, contact_elements=[7]).pressed_elements == [7]

    def test_held_press_spans_a_detection_window(self) -> None:
        sensor = SimulatedTactile(config(contact_schedule=[(0.2, 1.0, 1.5)]))
        pressed = [seq for seq in range(int(2 * RATE)) if sensor.press_amplitude(seq) == 1.5]
        assert pressed == list(range(100, 600))  # 1 s of 500 Hz frames, longer than a 0.5 s window
        _, frames = collect(config(contact_schedule=[(0.0, 10.0, 1.5)], noise_std=0.0), 0.05)
        for frame in frames:
            np.testing.assert_allclose(frame.data[:4] - sensor.rest[:4], 1.5, atol=1e-6)
            np.testing.assert_allclose(frame.data[4:], sensor.rest[4:], atol=1e-6)

    def test_press_is_applied_before_the_clamp_and_under_dead_elements(self) -> None:
        cfg = config(
            contact_schedule=[(0.0, 1.0, 5.0)], saturation_max=1.0, dead_elements=[0], dead_value=-9.0
        )
        frame = SimulatedTactile(cfg)._frame(0)
        assert np.all(frame[1:4] == np.float32(1.0)) and np.all(frame[0] == np.float32(-9.0))


class TestRebaseline:
    def test_rebaseline_subtracts_rest_plus_drift_from_the_scheduled_frame_on(self) -> None:
        cfg = config(drift_per_s=[0.0, 0.0, 2.0], rebaseline_at_s=[0.05], noise_std=0.0)
        sensor, frames = collect(cfg, 0.15)
        before = [f for f in frames if f.seq < 25]
        after = [f for f in frames if f.seq >= 25]
        assert before and after
        for f in before:
            assert f.baseline is None
            np.testing.assert_array_equal(f.data, sensor._frame(f.seq))
        applied = after[0].baseline
        assert applied is not None and applied.frame_range == (25, 25)
        np.testing.assert_array_equal(applied.values, sensor.drifted_rest(25))
        np.testing.assert_allclose(applied.values[:, 2] - sensor.rest[:, 2], 2.0 * 25 / RATE, atol=1e-6)
        for f in after:
            assert f.baseline is applied  # one object per re-baseline
            np.testing.assert_array_equal(f.data, sensor._frame(f.seq) - applied.values)
            np.testing.assert_allclose(f.data[:, 2], 2.0 * (f.seq - 25) / RATE, atol=1e-5)  # back to zero
            np.testing.assert_allclose(f.raw, sensor._frame(f.seq), atol=1e-6)  # the sensor has not

    def test_rebaseline_fires_on_the_next_produced_frame_after_a_drop(self) -> None:
        seed = next(
            s
            for s in range(500)
            if SimulatedTactile(config(seed=s, drop_probability=0.3))._is_dropped(25)
            and not SimulatedTactile(config(seed=s, drop_probability=0.3))._is_dropped(26)
        )
        cfg = config(seed=seed, drop_probability=0.3, rebaseline_at_s=[0.05])
        _, frames = collect(cfg, 0.12)
        stamped = [f for f in frames if f.baseline is not None]
        assert stamped and stamped[0].seq == 26 and stamped[0].baseline.frame_range == (26, 26)
        assert all(f.baseline is None for f in frames if f.seq < 26)

    def test_two_scheduled_times_give_two_captures_and_reconnect_restarts_them(self) -> None:
        cfg = config(rebaseline_at_s=[0.02, 0.06], noise_std=0.0)
        sensor = SimulatedTactile(cfg)
        sensor.connect()
        time.sleep(0.1)
        sensor.disconnect()
        frames = sensor.drain()
        objects = []
        for f in frames:
            if f.baseline is not None and (not objects or f.baseline is not objects[-1]):
                objects.append(f.baseline)
        assert [o.frame_range for o in objects] == [(10, 10), (30, 30)]
        sensor.connect()
        sensor.disconnect()
        again = sensor.drain()
        assert again and again[0].seq == 0 and again[0].baseline is None
