"""Tests for the simulated image sensor: determinism, the rendered press, the dead block, the delay."""

import time
from typing import Any

import numpy as np
import pytest

from taxel.schema import SchemaError, TactileFamily, TactileInfo
from taxel.sensors import (
    SimulatedVisuotactile,
    SimulatedVisuotactileConfig,
    TactileReading,
    TactileSensorConfig,
    make_tactile_sensor,
)

RATE = 60.0


def config(**overrides: Any) -> SimulatedVisuotactileConfig:
    base: dict[str, Any] = {"instance_id": "right_fingertip", "declared_rate_hz": RATE, "timeout_ms": 500.0}
    return SimulatedVisuotactileConfig(**{**base, **overrides})


def collect(
    cfg: SimulatedVisuotactileConfig, seconds: float
) -> tuple[SimulatedVisuotactile, list[TactileReading]]:
    sensor = SimulatedVisuotactile(cfg)
    sensor.connect()
    time.sleep(seconds)
    sensor.disconnect()
    frames = sensor.drain()
    assert frames, "no frames received"
    return sensor, frames


class TestRegistryShapeAndMetadata:
    def test_registered_and_discoverable(self) -> None:
        assert config().type == "simulated_visuotactile"
        assert TactileSensorConfig.get_choice_class("simulated_visuotactile") is SimulatedVisuotactileConfig
        assert isinstance(make_tactile_sensor(config()), SimulatedVisuotactile)
        assert SimulatedVisuotactile.find_sensors()[0]["type"] == "simulated_visuotactile"

    def test_frames_are_uint8_images_at_the_configured_size(self) -> None:
        with SimulatedVisuotactile(config(height=32, width=40)) as sensor:
            frame = sensor.read()
        assert frame.data.shape == (32, 40, 3) and frame.data.dtype == np.uint8
        assert config(height=32, width=40).shape == (32, 40, 3)

    def test_metadata_is_valid_for_a_visuotactile_instance(self) -> None:
        info = SimulatedVisuotactile(config(marker_grid=(3, 5))).metadata()
        assert TactileInfo.from_dict(info.to_dict()) == info
        assert info.family is TactileFamily.VISUOTACTILE and info.channels is None
        assert info.layout is not None and info.layout.surface == "gel"
        assert info.layout.positions_mm is None and "positions_mm" not in info.to_dict()["layout"]
        assert info.markers is not None and info.markers.count == 15 and info.markers.grid == [3, 5]
        assert info.session_meta == "tactile/right_fingertip/meta.json"
        assert info.declared_rate_hz == RATE

    def test_rate_and_contiguous_seq(self) -> None:
        _, frames = collect(config(), 0.3)
        n = len(frames)
        assert 0.6 * RATE * 0.3 < n < 1.3 * RATE * 0.3, n
        seqs = [f.seq for f in frames]
        assert seqs == list(range(seqs[0], seqs[0] + n))

    def test_baselines_are_not_subtracted_from_images(self) -> None:
        with SimulatedVisuotactile(config()) as sensor:
            with pytest.raises(NotImplementedError, match="reference frame"):
                sensor.apply_baseline(np.zeros((48, 64, 3), np.float32))
            with pytest.raises(NotImplementedError):
                sensor.capture_baseline(0.05)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"family": "taxel_array"}, "visuotactile"),
            ({"height": 4}, "at least 8"),
            ({"marker_grid": (0, 6)}, "marker_grid"),
            ({"marker_grid": (4, 100)}, "does not fit"),
            ({"noise_std": -1.0}, "non-negative"),
            ({"contact_schedule": [(0.0, 0.0, 10.0)]}, "duration_s > 0"),
            ({"contact_center": (1.5, 0.5)}, "contact_center"),
            ({"contact_radius": 0.0}, "contact_radius"),
            ({"dead_pixels": (40, 0, 10, 4)}, "does not fit"),
            ({"dead_pixels": (0, 0, 0, 4)}, "does not fit"),
            ({"dead_value": 300}, "8-bit"),
            ({"transport_delay_s": -0.1}, "transport_delay_s"),
        ],
    )
    def test_config_validation(self, overrides: dict, match: str) -> None:
        with pytest.raises((ValueError, SchemaError), match=match):
            config(**overrides)


class TestDeterminism:
    def test_same_seed_same_frames_and_pure_in_seq(self) -> None:
        a, b = SimulatedVisuotactile(config(seed=7)), SimulatedVisuotactile(config(seed=7))
        for seq in (0, 1, 33):
            np.testing.assert_array_equal(a._frame(seq), b._frame(seq))
            np.testing.assert_array_equal(a._frame(seq), a._frame(seq))
        assert not np.array_equal(a._frame(0), a._frame(1))  # noise differs between frames
        assert not np.array_equal(a._frame(0), SimulatedVisuotactile(config(seed=8))._frame(0))

    def test_received_frames_equal_the_pure_function(self) -> None:
        sensor, frames = collect(config(seed=3), 0.05)
        for f in frames:
            np.testing.assert_array_equal(f.data, sensor._frame(f.seq))


class TestPressDeadPixelsAndClock:
    def test_background_has_markers_and_no_noise(self) -> None:
        sensor = SimulatedVisuotactile(config(marker_grid=(2, 2), noise_std=0.0))
        background = sensor.background
        assert background.shape == (48, 64, 3) and background.dtype == np.uint8
        np.testing.assert_array_equal(sensor._frame(0), background)
        assert int(background[12, 16, 0]) < int(background[0, 0, 0]) - 50  # a marker is darker than the gel

    def test_press_brightens_the_patch_while_it_is_on(self) -> None:
        cfg = config(contact_schedule=[(0.5, 0.5, 60.0)], noise_std=0.0, marker_grid=(1, 1))
        sensor = SimulatedVisuotactile(cfg)
        cy, cx = 23, 31  # the patch centre of a 48x64 image, next to the Gaussian's peak
        before, on, after = sensor._frame(29), sensor._frame(30), sensor._frame(60)  # 60 Hz
        assert sensor.press_amplitude(29) == 0.0 and sensor.press_amplitude(30) == 60.0
        np.testing.assert_array_equal(before, sensor.background)
        np.testing.assert_array_equal(after, sensor.background)
        lift = on.astype(int) - sensor.background.astype(int)
        assert lift[cy, cx].min() >= 57 and lift[cy, cx].max() <= 60
        assert np.all(lift[0, 0] == 0) and np.all(lift[-1, -1] == 0)  # far from the centre: unchanged
        assert np.all(lift >= 0)

    def test_dead_block_is_constant_while_the_rest_varies(self) -> None:
        cfg = config(dead_pixels=(10, 20, 4, 6), dead_value=7, contact_schedule=[(0.0, 9.0, 40.0)])
        _, frames = collect(cfg, 0.1)
        stacked = np.stack([f.data for f in frames])
        assert np.all(stacked[:, 10:14, 20:26, :] == 7)
        assert stacked[:, 9, 20, 0].std() > 0 and stacked[:, 10, 19, 0].std() > 0  # the neighbours have noise
        assert np.any(stacked[:, 24, 32, :] != 7)

    def test_transport_delay_moves_the_content_and_shifts_device_ts(self) -> None:
        press = [(0.0, 0.1, 50.0)]
        plain = SimulatedVisuotactile(config(contact_schedule=press, noise_std=0.0))
        late = SimulatedVisuotactile(config(contact_schedule=press, noise_std=0.0, transport_delay_s=0.1))
        # 0.1 s is 6 frames at 60 Hz: the late sensor shows the press from frame 6 to frame 11.
        assert plain.press_amplitude(0) == 50.0 and late.press_amplitude(0) == 0.0
        assert late.press_amplitude(6) == 50.0 and late.press_amplitude(11) == 50.0
        assert late.press_amplitude(12) == 0.0
        for seq in range(6, 12):
            np.testing.assert_array_equal(late._frame(seq), plain._frame(seq - 6))
        _, frames = collect(config(transport_delay_s=0.1), 0.05)
        assert all(abs(f.device_ts - (f.host_ts - 0.1)) < 1e-9 for f in frames)
