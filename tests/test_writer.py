"""Simulated sensors through TactileDatasetWriter into a LeRobotDataset, end to end."""

import re
import time

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from taxel import __version__  # noqa: E402
from taxel.dataset.features import read_feature_entry  # noqa: E402
from taxel.dataset.reduce import reduce_episode  # noqa: E402
from taxel.dataset.sidecar import (  # noqa: E402
    baseline_path,
    episode_path,
    frame_times,
    raw_frames,
    read_baseline_episode,
    read_episode,
    read_session_meta,
)
from taxel.dataset.writer import TactileDatasetWriter, baseline_rows, tactile_features  # noqa: E402
from taxel.schema import Reduction, SchemaError, Skin  # noqa: E402
from taxel.sensors import SimulatedTactile, SimulatedTactileConfig, TactileReading, host_clock  # noqa: E402
from taxel.sensors.base import AppliedBaseline  # noqa: E402

FPS = 30
KEY = "observation.tactile.left_fingertip"


def sensor_config(**overrides) -> SimulatedTactileConfig:
    config = {
        "instance_id": "left_fingertip",
        "declared_rate_hz": 500.0,
        "n_elements": 16,
        "drop_probability": 0.05,
        "clock_offset_s": 0.25,
        "seed": 1,
    }
    return SimulatedTactileConfig(**{**config, **overrides})


def record(dataset, writer, n_episodes: int, n_frames: int, task: str = "press") -> None:
    """The loop shape of lerobot-record: one dataset frame per tick, then save, then a reset pause."""
    for _ in range(n_episodes):
        writer.start_episode()
        t0 = None
        for k in range(n_frames):
            if t0 is not None:
                time.sleep(max(t0 + k / FPS - host_clock(), 0.0))
            values = writer.frame()
            if t0 is None:
                t0 = host_clock()
            dataset.add_frame({**values, "task": task})
        writer.end_episode()
        dataset.save_episode()
        writer.idle()
        time.sleep(0.05)
        writer.idle()


@pytest.fixture
def connected():
    """A connected simulated sensor, disconnected after the test."""
    sensors = []

    def make(**overrides) -> SimulatedTactile:
        sensor = SimulatedTactile(sensor_config(**overrides))
        sensor.connect()
        sensors.append(sensor)
        return sensor

    yield make
    for sensor in sensors:
        sensor.disconnect()


def make_dataset(root, sensors, method="mean", repo_id="taxel/writer_test"):
    features = tactile_features(sensors, FPS, method)
    return LeRobotDataset.create(repo_id, fps=FPS, features=features, root=root, use_videos=False)


class TestFeatures:
    def test_reduction_declared_only_when_rates_differ(self, connected) -> None:
        fast, same = connected(), connected(instance_id="palm_0", declared_rate_hz=FPS)
        features = tactile_features([fast, same], FPS)
        assert read_feature_entry(features[KEY]).reduction == Reduction("mean", 33.333)
        assert read_feature_entry(features["observation.tactile.palm_0"]).reduction is None
        assert features[KEY]["shape"] == (16, 3)
        assert tactile_features([fast], FPS, "last", window_ms=50.0)[KEY]["info"]["reduction"] == {
            "method": "last",
            "window_ms": 50.0,
        }

    def test_nearest_is_refused(self, connected) -> None:
        with pytest.raises(ValueError, match="cannot be reproduced from the sidecar"):
            tactile_features([connected()], FPS, "nearest")

    def test_disconnected_sensor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not connected"):
            tactile_features([SimulatedTactile(sensor_config())], FPS)


class TestWriterSetup:
    def test_dataset_must_carry_the_sensor_metadata(self, connected, tmp_path) -> None:
        sensor = connected()
        dataset = make_dataset(tmp_path / "ds", [sensor])
        other = connected(declared_rate_hz=400.0)  # same instance id, different declared rate
        with pytest.raises(SchemaError, match="does not match the metadata"):
            TactileDatasetWriter(dataset, [other])
        with pytest.raises(SchemaError, match="has no feature"):
            TactileDatasetWriter(dataset, [connected(instance_id="thumb")])
        with pytest.raises(SchemaError, match="session fields \\['notes'\\]"):
            TactileDatasetWriter(dataset, [sensor], session={"notes": "x"})
        with pytest.raises(ValueError, match="apply_baseline needs a baseline"):
            TactileDatasetWriter(dataset, [sensor], baseline_s=0, apply_baseline=True)
        with pytest.raises(SchemaError, match="sample_every"):
            TactileDatasetWriter(dataset, [sensor], baseline_sample_every=0)

    def test_episode_protocol(self, connected, tmp_path) -> None:
        sensor = connected()
        writer = TactileDatasetWriter(make_dataset(tmp_path / "ds", [sensor]), [sensor], baseline_s=0)
        with pytest.raises(RuntimeError, match="start_session"):
            writer.start_episode()
        writer.start_session("s1")
        with pytest.raises(RuntimeError, match="start_episode"):
            writer.frame()
        assert writer.start_episode() == 0
        with pytest.raises(RuntimeError, match="no dataset frame"):
            writer.end_episode()
        writer.discard_episode()
        assert writer.episode_index is None and writer.records == {"left_fingertip": None}
        writer.finalize()  # nothing recorded: no meta.json
        assert not (tmp_path / "ds" / "tactile").exists()


class TestEndToEnd:
    def test_three_episodes_reload_with_feature_sidecar_and_record_intact(self, connected, tmp_path) -> None:
        sensor = connected()
        root = tmp_path / "ds"
        dataset = make_dataset(root, [sensor])
        writer = TactileDatasetWriter(dataset, [sensor], baseline_s=0.2, session={"skin": Skin(id="skin-7")})
        writer.start_session()
        record(dataset, writer, n_episodes=3, n_frames=60)
        sensor.disconnect()  # freezes the counters; finalize() copies them into the session record
        writer.finalize()
        dataset.finalize()
        counters = (sensor.frames_received, sensor.dropped_frames, sensor.overflowed_frames)
        assert counters[1] > 0  # the drop fault fired

        # The feature entry: shape, static metadata and the reduction that was applied.
        reloaded = LeRobotDataset("taxel/writer_test", root=root)
        assert reloaded.num_episodes == 3 and reloaded.num_frames == 180
        entry = reloaded.meta.features[KEY]
        assert entry["shape"] == (16, 3)
        info = read_feature_entry(entry)
        assert info.reduction == Reduction("mean", 33.333)
        assert info.to_dict() == {**sensor.metadata().to_dict(), "reduction": info.reduction.to_dict()}

        # The session record.
        meta = read_session_meta(root, "left_fingertip")
        assert len(meta.sessions) == 1
        session = meta.sessions[0]
        assert session.episodes == (0, 2)
        assert re.fullmatch(r"\d{8}T\d{6}Z", session.session_id)
        assert session.adapter.name == "taxel.simulated" and session.adapter.version == __version__
        assert session.baseline is not None and not session.baseline.applied
        assert np.asarray(session.baseline.values).shape == (16, 3)
        lo_seq, hi_seq = session.baseline.frame_range
        assert 0 <= lo_seq < hi_seq and session.baseline_stream.sample_every == 100
        assert session.skin == Skin(id="skin-7") and session.calibration is None and session.serial is None
        assert session.frames.received == counters[0]
        assert session.frames.dropped == counters[1]
        assert session.frames.overflowed == counters[2] == 0
        origins = session.clock.episode_origins
        assert len(origins) == 3 and origins == sorted(origins) and origins[0] > session.clock.host_clock
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", session.clock.wall_time)

        # The sidecar: every native frame the sensor produced, timing intact, and the copy reproducible.
        files = sorted(p.name for p in (root / "tactile" / "left_fingertip").iterdir())
        assert files == [
            "baseline",
            "episode-000000.parquet",
            "episode-000001.parquet",
            "episode-000002.parquet",
            "meta.json",
        ]
        for e in range(3):
            stream = read_episode(episode_path(root, "left_fingertip", e))
            assert stream.episode_index == e and stream.shape == (16, 3)
            assert np.all(np.diff(stream.host_ts) > 0)
            np.testing.assert_allclose(stream.device_ts - stream.host_ts, 0.25, atol=1e-6)
            lo, hi = int(stream.seq[0]), int(stream.seq[-1])
            assert stream.seq.tolist() == [s for s in range(lo, hi + 1) if not sensor._is_dropped(s)]
            assert hi - lo + 1 > 60 * 500 / FPS * 0.8  # ~2 s of 500 Hz frames per episode
            targets = frame_times(origins[e], 60, FPS)
            expected = reduce_episode("mean", stream.host_ts, stream.data, targets, info.reduction.window_ms)
            stored = np.stack([reloaded[e * 60 + k][KEY].numpy() for k in range(60)])
            np.testing.assert_array_equal(stored, expected)
            assert np.all(np.diff(targets) > 0) and stream.host_ts[0] <= targets[0]
            # Nothing applied: the baseline stream has no capture rows and its samples are the stored frames.
            rows = read_baseline_episode(baseline_path(root, "left_fingertip", e))
            assert rows.episode_index == e and not rows.captures.any()
            assert rows.sample_index.tolist() == list(range(0, len(stream), 100))
            np.testing.assert_array_equal(rows.values, stream.data[rows.sample_index])
            np.testing.assert_array_equal(rows.seq, stream.seq[rows.sample_index])
            np.testing.assert_array_equal(raw_frames(stream, rows), stream.data)

    def test_applied_baseline_and_a_rebaseline_mid_episode_stay_reproducible(
        self, connected, tmp_path
    ) -> None:
        sensor = connected()
        root = tmp_path / "ds"
        dataset = make_dataset(root, [sensor])
        writer = TactileDatasetWriter(
            dataset, [sensor], baseline_s=0.2, apply_baseline=True, baseline_sample_every=50
        )
        writer.start_session("applied")
        first = sensor.applied_baseline
        assert first is not None and first.frame_range is not None
        writer.start_episode()
        t0 = None
        second = None
        for k in range(45):
            if t0 is not None:
                time.sleep(max(t0 + k / FPS - host_clock(), 0.0))
            if k == 20:  # a backend re-baselining mid-episode
                second = sensor.apply_baseline(
                    first.values + np.float32(0.5), (first.frame_range[1] + 1, first.frame_range[1] + 10)
                )
            values = writer.frame()
            if t0 is None:
                t0 = host_clock()
            dataset.add_frame({**values, "task": "press"})
        writer.end_episode()
        dataset.save_episode()
        writer.finalize()
        dataset.finalize()

        session = read_session_meta(root, "left_fingertip").sessions[0]
        assert session.baseline.applied and session.baseline.frame_range == first.frame_range
        np.testing.assert_array_equal(np.asarray(session.baseline.values, np.float32), first.values)
        assert session.baseline_stream.sample_every == 50

        stream = read_episode(episode_path(root, "left_fingertip", 0))
        rows = read_baseline_episode(baseline_path(root, "left_fingertip", 0))
        captures = np.flatnonzero(rows.captures)
        assert len(captures) == 2 and rows.sample_index[captures[0]] == 0
        np.testing.assert_array_equal(rows.values[captures[0]], first.values)
        assert (
            rows.frame_range[captures[0]] == first.frame_range and rows.host_ts[captures[0]] == first.host_ts
        )
        change = int(rows.sample_index[captures[1]])
        assert 0 < change < len(stream) and rows.seq[captures[1]] == stream.seq[change]
        np.testing.assert_array_equal(rows.values[captures[1]], second.values)
        assert (
            rows.frame_range[captures[1]] == second.frame_range
            and rows.host_ts[captures[1]] == second.host_ts
        )
        assert stream.host_ts[change - 1] < second.host_ts <= stream.host_ts[change] + 0.05

        # Stored frames are raw minus the baseline in force; raw_frames undoes it to float32 rounding.
        truth = np.stack([sensor._frame(int(s)) for s in stream.seq])
        np.testing.assert_array_equal(stream.data[:change], truth[:change] - first.values)
        np.testing.assert_array_equal(stream.data[change:], truth[change:] - second.values)
        raw = raw_frames(stream, rows)
        np.testing.assert_allclose(raw, truth, rtol=0, atol=1e-6)
        samples = np.flatnonzero(rows.samples)
        assert rows.sample_index[samples].tolist() == list(range(0, len(stream), 50))
        np.testing.assert_array_equal(rows.values[samples], raw[rows.sample_index[samples]])

        # The main-table copy is still the reduction of the stored (subtracted) native stream.
        reloaded = LeRobotDataset("taxel/writer_test", root=root)
        origin = session.clock.episode_origins[0]
        expected = reduce_episode("mean", stream.host_ts, stream.data, frame_times(origin, 45, FPS), 33.333)
        stored = np.stack([reloaded[k][KEY].numpy() for k in range(45)])
        np.testing.assert_array_equal(stored, expected)

    def test_baseline_rows_from_readings(self) -> None:
        shape = (2, 3)
        a = AppliedBaseline(np.full(shape, 1.0, np.float32), (0, 9), 10.0)
        b = AppliedBaseline(np.full(shape, 2.0, np.float32), None, 11.0)
        readings = [
            TactileReading(np.full(shape, float(i), np.float32), 20.0 + i, seq=100 + i, baseline=base)
            for i, base in enumerate([a, a, a, b, b, None, None, None])
        ]
        rows = baseline_rows(readings, 3, shape)
        assert [(r.kind, r.sample_index, r.seq) for r in rows] == [
            ("capture", 0, 100),
            ("sample", 0, 100),
            ("capture", 3, 103),
            ("sample", 3, 103),
            ("capture", 5, 105),
            ("sample", 6, 106),
        ]
        assert rows[0].host_ts == 10.0 and rows[0].frame_range == (0, 9)
        assert rows[2].host_ts == 11.0 and rows[2].frame_range is None
        np.testing.assert_array_equal(rows[4].values, np.zeros(shape, np.float32))  # cleared: zeros
        assert rows[4].host_ts == 25.0
        np.testing.assert_array_equal(rows[1].values, np.full(shape, 1.0, np.float32))  # raw = 0 + a
        np.testing.assert_array_equal(rows[3].values, np.full(shape, 5.0, np.float32))  # raw = 3 + b
        np.testing.assert_array_equal(rows[5].values, np.full(shape, 6.0, np.float32))  # raw, nothing applied
        assert baseline_rows([], 3, shape) == []
        plain = [TactileReading(np.zeros(shape, np.float32), 1.0) for _ in range(4)]
        assert [(r.kind, r.sample_index) for r in baseline_rows(plain, 2, shape)] == [
            ("sample", 0),
            ("sample", 2),
        ]

    def test_slow_sensor_holds_the_last_sample_and_stays_reproducible(self, connected, tmp_path) -> None:
        sensor = connected(declared_rate_hz=5.0, drop_probability=0.0)
        root = tmp_path / "ds"
        dataset = make_dataset(root, [sensor], method="last")
        writer = TactileDatasetWriter(dataset, [sensor], baseline_s=0)
        writer.start_session("slow")
        record(dataset, writer, n_episodes=1, n_frames=30)
        writer.finalize()
        dataset.finalize()
        reloaded = LeRobotDataset("taxel/writer_test", root=root)
        stream = read_episode(episode_path(root, "left_fingertip", 0))
        origin = read_session_meta(root, "left_fingertip").sessions[0].clock.episode_origins[0]
        stored = np.stack([reloaded[k][KEY].numpy() for k in range(30)])
        expected = reduce_episode("last", stream.host_ts, stream.data, frame_times(origin, 30, FPS), 33.333)
        np.testing.assert_array_equal(stored, expected)
        distinct = len({row.tobytes() for row in stored})
        assert 3 <= distinct <= 8  # about one new 5 Hz sample per 6 frames; the rest are holds

    def test_resume_appends_a_session(self, connected, tmp_path) -> None:
        root = tmp_path / "ds"
        first = connected()
        dataset = make_dataset(root, [first])
        writer = TactileDatasetWriter(dataset, [first], baseline_s=0)
        writer.start_session("one")
        record(dataset, writer, n_episodes=2, n_frames=10)
        writer.finalize()
        dataset.finalize()
        first.disconnect()

        second = connected()
        dataset = LeRobotDataset.resume("taxel/writer_test", root=root)
        writer = TactileDatasetWriter(dataset, [second], baseline_s=0)
        with pytest.raises(SchemaError, match="already exists"):
            writer.start_session("one")
        writer = TactileDatasetWriter(dataset, [second], baseline_s=0)
        writer.start_session("two")
        record(dataset, writer, n_episodes=1, n_frames=10)
        writer.finalize()
        dataset.finalize()

        sessions = read_session_meta(root, "left_fingertip").sessions
        assert [(s.session_id, s.episodes) for s in sessions] == [("one", (0, 1)), ("two", (2, 2))]
        assert [len(s.clock.episode_origins) for s in sessions] == [2, 1]
        assert LeRobotDataset("taxel/writer_test", root=root).num_episodes == 3
        assert episode_path(root, "left_fingertip", 2).exists()

    def test_push_to_hub_would_upload_the_sidecar(self, connected, tmp_path, monkeypatch) -> None:
        import lerobot.datasets.lerobot_dataset as module
        from huggingface_hub.utils import filter_repo_objects

        sensor = connected()
        root = tmp_path / "ds"
        dataset = make_dataset(root, [sensor])
        writer = TactileDatasetWriter(dataset, [sensor], baseline_s=0)
        writer.start_session("hub")
        record(dataset, writer, n_episodes=1, n_frames=10)
        writer.finalize()
        dataset.finalize()

        uploads: list[dict] = []

        class FakeHfApi:
            def create_repo(self, **kwargs):
                pass

            def upload_folder(self, **kwargs):
                uploads.append(kwargs)

            def delete_tag(self, *args, **kwargs):
                pass

            def create_tag(self, *args, **kwargs):
                pass

        class FakeCard:
            def push_to_hub(self, **kwargs):
                pass

        monkeypatch.setattr(module, "HfApi", FakeHfApi)
        monkeypatch.setattr(module, "create_lerobot_dataset_card", lambda **kwargs: FakeCard())
        dataset.push_to_hub()

        (kwargs,) = uploads
        assert kwargs["folder_path"] == root
        files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
        uploaded = set(
            filter_repo_objects(
                files, allow_patterns=kwargs["allow_patterns"], ignore_patterns=kwargs["ignore_patterns"]
            )
        )
        assert {
            "tactile/left_fingertip/meta.json",
            "tactile/left_fingertip/episode-000000.parquet",
            "tactile/left_fingertip/baseline/episode-000000.parquet",
        } <= uploaded
        assert "meta/info.json" in uploaded
