"""The per-instance sidecar: meta.json, native-rate episode files and nearest-timestamp alignment."""

import hashlib
import json

import numpy as np
import pytest

from taxel.dataset.sidecar import (
    COLUMNS,
    align_nearest,
    episode_path,
    frame_times,
    meta_path,
    read_episode,
    read_session_meta,
    sidecar_dir,
    write_episode,
    write_session_meta,
)
from taxel.schema import Clock, FrameCounts, SchemaError, SessionMeta, SessionRecord
from taxel.sensors import TactileReading

SHAPE = (4, 3)


def readings(n: int, *, start: float = 100.0, period: float = 0.002, device: bool = True, seq: bool = True):
    rng = np.random.default_rng(0)
    return [
        TactileReading(
            rng.standard_normal(SHAPE).astype(np.float32),
            host_ts=start + i * period,
            device_ts=start + i * period + 0.5 if device else None,
            seq=10 + i if seq else None,
        )
        for i in range(n)
    ]


def session(episodes=(0, 1), origins=(10.0, 70.0)) -> SessionRecord:
    return SessionRecord(
        session_id="20261121T140305Z",
        episodes=episodes,
        clock=Clock(wall_time="2026-11-21T14:03:05.117Z", host_clock=1.5, episode_origins=list(origins)),
        frames=FrameCounts(received=1200, dropped=3, overflowed=0),
        adapter=None,
        calibration=None,
        baseline=None,
    )


class TestPaths:
    def test_layout(self, tmp_path) -> None:
        assert sidecar_dir(tmp_path, "left_fingertip") == tmp_path / "tactile" / "left_fingertip"
        assert sidecar_dir(tmp_path, "right", "markers") == tmp_path / "tactile" / "right" / "markers"
        assert meta_path(tmp_path, "left_fingertip") == tmp_path / "tactile" / "left_fingertip" / "meta.json"
        assert episode_path(tmp_path, "left_fingertip", 7).name == "episode-000007.parquet"
        assert episode_path(tmp_path, "right", 0, "force").parent.name == "force"

    def test_validation(self, tmp_path) -> None:
        with pytest.raises(SchemaError):
            sidecar_dir(tmp_path, "left/fingertip")
        with pytest.raises(SchemaError):
            episode_path(tmp_path, "right", 0, "depth")
        with pytest.raises(ValueError):
            episode_path(tmp_path, "right", -1)


class TestSessionMeta:
    def test_round_trip_and_layout(self, tmp_path) -> None:
        meta = SessionMeta(instance_id="left_fingertip", sessions=[session()])
        path = write_session_meta(tmp_path, meta)
        assert path == tmp_path / "tactile" / "left_fingertip" / "meta.json"
        text = path.read_text()
        assert text.startswith('{\n  "instance_id": "left_fingertip",\n') and text.endswith("}\n")
        assert list(json.loads(text)["sessions"][0])[:4] == ["session_id", "episodes", "clock", "frames"]
        assert read_session_meta(tmp_path, "left_fingertip") == meta

    def test_wrong_instance_and_missing(self, tmp_path) -> None:
        path = write_session_meta(tmp_path, SessionMeta(instance_id="thumb", sessions=[session()]))
        (tmp_path / "tactile" / "palm_0").mkdir()
        (tmp_path / "tactile" / "palm_0" / "meta.json").write_text(path.read_text())
        with pytest.raises(SchemaError, match="is for instance 'thumb'"):
            read_session_meta(tmp_path, "palm_0")
        with pytest.raises(FileNotFoundError):
            read_session_meta(tmp_path, "left_fingertip")


class TestEpisodeFiles:
    @pytest.fixture(autouse=True)
    def _needs_pyarrow(self):
        pytest.importorskip("pyarrow")

    def test_round_trip_and_column_types(self, tmp_path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        frames = readings(50)
        path = write_episode(episode_path(tmp_path, "left_fingertip", 3), 3, frames, SHAPE)
        table = pq.read_table(path)
        assert table.column_names == list(COLUMNS)
        types = {name: table.schema.field(name).type for name in COLUMNS}
        assert types["episode_index"] == pa.int64() and types["sample_index"] == pa.int64()
        assert types["host_ts"] == pa.float64() and types["device_ts"] == pa.float64()
        assert types["seq"] == pa.int64()
        assert isinstance(types["data"], pa.FixedShapeTensorType)
        assert tuple(types["data"].shape) == SHAPE and types["data"].value_type == pa.float32()

        stream = read_episode(path)
        assert len(stream) == 50 and stream.episode_index == 3 and stream.shape == SHAPE
        assert table.column("sample_index").to_pylist() == list(range(50))
        assert set(table.column("episode_index").to_pylist()) == {3}
        np.testing.assert_array_equal(stream.host_ts, [f.host_ts for f in frames])
        np.testing.assert_array_equal(stream.device_ts, [f.device_ts for f in frames])
        np.testing.assert_array_equal(stream.seq, [f.seq for f in frames])
        np.testing.assert_array_equal(stream.data, np.stack([f.data for f in frames]))
        assert stream.data.dtype == np.float32 and stream.host_ts.dtype == np.float64

    def test_host_ts_is_monotonic_in_what_is_written(self, tmp_path) -> None:
        frames = readings(20)
        stream = read_episode(write_episode(tmp_path / "e.parquet", 0, frames, SHAPE))
        assert np.all(np.diff(stream.host_ts) > 0)
        backwards = frames[:5] + [TactileReading(frames[5].data, host_ts=frames[2].host_ts)]
        with pytest.raises(ValueError, match="host_ts must not decrease"):
            write_episode(tmp_path / "bad.parquet", 0, backwards, SHAPE)

    def test_sensor_without_clock_or_counter(self, tmp_path) -> None:
        stream = read_episode(
            write_episode(tmp_path / "e.parquet", 0, readings(5, device=False, seq=False), SHAPE)
        )
        assert stream.device_ts is None and stream.seq is None

    def test_empty_episode_keeps_shape_and_index(self, tmp_path) -> None:
        stream = read_episode(write_episode(tmp_path / "e.parquet", 4, [], SHAPE))
        assert len(stream) == 0 and stream.episode_index == 4 and stream.shape == SHAPE
        assert stream.device_ts is None and stream.seq is None

    def test_wrong_dtype_or_shape_is_refused(self, tmp_path) -> None:
        bad_dtype = [TactileReading(np.zeros(SHAPE, dtype=np.float64), host_ts=1.0)]
        with pytest.raises(ValueError, match="expected float32"):
            write_episode(tmp_path / "e.parquet", 0, bad_dtype, SHAPE)
        bad_shape = [TactileReading(np.zeros((4, 2), dtype=np.float32), host_ts=1.0)]
        with pytest.raises(ValueError, match="expected float32"):
            write_episode(tmp_path / "e.parquet", 0, bad_shape, SHAPE)

    def test_partial_seq_is_an_error(self, tmp_path) -> None:
        frames = readings(3)
        frames[1] = TactileReading(frames[1].data, host_ts=frames[1].host_ts, device_ts=frames[1].device_ts)
        path = write_episode(tmp_path / "e.parquet", 0, frames, SHAPE)
        with pytest.raises(SchemaError, match="'seq' is null on 1 of 3"):
            read_episode(path)


class TestAlignment:
    def test_nearest_with_ties_and_edges(self) -> None:
        host_ts = [0.0, 0.1, 0.2, 0.3]
        targets = [0.04, 0.05, 0.06, 0.1, 0.26, -1.0, 5.0]
        np.testing.assert_array_equal(align_nearest(host_ts, targets), [0, 0, 1, 1, 3, 0, 3])
        assert align_nearest(host_ts, 0.29).tolist() == [3]

    def test_synthetic_stream_at_two_rates(self) -> None:
        native = 100.0 + np.arange(500) / 500.0  # 500 Hz for one second
        targets = frame_times(100.0, 30, 30.0)  # 30 fps over the same second
        idx = align_nearest(native, targets)
        np.testing.assert_array_equal(idx, np.rint((targets - 100.0) * 500.0).astype(int))
        assert np.all(np.abs(native[idx] - targets) <= 0.5 / 500.0 + 1e-12)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            align_nearest([], [0.0])
        with pytest.raises(ValueError, match="sorted"):
            align_nearest([0.2, 0.1], [0.0])


class TestBesideLeRobot:
    def test_dataset_with_sidecar_loads_untouched(self, tmp_path) -> None:
        pytest.importorskip("lerobot")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from taxel.dataset.features import feature_entry
        from taxel.schema import TactileInfo, feature_key

        info = TactileInfo(
            family="taxel_array",
            declared_rate_hz=500,
            channels=["fx", "fy", "fz"],
            session_meta="tactile/left_fingertip/meta.json",
        )
        key = feature_key("left_fingertip")
        root = tmp_path / "ds"
        dataset = LeRobotDataset.create(
            "taxel/sidecar_test", fps=30, features={key: feature_entry(info, SHAPE)}, root=root
        )
        for _ in range(6):
            dataset.add_frame({key: np.zeros(SHAPE, dtype=np.float32), "task": "press"})
        dataset.save_episode()
        write_episode(episode_path(root, "left_fingertip", 0), 0, readings(100), SHAPE)
        write_session_meta(
            root, SessionMeta(instance_id="left_fingertip", sessions=[session((0, 0), (100.0,))])
        )
        dataset.finalize()

        digest = {
            p: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "tactile").rglob("*")
            if p.is_file()
        }
        reloaded = LeRobotDataset("taxel/sidecar_test", root=root)
        assert reloaded.num_episodes == 1 and reloaded.num_frames == 6
        assert tuple(reloaded[0][key].shape) == SHAPE
        assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in digest} == digest
        assert read_session_meta(root, "left_fingertip").sessions[0].clock.episode_origins == [100.0]
