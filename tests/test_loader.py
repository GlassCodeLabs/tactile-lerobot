"""The doctor's dataset loader: a taxel recording, the same recording with its metadata damaged, and plain
LeRobot datasets with tactile-looking features and no sidecar."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from taxel.doctor.loader import DoctorDataset, candidate_keys, resolve_root, sample_indices, tactile_keys
from taxel.schema import Range, SchemaError

KEY = "observation.tactile.left_fingertip"
SENSOR = {
    "type": "simulated",
    "declared_rate_hz": 500,
    "n_elements": 16,
    "drop_probability": 0.03,
    "saturation_max": 2.0,
    "seed": 1,
}
needs_lerobot = pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="needs taxel[lerobot]")


class TestDiscovery:
    FEATURES = {
        "observation.tactile.left_fingertip": {"dtype": "float32", "shape": [16, 3]},
        "observation.tactile.right_fingertip": {"dtype": "video", "shape": [240, 320, 3]},
        "observation.tactile.right_fingertip.markers": {"dtype": "float32", "shape": [63, 2]},
        "observation.images.left_tactile_left": {"dtype": "video", "shape": [400, 700, 3]},
        "observation.state": {"dtype": "float32", "shape": [6]},
    }

    def test_tactile_looking_keys(self) -> None:
        assert tactile_keys(self.FEATURES) == {
            "left_fingertip": "observation.tactile.left_fingertip",
            "right_fingertip": "observation.tactile.right_fingertip",
        }
        assert tactile_keys({"observation.tactile": {"dtype": "float32", "shape": [20]}}) == {
            "tactile": "observation.tactile"
        }

    def test_explicit_names_by_instance_or_feature_key(self) -> None:
        chosen = tactile_keys(self.FEATURES, ["left_fingertip", "observation.images.left_tactile_left"])
        assert chosen == {
            "left_fingertip": "observation.tactile.left_fingertip",
            "left_tactile_left": "observation.images.left_tactile_left",
        }
        assert tactile_keys(self.FEATURES, ["observation.state"]) == {
            "observation.state": "observation.state"
        }
        with pytest.raises(KeyError, match="neither a tactile instance"):
            tactile_keys(self.FEATURES, ["thumb"])

    def test_candidate_keys_by_name(self) -> None:
        features = {
            **self.FEATURES,
            "observation.wrench": {"dtype": "float32", "shape": [6]},
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.Force_valid": {"dtype": "int64", "shape": [6]},
            "touch_timestamp": {"dtype": "float64", "shape": [1, 1, 1]},
        }
        assert candidate_keys(features) == [
            "observation.tactile.left_fingertip",
            "observation.tactile.right_fingertip",
            "observation.tactile.right_fingertip.markers",
            "observation.images.left_tactile_left",
            "observation.wrench",
        ]
        assert candidate_keys({"observation.state": {"dtype": "float32", "shape": [6]}}) == []

    def test_sample_indices(self) -> None:
        assert sample_indices(10, 4).tolist() == [0, 3, 6, 9]
        assert sample_indices(3, 60).tolist() == [0, 1, 2] and sample_indices(1, 1).tolist() == [0]
        assert len(sample_indices(1891, 60)) == 60

    def test_resolve_root(self, tmp_path) -> None:
        assert resolve_root(tmp_path) == tmp_path
        with pytest.raises(FileNotFoundError, match="does not look like a Hub repo id"):
            resolve_root(tmp_path / "missing")


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> Path:
    """Two one-second episodes from the simulated sensor through ``taxel record``, baseline applied."""
    pytest.importorskip("lerobot")
    from taxel.cli.main import main

    base = tmp_path_factory.mktemp("recorded")
    args = [
        "record",
        "--robot.type=cameras_only",
        f"--robot.calibration_dir={base / 'calibration'}",
        "--tactile=" + json.dumps({"left_fingertip": SENSOR}),
        "--dataset.repo_id=taxel/loader_test",
        f"--dataset.root={base / 'ds'}",
        "--dataset.no_stamp=true",
        "--dataset.num_episodes=2",
        "--dataset.episode_time_s=1",
        "--dataset.reset_time_s=0.3",
        "--dataset.fps=30",
        "--dataset.single_task=press",
        "--baseline_s=0.2",
        "--apply_baseline=true",
        "--session_id=first",
        "--play_sounds=false",
    ]
    assert main(args) == 0
    return base / "ds"


def plain_dataset(root: Path, *, use_videos: bool, bare: bool = False):
    """A dataset LeRobot itself wrote: a (4, 3) tactile array, a camera and a visuotactile image feature.

    With ``bare`` the one feature is ``observation.tactile`` of shape (20,), as some public datasets name it.
    Returns the frames fed in, keyed by feature, so a test can compare decoded images exactly.
    """
    from lerobot.datasets import LeRobotDataset

    if bare:
        features = {"observation.tactile": {"dtype": "float32", "shape": [20], "names": None}}
    else:
        image = {
            "dtype": "video" if use_videos else "image",
            "shape": [32, 40, 3],
            "names": ["height", "width", "channels"],
        }
        features = {
            "observation.tactile.probe": {"dtype": "float32", "shape": [4, 3], "names": None},
            "observation.images.cam": image,
            "observation.tactile.gel": image,
        }
    dataset = LeRobotDataset.create(
        "taxel/plain", fps=30, features=features, root=root, use_videos=use_videos
    )
    rng = np.random.default_rng(0)
    fed: dict[str, list] = {k: [] for k in features}
    for _ in range(2):
        for _ in range(12):
            frame = {}
            for key, ft in features.items():
                if ft["dtype"] == "float32":
                    frame[key] = rng.normal(size=ft["shape"]).astype(np.float32)
                else:
                    frame[key] = rng.integers(0, 255, size=ft["shape"], dtype=np.uint8)
                fed[key].append(frame[key])
            dataset.add_frame({**frame, "task": "t"})
        dataset.save_episode()
    dataset.finalize()
    return {k: np.stack(v) for k, v in fed.items()}


@pytest.fixture(scope="module")
def plain_video(tmp_path_factory):
    pytest.importorskip("lerobot")
    root = tmp_path_factory.mktemp("plain") / "ds"
    return root, plain_dataset(root, use_videos=True)


@pytest.fixture(scope="module")
def plain_image(tmp_path_factory):
    pytest.importorskip("lerobot")
    root = tmp_path_factory.mktemp("plain_img") / "ds"
    return root, plain_dataset(root, use_videos=False)


@needs_lerobot
class TestRecordedDataset:
    def test_instance_metadata_and_sidecar(self, recorded) -> None:
        ds = DoctorDataset(recorded)
        assert ds.fps == 30 and ds.episode_indices == [0, 1] and ds.camera_keys == []
        instances = ds.instances()
        assert list(instances) == ["left_fingertip"]
        inst = instances["left_fingertip"]
        assert inst.key == KEY and inst.kind == "array" and inst.shape == (16, 3) and inst.n_channels == 3
        assert inst.metadata == "info" and inst.error is None and inst.sidecar
        assert inst.declared_rate_hz == 500 and inst.range == Range(min=-2.0, max=2.0)
        assert inst.reduction is not None and inst.reduction.method == "mean"
        assert inst.unknown_keys == [] and inst.missing_keys == []
        assert inst.session_for(1).session_id == "first" and inst.session_for(2) is None
        report = inst.report().to_dict()
        assert report["family"] == "taxel_array" and report["range"] == {"min": -2.0, "max": 2.0}

    def test_episode_inputs_match_lerobot_and_the_sidecar(self, recorded) -> None:
        from lerobot.datasets import LeRobotDataset

        ds = DoctorDataset(recorded)
        inst = ds.instances()["left_fingertip"]
        reloaded = LeRobotDataset("taxel/loader_test", root=recorded)
        for e in (0, 1):
            ep = ds.episode_inputs(inst, e)
            row = reloaded.meta.episodes[e]
            n, start = int(row["length"]), int(row["dataset_from_index"])
            assert ep.n_frames == n and ep.frames.shape == (n, 16, 3) and ep.frames.dtype == np.float32
            stored = np.stack([reloaded[start + k][KEY].numpy() for k in range(n)])
            np.testing.assert_array_equal(ep.frames, stored)
            np.testing.assert_allclose(ep.timestamps, np.arange(n) / 30, atol=1e-6)
            assert ep.native is not None and len(ep.native) > n and ep.native.episode_index == e
            assert ep.baseline_rows is not None and ep.baseline_rows.captures[0]
            assert ep.session is not None and ep.session.session_id == "first"
            assert ep.origin == ep.session.clock.episode_origins[e]
            assert ep.missing == []

    def test_lenient_findings_are_carried_not_raised(self, recorded, tmp_path) -> None:
        root = tmp_path / "damaged"
        shutil.copytree(recorded, root)
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["features"][KEY]["info"]["serial"] = "PX3-00412"  # a per-session fact that strayed into info
        info_path.write_text(json.dumps(info))
        meta_path = root / "tactile" / "left_fingertip" / "meta.json"
        meta = json.loads(meta_path.read_text())
        del meta["sessions"][0]["baseline"]["frame_range"]  # a 0.2 record
        meta_path.write_text(json.dumps(meta))
        inst = DoctorDataset(root).instances()["left_fingertip"]
        assert inst.metadata == "info" and inst.sidecar and inst.error is None
        assert inst.unknown_keys == ["info.serial"]
        assert inst.missing_keys == ["meta.sessions[0].baseline.frame_range"]
        assert inst.session_for(0).baseline.frame_range is None

    def test_missing_files_are_named(self, recorded, tmp_path) -> None:
        root = tmp_path / "partial"
        shutil.copytree(recorded, root)
        (root / "tactile" / "left_fingertip" / "episode-000001.parquet").unlink()
        ds = DoctorDataset(root)
        inst = ds.instances()["left_fingertip"]
        ep = ds.episode_inputs(inst, 1)
        assert ep.native is None and ep.baseline_rows is not None and ep.frames.shape[1:] == (16, 3)
        assert ep.missing == ["no native stream: tactile/left_fingertip/episode-000001.parquet not found"]
        (root / "tactile" / "left_fingertip" / "meta.json").unlink()
        ds = DoctorDataset(root)
        inst = ds.instances()["left_fingertip"]
        assert not inst.sidecar and inst.metadata == "info"
        ep = ds.episode_inputs(inst, 0)
        assert ep.native is None and ep.session is None and ep.origin is None
        assert ep.missing == ["no sidecar: tactile/left_fingertip/meta.json not found"]

    def test_an_unreadable_info_block_is_reported(self, recorded, tmp_path) -> None:
        root = tmp_path / "invalid"
        shutil.copytree(recorded, root)
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["features"][KEY]["info"] = {"family": "taxel_array"}
        info_path.write_text(json.dumps(info))
        inst = DoctorDataset(root).instances()["left_fingertip"]
        assert inst.metadata == "invalid" and inst.info is None and inst.declared_rate_hz is None
        assert (
            "missing required key" in inst.error and inst.sidecar
        )  # the sidecar is still found by convention
        assert inst.report().to_dict()["family"] == "array"


@needs_lerobot
class TestPlainDataset:
    def test_array_instance_without_sidecar(self, plain_video) -> None:
        root, fed = plain_video
        ds = DoctorDataset(root)
        instances = ds.instances()
        assert sorted(instances) == ["gel", "probe"] and ds.camera_keys == ["observation.images.cam"]
        probe = instances["probe"]
        assert probe.kind == "array" and probe.metadata == "none" and not probe.sidecar
        assert probe.info is None and probe.declared_rate_hz is None and probe.range is None
        ep = ds.episode_inputs(probe, 1)
        assert ep.frames.shape == (12, 4, 3)
        np.testing.assert_array_equal(ep.frames, fed["observation.tactile.probe"][12:])
        assert ep.native is None and ep.baseline_rows is None and ep.session is None
        assert ep.missing == ["no sidecar: tactile/probe/meta.json not found"]
        assert ep.instance.report().to_dict()["metadata"] == "none"

    def test_lerobot_video_keys_are_not_a_tactile_block(self, plain_video) -> None:
        """LeRobot writes ``has_audio`` and ``is_depth_map`` beside a video feature's ``video.*`` keys;
        a block with none of the tactile block's required keys is core's, not a broken instance."""
        root, _ = plain_video
        features = json.loads((root / "meta" / "info.json").read_text())["features"]
        info = features["observation.tactile.gel"]["info"]
        assert "has_audio" in info and not any(k in info for k in ("family", "declared_rate_hz"))
        gel = DoctorDataset(root).instances()["gel"]
        assert gel.metadata == "none" and gel.error is None and gel.info is None
        assert gel.unknown_keys == [] and gel.report().to_dict()["metadata"] == "none"

    def test_one_sequential_pass_yields_the_same_frames(self, plain_video, tmp_path) -> None:
        root, _ = plain_video
        ds = DoctorDataset(root)
        key = "observation.tactile.gel"
        triples = list(ds.iter_frames(key, 1, rgb_at=[0, 5, 11]))
        assert [k for k, _, _ in triples] == list(range(12))
        gray = np.stack([g for _, g, _ in triples])
        assert (
            gray.shape == (12, 32, 40) and gray.dtype == np.float32 and 0 <= gray.min() <= gray.max() <= 255
        )
        given = [k for k, _, rgb in triples if rgb is not None]
        assert given == [0, 5, 11]
        rgb = np.stack([rgb for _, _, rgb in triples if rgb is not None])
        np.testing.assert_array_equal(rgb, ds.frames(key, 1, indices=[0, 5, 11]))  # the same decoder output
        assert sum(1 for _, _, rgb in ds.iter_frames(key, 0) if rgb is not None) == 0
        with pytest.raises(SchemaError, match="not a video or image feature"):
            next(ds.iter_frames("observation.tactile.probe", 0))
        # The video's frames are matched to the table's rows by order and checked against the timestamps:
        # a column that claims twice the video's rate is caught at the second frame.
        import pyarrow as pa
        import pyarrow.parquet as pq

        copy = tmp_path / "doubled"
        shutil.copytree(root, copy)
        path = copy / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(path)
        doubled = pa.array((table.column("timestamp").to_numpy() * 2).astype(np.float32))
        pq.write_table(table.set_column(table.column_names.index("timestamp"), "timestamp", doubled), path)
        with pytest.raises(SchemaError, match="video frame at 0.0333 s where the table's frame 1"):
            list(DoctorDataset(copy).iter_frames(key, 0))

    def test_clock_column_names_and_files_on_disk(self, plain_video, tmp_path) -> None:
        root, _ = plain_video
        ds = DoctorDataset(root)
        assert ds.instances()["probe"].names is None  # no per-element names in this feature
        ds.check_clock("timestamp")  # any one-number-per-frame column will do
        np.testing.assert_array_equal(ds.clock_times("timestamp", 1), ds.timestamps(1))
        with pytest.raises(KeyError, match="clock column 'nope'"):
            ds.check_clock("nope")
        with pytest.raises(SchemaError, match="float32 \\[4, 3\\], not one number per frame"):
            ds.check_clock("observation.tactile.probe")
        inputs = ds.episode_inputs(ds.instances()["probe"], 1, clock="timestamp")
        assert inputs.clock == "timestamp" and np.array_equal(inputs.clock_times, inputs.timestamps)
        assert ds.has_frames("observation.images.cam", 0) and ds.has_frames("observation.tactile.gel", 1)
        partial_root = tmp_path / "partial"
        shutil.copytree(root, partial_root)
        for video in (partial_root / "videos" / "observation.images.cam").rglob("*.mp4"):
            video.unlink()
        partial = DoctorDataset(partial_root)
        assert not partial.has_frames("observation.images.cam", 0)
        with pytest.raises(FileNotFoundError, match="observation.images.cam: episode 0's video file videos/"):
            next(partial.iter_frames("observation.images.cam", 0))

    def test_png_frames_through_the_same_pass(self, plain_image) -> None:
        root, fed = plain_image
        ds = DoctorDataset(root)
        key = "observation.tactile.gel"
        triples = list(ds.iter_frames(key, 1, rgb_at=[3]))
        assert len(triples) == 12 and triples[3][2] is not None and triples[4][2] is None
        np.testing.assert_array_equal(triples[3][2], fed[key][12 + 3])  # PNG is lossless
        np.testing.assert_allclose(triples[3][1], fed[key][12 + 3].astype(np.float32).mean(-1))

    def test_image_instance_and_video_frames(self, plain_video) -> None:
        from lerobot.datasets import LeRobotDataset

        root, _ = plain_video
        ds = DoctorDataset(root)
        gel = ds.instances()["gel"]
        assert gel.kind == "image" and gel.shape == (32, 40, 3)
        assert ds.episode_inputs(gel, 0).frames is None
        frames = ds.frames("observation.tactile.gel", 1, max_frames=4)
        assert frames.shape == (4, 32, 40, 3) and frames.dtype == np.uint8
        assert ds.frames("observation.images.cam", 0).shape == (12, 32, 40, 3)
        reloaded = LeRobotDataset("taxel/plain", root=root, video_backend="pyav")
        for i, k in enumerate((0, 4, 7, 11)):  # the evenly spread indices of 4 over 12 frames
            theirs = reloaded[12 + k]["observation.tactile.gel"].permute(1, 2, 0).numpy()
            np.testing.assert_array_equal(frames[i], np.rint(theirs * 255).astype(np.uint8))

    def test_video_frames_are_queried_at_the_stored_timestamps(
        self, plain_video, tmp_path, monkeypatch
    ) -> None:
        """A dataset whose ``timestamp`` column holds the video's real presentation times, a 29.97 Hz
        video under a declared 30 fps, is read at those times, as LeRobot itself reads it, not at
        ``frame_index / fps``: by frame three the two differ by more than the decoder's tolerance."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch
        from lerobot.datasets import video_utils

        root, _ = plain_video
        copy = tmp_path / "pts"
        shutil.copytree(root, copy)
        path = copy / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(path)
        frame_index = np.asarray(table.column("frame_index").to_numpy(), dtype=np.float64)
        real = (frame_index * 1001 / 30000).astype(np.float32)
        table = table.set_column(table.column_names.index("timestamp"), "timestamp", pa.array(real))
        pq.write_table(table, path)

        ds = DoctorDataset(copy)
        np.testing.assert_allclose(ds.frame_stamps(1), real[12:], rtol=0, atol=1e-7)
        assert ds.frame_stamps(1) is ds.frame_stamps(1)  # read once
        asked: list[list[float]] = []

        def fake_decode(video_path, timestamps, tolerance_s, backend, return_uint8):
            asked.append(list(timestamps))
            return torch.zeros((len(timestamps), 3, 32, 40), dtype=torch.uint8)

        monkeypatch.setattr(video_utils, "decode_video_frames", fake_decode)
        frames = ds.frames("observation.images.cam", 1, indices=[11, 0, 5])
        assert frames.shape == (3, 32, 40, 3)
        from_ts = ds.episode(1).videos["observation.images.cam"][1]
        np.testing.assert_allclose(asked, [from_ts + real[12:][[0, 5, 11]]], rtol=0, atol=1e-6)
        assert abs(asked[0][2] - (from_ts + 11 / 30)) > 1e-4  # the nominal time would miss the frame

    def test_image_dtype_frames_decode_exactly(self, plain_image) -> None:
        root, fed = plain_image
        ds = DoctorDataset(root)
        frames = ds.frames("observation.tactile.gel", 0, max_frames=3)
        np.testing.assert_array_equal(frames, fed["observation.tactile.gel"][[0, 6, 11]])
        assert ds.frames("observation.images.cam", 1).shape == (12, 32, 40, 3)

    def test_bare_one_dimensional_feature(self, tmp_path) -> None:
        root = tmp_path / "bare"
        fed = plain_dataset(root, use_videos=False, bare=True)
        ds = DoctorDataset(root)
        inst = ds.instances()["tactile"]
        assert inst.key == "observation.tactile" and inst.shape == (20,) and inst.n_channels == 1
        ep = ds.episode_inputs(inst, 0)
        assert ep.frames.shape == (12, 20, 1)
        np.testing.assert_array_equal(ep.frames[:, :, 0], fed["observation.tactile"][:12])

    def test_not_a_dataset(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="not a LeRobot dataset"):
            DoctorDataset(tmp_path)
