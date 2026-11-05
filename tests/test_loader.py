"""The doctor's dataset loader: a taxel recording, the same recording with its metadata damaged, and plain
LeRobot datasets with tactile-looking features and no sidecar."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from taxel.doctor.loader import DoctorDataset, resolve_root, tactile_keys
from taxel.schema import Range

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
