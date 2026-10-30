"""Feature entries built from ``TactileInfo``, suffixed streams, and their survival through LeRobot."""

import copy

import numpy as np
import pytest

from taxel.dataset.features import (
    feature_entries,
    feature_entry,
    read_feature_entry,
    read_stream_entry,
    stream_entry,
    tactile_instances,
)
from taxel.schema import Reduction, SchemaError, StreamInfo, TactileInfo, feature_key

LEFT = TactileInfo.from_dict(
    {
        "family": "taxel_array",
        "vendor": "PaXini",
        "model": "PX-6AX-GEN3",
        "declared_rate_hz": 91,
        "units": {"force": "N"},
        "channels": ["fx", "fy", "fz"],
        "reduction": {"method": "mean", "window_ms": 33.3},
        "mount": {"body": "so101_follower", "link": "gripper_left_finger", "pose": None},
        "device": None,
        "session_meta": "tactile/left_fingertip/meta.json",
    }
)
RIGHT = TactileInfo.from_dict(
    {
        "family": "visuotactile",
        "vendor": "GelSight",
        "model": "DIGIT",
        "declared_rate_hz": 60,
        "reduction": {"method": "nearest", "window_ms": 33.3},
        "markers": {"count": 63, "grid": [7, 9]},
        "derived": {"method": "marker-flow-depth", "version": "0.1.0"},
        "mount": None,
        "device": None,
        "session_meta": "tactile/right_fingertip/meta.json",
    }
)
WRENCH = TactileInfo(
    family="force_torque",
    declared_rate_hz=1000,
    channels=["fx", "fy", "fz", "mx", "my", "mz"],
    session_meta="tactile/wrist/meta.json",
)


class TestFeatureEntry:
    def test_array_entry_carries_info_verbatim(self) -> None:
        entry = feature_entry(LEFT, (52, 3))
        assert entry == {"dtype": "float32", "shape": (52, 3), "names": None, "info": LEFT.to_dict()}
        assert list(entry["info"]) == list(LEFT.to_dict())

    def test_shape_must_match_channels(self) -> None:
        with pytest.raises(SchemaError, match=r"\(n_elements, 3\)"):
            feature_entry(LEFT, (52, 6))
        with pytest.raises(SchemaError, match=r"\(n_elements, 3\)"):
            feature_entry(LEFT, (52,))
        with pytest.raises(SchemaError, match=r"\(n_elements, 6\)"):
            feature_entry(WRENCH, (1, 3))
        assert feature_entry(WRENCH, (1, 6))["shape"] == (1, 6)

    def test_shape_must_be_positive_integers(self) -> None:
        for bad in ((0, 3), (52, -3), (52.0, 3), "52x3", ()):
            with pytest.raises(SchemaError, match="positive integers"):
                feature_entry(LEFT, bad)  # type: ignore[arg-type]

    def test_visuotactile_entry_is_a_camera_entry(self) -> None:
        entry = feature_entry(RIGHT, (240, 320, 3))
        assert entry["dtype"] == "video"
        assert entry["names"] == ["height", "width", "channels"]
        assert entry["info"] == RIGHT.to_dict()
        assert feature_entry(RIGHT, (240, 320, 3), use_videos=False)["dtype"] == "image"
        with pytest.raises(SchemaError, match=r"\(height, width, 3\)"):
            feature_entry(RIGHT, (240, 320))


class TestStreamEntry:
    def test_stream_entry_carries_only_reduction_and_session_meta(self) -> None:
        reduction = Reduction(method="mean", window_ms=33.3)
        entry = stream_entry(RIGHT, "markers", (63, 2), reduction)
        assert entry == {
            "dtype": "float32",
            "shape": (63, 2),
            "names": None,
            "info": {"reduction": {"method": "mean", "window_ms": 33.3}, "session_meta": RIGHT.session_meta},
        }
        assert stream_entry(RIGHT, "force", (63, 3))["info"] == {"session_meta": RIGHT.session_meta}

    def test_stream_shapes(self) -> None:
        with pytest.raises(SchemaError, match=r"\.markers frames are \(N, 2\)"):
            stream_entry(RIGHT, "markers", (63, 3))
        with pytest.raises(SchemaError, match=r"\.displacement frames are \(N, 3\)"):
            stream_entry(RIGHT, "displacement", (63, 2))
        with pytest.raises(SchemaError, match="markers.count is 63"):
            stream_entry(RIGHT, "markers", (64, 2))

    def test_streams_are_checked_against_the_instance(self) -> None:
        with pytest.raises(SchemaError, match="visuotactile instances only"):
            stream_entry(LEFT, "force", (52, 3))
        no_derived = copy.deepcopy(RIGHT)
        no_derived.derived = None
        with pytest.raises(SchemaError, match="require a 'derived' block"):
            stream_entry(no_derived, "force", (63, 3))
        stream_entry(no_derived, "markers", (63, 2))
        with pytest.raises(SchemaError, match="unknown stream suffixes"):
            stream_entry(RIGHT, "depth", (63, 1))

    def test_feature_entries_builds_the_whole_instance(self) -> None:
        entries = feature_entries(
            "right_fingertip",
            RIGHT,
            (240, 320, 3),
            {"markers": ((63, 2), Reduction("nearest", 33.3)), "force": ((63, 3), None)},
        )
        assert list(entries) == [
            "observation.tactile.right_fingertip",
            "observation.tactile.right_fingertip.markers",
            "observation.tactile.right_fingertip.force",
        ]
        assert read_feature_entry(entries["observation.tactile.right_fingertip"]) == RIGHT
        assert read_stream_entry(entries["observation.tactile.right_fingertip.force"]) == StreamInfo(
            session_meta=RIGHT.session_meta
        )
        with pytest.raises(SchemaError, match="visuotactile instances only"):
            feature_entries("left_fingertip", LEFT, (52, 3), {"markers": ((63, 2), None)})


class TestReadBack:
    def test_round_trip(self) -> None:
        assert read_feature_entry(feature_entry(LEFT, (52, 3))) == LEFT
        assert read_feature_entry(feature_entry(RIGHT, (240, 320, 3))) == RIGHT

    def test_per_session_fact_in_info_is_rejected(self) -> None:
        entry = feature_entry(LEFT, (52, 3))
        entry["info"]["serial"] = "PX3-00412"
        with pytest.raises(SchemaError, match=r"info: unknown keys \['serial'\]"):
            read_feature_entry(entry)

    def test_entry_must_agree_with_info(self) -> None:
        entry = feature_entry(LEFT, (52, 3))
        with pytest.raises(SchemaError, match="no 'info' block"):
            read_feature_entry({k: v for k, v in entry.items() if k != "info"})
        with pytest.raises(SchemaError, match="features are float32"):
            read_feature_entry({**entry, "dtype": "float64"})
        with pytest.raises(SchemaError, match=r"\(n_elements, 3\)"):
            read_feature_entry({**entry, "shape": [52, 4]})
        with pytest.raises(SchemaError, match="video or image"):
            read_feature_entry({**feature_entry(RIGHT, (240, 320, 3)), "dtype": "float32"})
        with pytest.raises(SchemaError, match="streams are float32"):
            read_stream_entry({**stream_entry(RIGHT, "markers", (63, 2)), "dtype": "video"})

    def test_tactile_instances(self) -> None:
        features = {
            "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.tactile.right_fingertip.markers": {},
            "observation.tactile.right_fingertip": {},
            "observation.tactile.left_fingertip": {},
            "observation.images.wrist": {},
        }
        assert tactile_instances(features) == {"left_fingertip": [], "right_fingertip": ["markers"]}
        with pytest.raises(SchemaError, match="unknown stream suffix 'depth'"):
            tactile_instances({"observation.tactile.right_fingertip.depth": {}})
        with pytest.raises(SchemaError, match=r"without an instance key: \['palm_0'\]"):
            tactile_instances({"observation.tactile.palm_0.force": {}})


# --- Through LeRobot --------------------------------------------------------------------------------


@pytest.fixture
def lerobot_dataset():
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def write_dataset(cls, root, info, n_frames: int = 5, seed: int = 0):
    key = feature_key("left_fingertip")
    rng = np.random.default_rng(seed)
    features = {key: feature_entry(info, (52, 3))}
    dataset = cls.create("taxel/features_test", fps=30, features=features, root=root)
    for _ in range(n_frames):
        dataset.add_frame({key: rng.standard_normal((52, 3)).astype(np.float32), "task": "press"})
    dataset.save_episode()
    dataset.finalize()
    return dataset


class TestThroughLeRobot:
    def test_entry_survives_create_and_reload(self, lerobot_dataset, tmp_path) -> None:
        write_dataset(lerobot_dataset, tmp_path / "ds", LEFT)
        reloaded = lerobot_dataset("taxel/features_test", root=tmp_path / "ds")
        entry = reloaded.meta.features[feature_key("left_fingertip")]
        assert entry["shape"] == (52, 3)
        assert read_feature_entry(entry) == LEFT
        assert tuple(reloaded[0][feature_key("left_fingertip")].shape) == (52, 3)

    def test_suffixed_entries_survive_in_info_json(self, lerobot_dataset, tmp_path) -> None:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        features = feature_entries(
            "right_fingertip",
            RIGHT,
            (240, 320, 3),
            {"markers": ((63, 2), Reduction("nearest", 33.3)), "force": ((63, 3), None)},
        )
        LeRobotDatasetMetadata.create("taxel/streams_test", fps=30, features=features, root=tmp_path / "ds")
        meta = LeRobotDatasetMetadata("taxel/streams_test", root=tmp_path / "ds")
        assert tactile_instances(meta.features) == {"right_fingertip": ["force", "markers"]}
        assert read_feature_entry(meta.features[feature_key("right_fingertip")]) == RIGHT
        assert read_stream_entry(meta.features[feature_key("right_fingertip", "markers")]) == StreamInfo(
            session_meta=RIGHT.session_meta, reduction=Reduction("nearest", 33.3)
        )

    def test_identical_info_passes_the_merge_check_and_a_differing_one_fails_it(
        self, lerobot_dataset, tmp_path
    ) -> None:
        """``merge`` validates feature entries with ``features_equal_for_merge`` before copying anything,
        so two sessions of the same instance merge only when their static metadata is identical."""
        from lerobot.datasets.feature_utils import features_equal_for_merge

        a = write_dataset(lerobot_dataset, tmp_path / "a", LEFT, seed=1)
        b = write_dataset(lerobot_dataset, tmp_path / "b", LEFT, seed=2)
        assert features_equal_for_merge(a.meta.features, b.meta.features)

        other = copy.deepcopy(LEFT)
        other.declared_rate_hz = 100
        c = write_dataset(lerobot_dataset, tmp_path / "c", other, seed=3)
        assert not features_equal_for_merge(a.meta.features, c.meta.features)

    @pytest.mark.xfail(
        strict=True,
        reason="LeRobot 0.6.x merge rewrites the data parquet through pandas, and pyarrow cannot convert a "
        "2-D array column from a DataFrame (lerobot.datasets.aggregate). Independent of the info block.",
    )
    def test_merge_copies_two_dimensional_features(self, lerobot_dataset, tmp_path) -> None:
        from lerobot.datasets.dataset_tools import merge_datasets

        a = write_dataset(lerobot_dataset, tmp_path / "a", LEFT, seed=1)
        b = write_dataset(lerobot_dataset, tmp_path / "b", LEFT, seed=2)
        merged = merge_datasets([a, b], "taxel/merged", output_dir=tmp_path / "merged")
        assert merged.num_episodes == 2
        assert read_feature_entry(merged.meta.features[feature_key("left_fingertip")]) == LEFT
