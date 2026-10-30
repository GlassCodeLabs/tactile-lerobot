"""Round-trip and validation tests for ``taxel.schema`` against the examples in ``SCHEMA.md``."""

import copy
import json

import pytest

from taxel.schema import (
    Adapter,
    Baseline,
    Calibration,
    SchemaError,
    SessionMeta,
    SessionRecord,
    StreamInfo,
    TactileFamily,
    TactileInfo,
    feature_key,
    read_info_lenient,
    session_meta_path,
)

N_TAXELS = 52


def positions(n: int) -> list[list[float]]:
    return [[1.5 * i, 0.0, 0.0] for i in range(n)]


# The three feature entries from SCHEMA.md, "Static metadata", with the "..." rows filled in.
LEFT_FINGERTIP = {
    "family": "taxel_array",
    "vendor": "PaXini",
    "model": "PX-6AX-GEN3 DP-S2015-Elite",
    "declared_rate_hz": 91,
    "units": {"force": "N"},
    "channels": ["fx", "fy", "fz"],
    "layout": {"frame": "sensor", "positions_mm": positions(N_TAXELS), "normals": [[0, 0, 1]] * N_TAXELS},
    "reduction": {"method": "mean", "window_ms": 33.3},
    "mount": {"body": "so101_follower", "link": "gripper_left_finger", "pose": None},
    "device": {"id": "paxini_hub_0", "clock": "paxini_hub_0"},
    "session_meta": "tactile/left_fingertip/meta.json",
}

RIGHT_FINGERTIP = {
    "family": "visuotactile",
    "vendor": "GelSight",
    "model": "DIGIT",
    "declared_rate_hz": 60,
    "reduction": {"method": "nearest", "window_ms": 33.3},
    "markers": {"count": 63, "grid": [7, 9]},
    "derived": {"method": "marker-flow-depth", "version": "0.1.0"},
    "mount": {"body": "so101_follower", "link": "gripper_right_finger", "pose": None},
    "device": None,
    "session_meta": "tactile/right_fingertip/meta.json",
}

PALM = {
    "family": "magnetic",
    "vendor": "NYU / AnySkin",
    "model": "AnySkin v1",
    "declared_rate_hz": 200,
    "units": {"field": "uT"},
    "channels": ["bx", "by", "bz"],
    "layout": {
        "frame": "sensor",
        "positions_mm": [[0, 0, 0], [6, 0, 0], [-6, 0, 0], [0, 6, 0], [0, -6, 0]],
        "normals": None,
    },
    "reduction": {"method": "mean", "window_ms": 33.3},
    "mount": {"body": "so101_follower", "link": "palm", "pose": None},
    "device": None,
    "session_meta": "tactile/palm_0/meta.json",
}

# The sidecar from SCHEMA.md, "Per-session metadata", with the "..." rows filled in.
LEFT_FINGERTIP_META = {
    "instance_id": "left_fingertip",
    "sessions": [
        {
            "session_id": "2026-11-21a",
            "episodes": [0, 41],
            "serial": "PX3-00412",
            "firmware": None,
            "device": {"serial": "PXH-0193", "firmware": "1.4.2"},
            "adapter": {"name": "tactile-lerobot.paxini", "version": "0.1.0"},
            "calibration": {
                "version": "vendor-factory-2026-03",
                "method": "vendor",
                "date": "2026-03-14",
                "reference": None,
            },
            "baseline": {
                "captured_at": "2026-11-21T14:03:11Z",
                "duration_s": 2.0,
                "applied": True,
                "values": [[0.01, -0.02, 0.00]] * N_TAXELS,
            },
            "gel": None,
            "skin": None,
        }
    ],
}


def json_round_trip(obj: object) -> object:
    return json.loads(json.dumps(obj))


class TestInfoRoundTrip:
    @pytest.mark.parametrize(
        "example", [LEFT_FINGERTIP, RIGHT_FINGERTIP, PALM], ids=["taxel", "visuo", "magnetic"]
    )
    def test_example_round_trips_exactly(self, example: dict) -> None:
        info = TactileInfo.from_dict(example)
        assert info.to_dict() == example
        assert json_round_trip(info.to_dict()) == example

    def test_key_order_matches_schema(self) -> None:
        assert list(TactileInfo.from_dict(LEFT_FINGERTIP).to_dict()) == list(LEFT_FINGERTIP)

    def test_optional_blocks_are_omitted_not_nulled(self) -> None:
        out = TactileInfo.from_dict(RIGHT_FINGERTIP).to_dict()
        assert "channels" not in out and "layout" not in out and "units" not in out
        assert out["device"] is None  # always written

    def test_video_keys_are_ignored_on_read(self) -> None:
        with_core_keys = {"video.fps": 30, "video.codec": "av1", **RIGHT_FINGERTIP}
        assert TactileInfo.from_dict(with_core_keys).to_dict() == RIGHT_FINGERTIP

    def test_family_is_an_enum_after_construction(self) -> None:
        info = TactileInfo.from_dict(PALM)
        assert info.family is TactileFamily.MAGNETIC
        assert info.n_channels == 3


class TestInfoValidation:
    @pytest.mark.parametrize("family", ["taxel_array", "magnetic", "force_torque"])
    def test_channels_required_for_array_families(self, family: str) -> None:
        example = {**PALM, "family": family}
        del example["channels"]
        with pytest.raises(SchemaError, match="channels is required"):
            TactileInfo.from_dict(example)

    def test_force_torque_declares_six_channels(self) -> None:
        with pytest.raises(SchemaError, match="six channels"):
            TactileInfo.from_dict({**PALM, "family": "force_torque"})
        ok = {**PALM, "family": "force_torque", "channels": ["fx", "fy", "fz", "mx", "my", "mz"]}
        del ok["layout"]
        assert TactileInfo.from_dict(ok).n_channels == 6

    def test_visuotactile_needs_no_channels(self) -> None:
        assert TactileInfo.from_dict(RIGHT_FINGERTIP).channels is None

    def test_unknown_key_is_rejected_with_path(self) -> None:
        with pytest.raises(SchemaError, match=r"info\.layout: unknown keys \['frame_id'\]"):
            TactileInfo.from_dict({**PALM, "layout": {**PALM["layout"], "frame_id": "x"}})

    def test_missing_required_key(self) -> None:
        example = dict(PALM)
        del example["session_meta"]
        with pytest.raises(SchemaError, match="missing required key 'session_meta'"):
            TactileInfo.from_dict(example)

    def test_wrong_scalar_type(self) -> None:
        with pytest.raises(SchemaError, match="declared_rate_hz: expected a number"):
            TactileInfo.from_dict({**PALM, "declared_rate_hz": "200"})

    def test_bad_family(self) -> None:
        with pytest.raises(SchemaError, match="family must be one of"):
            TactileInfo.from_dict({**PALM, "family": "optical"})

    def test_bad_reduction_method(self) -> None:
        with pytest.raises(SchemaError, match="reduction.method"):
            TactileInfo.from_dict({**PALM, "reduction": {"method": "median", "window_ms": 33.3}})

    def test_normals_must_match_positions(self) -> None:
        layout = {**PALM["layout"], "normals": [[0, 0, 1]] * 4}
        with pytest.raises(SchemaError, match="normals has 4 rows"):
            TactileInfo.from_dict({**PALM, "layout": layout})

    def test_session_meta_path_shape(self) -> None:
        with pytest.raises(SchemaError, match="session_meta"):
            TactileInfo.from_dict({**PALM, "session_meta": "/abs/meta.json"})


class TestStreams:
    def test_derived_streams_require_derived_block(self) -> None:
        example = dict(RIGHT_FINGERTIP)
        del example["derived"]
        info = TactileInfo.from_dict(example)
        info.check_streams(["markers"])
        with pytest.raises(SchemaError, match="require a 'derived' block"):
            info.check_streams(["markers", "force"])
        with pytest.raises(SchemaError, match="require a 'derived' block"):
            info.check_streams(["displacement"])

    def test_derived_block_satisfies_check(self) -> None:
        TactileInfo.from_dict(RIGHT_FINGERTIP).check_streams(["markers", "displacement", "force"])

    def test_array_families_have_no_suffixed_streams(self) -> None:
        with pytest.raises(SchemaError, match="visuotactile instances only"):
            TactileInfo.from_dict(PALM).check_streams(["force"])

    def test_unknown_suffix(self) -> None:
        with pytest.raises(SchemaError, match="unknown stream suffixes"):
            TactileInfo.from_dict(RIGHT_FINGERTIP).check_streams(["depth"])

    def test_stream_info_round_trips_and_is_strict(self) -> None:
        session_meta = RIGHT_FINGERTIP["session_meta"]
        block = {"reduction": {"method": "nearest", "window_ms": 33.3}, "session_meta": session_meta}
        assert StreamInfo.from_dict(block).to_dict() == block
        assert StreamInfo.from_dict({"session_meta": "tactile/x/meta.json"}).to_dict() == {
            "session_meta": "tactile/x/meta.json"
        }
        with pytest.raises(SchemaError, match="unknown keys \\['family'\\]"):
            StreamInfo.from_dict({**block, "family": "visuotactile"})
        with pytest.raises(SchemaError, match="session_meta"):
            StreamInfo(session_meta="meta.json")


class TestLenientReader:
    def test_unknown_keys_are_reported_not_raised(self) -> None:
        layout = {**LEFT_FINGERTIP["layout"], "units": "mm"}
        data = {**LEFT_FINGERTIP, "serial": "PX3-00412", "layout": layout}
        info, unknown = read_info_lenient(data)
        assert info == TactileInfo.from_dict(LEFT_FINGERTIP)
        assert unknown == ["info.serial", "info.layout.units"]

    def test_video_keys_are_not_unknown(self) -> None:
        info, unknown = read_info_lenient({**RIGHT_FINGERTIP, "video.fps": 30, "video.codec": "av1"})
        assert info == TactileInfo.from_dict(RIGHT_FINGERTIP)
        assert unknown == []

    def test_wrong_types_still_raise(self) -> None:
        with pytest.raises(SchemaError, match="declared_rate_hz: expected a number"):
            read_info_lenient({**LEFT_FINGERTIP, "declared_rate_hz": "91"})
        with pytest.raises(SchemaError, match="missing required key 'session_meta'"):
            read_info_lenient({k: v for k, v in LEFT_FINGERTIP.items() if k != "session_meta"})


class TestSessionRoundTrip:
    def test_example_round_trips_exactly(self) -> None:
        meta = SessionMeta.from_dict(LEFT_FINGERTIP_META)
        assert meta.to_dict() == LEFT_FINGERTIP_META
        assert json_round_trip(meta.to_dict()) == LEFT_FINGERTIP_META

    def test_key_order_matches_schema(self) -> None:
        session = SessionMeta.from_dict(LEFT_FINGERTIP_META).sessions[0]
        assert list(session.to_dict()) == list(LEFT_FINGERTIP_META["sessions"][0])

    def test_episodes_is_a_tuple_in_python_and_a_list_in_json(self) -> None:
        session = SessionRecord.from_dict(LEFT_FINGERTIP_META["sessions"][0])
        assert session.episodes == (0, 41)
        assert session.to_dict()["episodes"] == [0, 41]

    def test_provenance_keys_are_always_written(self) -> None:
        record = SessionRecord("s", (0, 0), adapter=None, calibration=None, baseline=None)
        out = record.to_dict()
        assert out["adapter"] is None and out["calibration"] is None and out["baseline"] is None

    def test_provenance_keys_are_required_on_read(self) -> None:
        with pytest.raises(SchemaError, match="missing required key 'baseline'"):
            SessionRecord.from_dict(
                {"session_id": "s", "episodes": [0, 0], "adapter": None, "calibration": None}
            )

    def test_visuotactile_baseline_is_a_reference_frame(self) -> None:
        session = copy.deepcopy(LEFT_FINGERTIP_META["sessions"][0])
        session["baseline"] = {
            "captured_at": "2026-11-21T14:03:11Z",
            "duration_s": 2.0,
            "applied": False,
            "reference_frame": "episode_000000/frame_0000",
        }
        assert SessionRecord.from_dict(session).to_dict()["baseline"] == session["baseline"]


class TestSessionValidation:
    def test_applied_requires_stored_values(self) -> None:
        with pytest.raises(SchemaError, match="applied is true but no values"):
            Baseline("2026-11-21T14:03:11Z", 2.0, applied=True, reference_frame="episode_000000/frame_0000")

    def test_baseline_needs_exactly_one_representation(self) -> None:
        with pytest.raises(SchemaError, match="exactly one"):
            Baseline("t", 2.0, applied=False)
        with pytest.raises(SchemaError, match="exactly one"):
            Baseline("t", 2.0, applied=False, values=[[0.0]], reference_frame="x")

    def test_baseline_values_must_be_rectangular(self) -> None:
        with pytest.raises(SchemaError, match="baseline.values"):
            Baseline("t", 2.0, applied=False, values=[[0.0, 1.0], [0.0]])

    def test_episode_range_order(self) -> None:
        with pytest.raises(SchemaError, match="0 <= first <= last"):
            SessionRecord("s", (5, 2), adapter=None, calibration=None, baseline=None)

    def test_sessions_must_not_overlap(self) -> None:
        a = SessionRecord("a", (0, 10), adapter=None, calibration=None, baseline=None)
        b = SessionRecord("b", (10, 20), adapter=None, calibration=None, baseline=None)
        with pytest.raises(SchemaError, match="overlap"):
            SessionMeta("left_fingertip", [a, b])
        SessionMeta("left_fingertip", [a, SessionRecord("c", (11, 20), None, None, None)])

    def test_instance_id_is_a_slug(self) -> None:
        with pytest.raises(SchemaError, match="instance_id"):
            SessionMeta("left/fingertip", [])

    def test_nested_unknown_key_reports_full_path(self) -> None:
        meta = copy.deepcopy(LEFT_FINGERTIP_META)
        meta["sessions"][0]["calibration"]["expires"] = "never"
        with pytest.raises(
            SchemaError, match=r"meta\.sessions\[0\]\.calibration: unknown keys \['expires'\]"
        ):
            SessionMeta.from_dict(meta)

    def test_adapter_and_calibration_fields(self) -> None:
        with pytest.raises(SchemaError, match="adapter.version"):
            Adapter("taxel.simulated", "")
        assert Calibration().to_dict() == {"version": None, "method": None, "date": None, "reference": None}


class TestHelpers:
    def test_feature_key(self) -> None:
        assert feature_key("left_fingertip") == "observation.tactile.left_fingertip"
        assert feature_key("right_fingertip", "markers") == "observation.tactile.right_fingertip.markers"
        with pytest.raises(SchemaError):
            feature_key("right_fingertip", "depth")
        with pytest.raises(SchemaError):
            feature_key("left/fingertip")

    def test_session_meta_path(self) -> None:
        assert session_meta_path("palm_0") == "tactile/palm_0/meta.json"
