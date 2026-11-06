"""Round-trip and validation tests for ``taxel.schema`` against the examples in ``SCHEMA.md``."""

import copy
import json

import pytest

from taxel.schema import (
    CHECK_NAMES,
    Adapter,
    Axes,
    Baseline,
    BaselineStream,
    Calibration,
    Clock,
    DriftReport,
    FrameCounts,
    GelReport,
    HealthReport,
    InstanceChecks,
    Layout,
    Origin,
    Range,
    SchemaError,
    SessionMeta,
    SessionRecord,
    Skipped,
    StreamInfo,
    TactileFamily,
    TactileInfo,
    Thresholds,
    feature_key,
    health_report_path,
    read_info_lenient,
    read_session_meta_lenient,
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
    "layout": {
        "origin": {"at": "corner", "description": "taxel 0, the corner of the array nearest the connector"},
        "axes": {
            "x": "along the long side of the array, away from the connector",
            "y": "across the array, from taxel 0 toward taxel 8",
            "z": "out of the sensing surface, toward the contacting object",
        },
        "surface": "pad",
        "positions_mm": positions(N_TAXELS),
        "normals": [[0, 0, 1]] * N_TAXELS,
    },
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
    "layout": {
        "origin": {"at": "corner", "description": "the top-left pixel of the image"},
        "axes": {
            "x": "along an image row, toward the right",
            "y": "down an image column",
            "z": "out of the gel, toward the contacting object",
        },
        "surface": "gel",
    },
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
        "origin": {"at": "element", "description": "the central magnetometer"},
        "axes": {
            "x": "across the board, from the central magnetometer toward the second",
            "y": "across the board, from the central magnetometer toward the fourth",
            "z": "out of the skin, toward the contacting object",
        },
        "surface": "die",
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
            "clock": {
                "wall_time": "2026-11-21T14:03:05.117Z",
                "host_clock": 10234.117,
                "episode_origins": [10240.004 + 61.5 * i for i in range(42)],
            },
            "frames": {"received": 30412, "dropped": 7, "overflowed": 0},
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
                "frame_range": [364, 545],
                "values": [[0.01, -0.02, 0.00]] * N_TAXELS,
            },
            "baseline_stream": {"sample_every": 100},
            "gel": None,
            "skin": None,
        }
    ],
}


FRAME = {k: PALM["layout"][k] for k in ("origin", "axes", "surface")}


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
        assert "channels" not in out and "units" not in out
        assert "positions_mm" not in out["layout"] and "normals" not in out["layout"]
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
        channels = ["fx", "fy", "fz", "mx", "my", "mz"]
        ok = {**PALM, "family": "force_torque", "channels": channels, "layout": FRAME}
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


class TestLayout:
    def test_layout_is_required_for_every_family(self) -> None:
        for example in (LEFT_FINGERTIP, RIGHT_FINGERTIP, PALM):
            without = {k: v for k, v in example.items() if k != "layout"}
            with pytest.raises(SchemaError, match="layout is required"):
                TactileInfo.from_dict(without)

    def test_origin_and_surface_are_closed_enums(self) -> None:
        bad_origin = {**PALM["layout"], "origin": {"at": "top-left", "description": "x"}}
        with pytest.raises(SchemaError, match=r"info\.layout\.origin: origin\.at must be one of"):
            TactileInfo.from_dict({**PALM, "layout": bad_origin})
        with pytest.raises(SchemaError, match=r"info\.layout: layout\.surface must be one of"):
            TactileInfo.from_dict({**PALM, "layout": {**PALM["layout"], "surface": "elastomer"}})
        for surface in ("gel", "die", "skin", "pad", "vendor"):
            assert Layout.from_dict({**FRAME, "surface": surface}).surface == surface
        for at in ("center", "corner", "element", "mount"):
            assert Origin(at, "somewhere").at == at

    def test_frame_text_is_required(self) -> None:
        with pytest.raises(SchemaError, match="origin.description must be a non-empty string"):
            Origin("corner", "")
        with pytest.raises(SchemaError, match="axes.z must be a non-empty string"):
            Axes("x", "y", "")
        axes = {k: v for k, v in PALM["layout"]["axes"].items() if k != "z"}
        with pytest.raises(SchemaError, match=r"info\.layout\.axes: missing required key 'z'"):
            TactileInfo.from_dict({**PALM, "layout": {**PALM["layout"], "axes": axes}})

    @pytest.mark.parametrize("family", ["taxel_array", "magnetic"])
    def test_positions_required_for_element_arrays(self, family: str) -> None:
        with pytest.raises(SchemaError, match=f"positions_mm is required for family '{family}'"):
            TactileInfo.from_dict({**PALM, "family": family, "layout": FRAME})

    def test_positions_omitted_for_visuotactile(self) -> None:
        layout = {**RIGHT_FINGERTIP["layout"], "positions_mm": [[0, 0, 0]]}
        with pytest.raises(SchemaError, match="omitted for visuotactile"):
            TactileInfo.from_dict({**RIGHT_FINGERTIP, "layout": layout})

    def test_force_torque_has_at_most_one_position(self) -> None:
        wrench = {**PALM, "family": "force_torque", "channels": ["fx", "fy", "fz", "mx", "my", "mz"]}
        with pytest.raises(SchemaError, match="one element; layout.positions_mm has 5 rows"):
            TactileInfo.from_dict(wrench)
        one = {**wrench, "layout": {**FRAME, "positions_mm": [[0, 0, 0]], "normals": None}}
        assert TactileInfo.from_dict(one).to_dict()["layout"] == one["layout"]

    def test_normals_only_beside_positions(self) -> None:
        assert Layout.from_dict(FRAME).to_dict() == FRAME
        with_positions = Layout.from_dict({**FRAME, "positions_mm": [[0, 0, 0]]}).to_dict()
        assert with_positions == {**FRAME, "positions_mm": [[0, 0, 0]], "normals": None}
        with pytest.raises(SchemaError, match="normals is given without positions_mm"):
            Layout.from_dict({**FRAME, "normals": [[0, 0, 1]]})


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
            "frame_range": None,
            "reference_frame": "episode_000000/frame_0000",
        }
        assert SessionRecord.from_dict(session).to_dict()["baseline"] == session["baseline"]


class TestSessionValidation:
    def test_applied_requires_stored_values(self) -> None:
        with pytest.raises(SchemaError, match="applied is true but no values"):
            Baseline("2026-11-21T14:03:11Z", 2.0, True, None, reference_frame="episode_000000/frame_0000")

    def test_baseline_needs_exactly_one_representation(self) -> None:
        with pytest.raises(SchemaError, match="exactly one"):
            Baseline("t", 2.0, applied=False, frame_range=None)
        with pytest.raises(SchemaError, match="exactly one"):
            Baseline("t", 2.0, False, None, values=[[0.0]], reference_frame="x")

    def test_baseline_values_must_be_rectangular(self) -> None:
        with pytest.raises(SchemaError, match="baseline.values"):
            Baseline("t", 2.0, False, None, values=[[0.0, 1.0], [0.0]])

    def test_frame_range_is_an_ordered_seq_pair(self) -> None:
        ok = Baseline("t", 2.0, False, [7, 7], values=[[0.0]])
        assert ok.frame_range == (7, 7) and ok.to_dict()["frame_range"] == [7, 7]
        for bad in ([5, 2], [-1, 3], [1.5, 2], [1], [True, 2]):
            with pytest.raises(SchemaError, match="baseline.frame_range"):
                Baseline("t", 2.0, False, bad, values=[[0.0]])

    def test_frame_range_is_required_on_a_strict_read(self) -> None:
        session = copy.deepcopy(LEFT_FINGERTIP_META["sessions"][0])
        del session["baseline"]["frame_range"]
        with pytest.raises(SchemaError, match="missing required key 'frame_range'"):
            SessionRecord.from_dict(session)

    def test_baseline_stream_period(self) -> None:
        assert BaselineStream.from_dict({"sample_every": 100}).to_dict() == {"sample_every": 100}
        for bad in (0, -5, 2.5, True):
            with pytest.raises(SchemaError, match="sample_every must be a positive integer"):
                BaselineStream(bad)
        record = SessionRecord("s", (0, 0), adapter=None, calibration=None, baseline=None)
        assert record.to_dict()["baseline_stream"] is None
        assert SessionRecord.from_dict(record.to_dict()) == record

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


class TestSessionLenientReader:
    def test_missing_frame_range_is_tolerated_and_reported(self) -> None:
        older = copy.deepcopy(LEFT_FINGERTIP_META)
        del older["sessions"][0]["baseline"]["frame_range"]
        meta, unknown, missing = read_session_meta_lenient(older)
        assert meta.sessions[0].baseline.frame_range is None
        assert meta.sessions[0].baseline.values == LEFT_FINGERTIP_META["sessions"][0]["baseline"]["values"]
        assert unknown == [] and missing == ["meta.sessions[0].baseline.frame_range"]
        with pytest.raises(SchemaError, match="frame_range"):
            SessionMeta.from_dict(older)

    def test_unknown_keys_are_reported_not_raised(self) -> None:
        data = copy.deepcopy(LEFT_FINGERTIP_META)
        data["sessions"][0]["operator"] = "vic"
        data["sessions"][0]["calibration"]["expires"] = "never"
        meta, unknown, missing = read_session_meta_lenient(data)
        assert meta == SessionMeta.from_dict(LEFT_FINGERTIP_META)
        assert unknown == ["meta.sessions[0].operator", "meta.sessions[0].calibration.expires"]
        assert missing == []

    def test_other_missing_keys_and_wrong_types_still_raise(self) -> None:
        data = copy.deepcopy(LEFT_FINGERTIP_META)
        del data["sessions"][0]["baseline"]["applied"]
        with pytest.raises(SchemaError, match="missing required key 'applied'"):
            read_session_meta_lenient(data)
        data = copy.deepcopy(LEFT_FINGERTIP_META)
        data["sessions"][0]["baseline"]["frame_range"] = "364-545"
        with pytest.raises(SchemaError, match="frame_range"):
            read_session_meta_lenient(data)


class TestClockAndFrames:
    def test_clock_validation(self) -> None:
        Clock(wall_time="2026-11-21T14:03:05Z", host_clock=1, episode_origins=[])
        with pytest.raises(SchemaError, match="clock.wall_time must be UTC ISO 8601"):
            Clock(wall_time="2026-11-21 14:03:05", host_clock=1.0, episode_origins=[])
        with pytest.raises(SchemaError, match="clock.wall_time must be UTC ISO 8601"):
            Clock(wall_time="2026-11-21T14:03:05+01:00", host_clock=1.0, episode_origins=[])
        with pytest.raises(SchemaError, match="clock.host_clock must be a number"):
            Clock(wall_time="2026-11-21T14:03:05Z", host_clock="1.0", episode_origins=[])
        with pytest.raises(SchemaError, match="clock.episode_origins must be numbers"):
            Clock(wall_time="2026-11-21T14:03:05Z", host_clock=1.0, episode_origins=[1.0, None])

    def test_frames_validation(self) -> None:
        FrameCounts(received=0, dropped=0, overflowed=0)
        for bad in ({"received": -1}, {"dropped": 1.5}, {"overflowed": True}):
            with pytest.raises(SchemaError, match="must be a non-negative integer"):
                FrameCounts(**{"received": 0, "dropped": 0, "overflowed": 0, **bad})

    def test_one_origin_per_episode(self) -> None:
        record = dict(LEFT_FINGERTIP_META["sessions"][0])
        record["clock"] = {**record["clock"], "episode_origins": [10240.0]}
        with pytest.raises(SchemaError, match="episode_origins has 1 values for 42 episodes"):
            SessionRecord.from_dict(record)

    def test_clock_and_frames_are_written_as_null_when_unknown(self) -> None:
        record = SessionRecord(session_id="s", episodes=(0, 0), adapter=None, calibration=None, baseline=None)
        encoded = record.to_dict()
        assert list(encoded)[:4] == ["session_id", "episodes", "clock", "frames"]
        assert list(encoded)[-4:] == ["baseline", "baseline_stream", "gel", "skin"]
        assert encoded["clock"] is None and encoded["frames"] is None
        assert SessionRecord.from_dict(encoded) == record


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


class TestRange:
    def test_round_trips_scalar_and_per_channel(self) -> None:
        for limits in ({"min": -5.0, "max": 25.5}, {"min": [-5.0, -5.0, 0.0], "max": [5.0, 5.0, 25.5]}):
            data = {**LEFT_FINGERTIP, "range": limits}
            info = TactileInfo.from_dict(data)
            assert info.range is not None and info.to_dict() == data
            keys = list(info.to_dict())
            assert keys.index("range") == keys.index("channels") + 1
        assert "range" not in TactileInfo.from_dict(LEFT_FINGERTIP).to_dict()

    def test_limits_broadcast_per_channel(self) -> None:
        lo, hi = Range(min=-1.0, max=2.0).limits(3)
        assert lo.tolist() == [-1.0, -1.0, -1.0] and hi.tolist() == [2.0, 2.0, 2.0]
        lo, hi = Range(min=[-1.0, -2.0], max=3.0).limits(2)
        assert lo.tolist() == [-1.0, -2.0] and hi.tolist() == [3.0, 3.0]
        assert Range(min=[-1.0, -2.0], max=3.0).n_channels == 2 and Range(min=0, max=1).n_channels is None

    def test_validation(self) -> None:
        with pytest.raises(SchemaError, match="below range.max"):
            Range(min=1.0, max=1.0)
        with pytest.raises(SchemaError, match="below range.max"):
            Range(min=[0.0, 2.0], max=[1.0, 1.0])
        with pytest.raises(SchemaError, match="2 values and range.max 3"):
            Range(min=[0.0, 0.0], max=[1.0, 1.0, 1.0])
        with pytest.raises(SchemaError, match="number or a list"):
            Range(min="0", max=1.0)
        with pytest.raises(SchemaError, match="range lists 2 values for 3 channels"):
            TactileInfo.from_dict({**LEFT_FINGERTIP, "range": {"min": [0.0, 0.0], "max": [1.0, 1.0]}})
        with pytest.raises(SchemaError, match="array families"):
            TactileInfo.from_dict({**RIGHT_FINGERTIP, "range": {"min": 0.0, "max": 255.0}})


# The report example from SCHEMA.md, "Health annotations".
HEALTH = {
    "doctor_version": "0.1.0",
    "generated_at": "2026-11-05T14:03:05Z",
    "thresholds": {
        "drift": 0.05,
        "drift_units": None,
        "dropped": 0.02,
        "max_gap_ms": None,
        "gel": 0.05,
        "cluster_px": 25,
        "skew": 50.0,
        "confidence": 0.6,
        "saturated": 0.01,
        "integrity": True,
    },
    "instances": {
        "left_fingertip": {
            "key": "observation.tactile.left_fingertip",
            "family": "taxel_array",
            "metadata": "info",
            "sidecar": True,
            "declared_rate_hz": 91,
            "range": None,
            "unknown_keys": [],
            "missing_keys": ["meta.sessions[0].baseline.frame_range"],
            "error": None,
        }
    },
    "episodes": {
        "episode_000012": {
            "left_fingertip": {
                "integrity": {
                    "copy": {
                        "frames": 612,
                        "mismatches": 0,
                        "first_mismatch": None,
                        "max_abs_diff": 0.0,
                        "flag": False,
                    },
                    "raw": {
                        "samples": 7,
                        "mismatches": 0,
                        "first_mismatch": None,
                        "max_abs_diff": 0.0,
                        "flag": False,
                    },
                    "files": {
                        "native": True,
                        "baseline": True,
                        "capture_at_zero": True,
                        "origins": True,
                        "counters": True,
                        "flag": False,
                    },
                    "flag": False,
                },
                "drift": {
                    "no_contact_level": [0.011, 0.008, 0.029],
                    "level_change": [0.001, 0.001, 0.002],
                    "offset": [0.012, 0.009, 0.031],
                    "offset_pct_range": None,
                    "slope_per_s": [0.001, 0.0, 0.002],
                    "baseline": "capture",
                    "source": "samples",
                    "baseline_age_s": 3.2,
                    "frame_range": "not recorded",
                    "windows": 20,
                    "no_contact_windows": 14,
                    "candidates": 91,
                    "first_frame_zero": False,
                    "rebaselines": [{"sample_index": 1200, "host_ts": 10301.1, "delta": [0.0, 0.0, 0.03]}],
                    "confidence": "normal",
                    "flag": False,
                },
                "framerate": {
                    "effective_hz": 89.4,
                    "declared_hz": 91,
                    "dropped_pct": 0.018,
                    "max_gap_ms": 31.0,
                    "max_gap_threshold_ms": 109.9,
                    "gap_histogram": {"edges_periods": [1.5, 2.5, 3.5], "counts": [600, 9, 2, 0]},
                    "frames": {"received": 30412, "dropped": 7, "overflowed": 0},
                    "source": "sidecar",
                    "timestamps": "measured",
                    "flag": False,
                },
                "saturation": {
                    "saturated_pct": [0.0, 0.0, 0.0],
                    "limit": "observed",
                    "min": [-0.4, -0.5, -0.1],
                    "max": [3.1, 2.9, 24.8],
                    "confidence": "low",
                    "source": "sidecar",
                    "flag": False,
                },
                "gel": {"skipped": "no visuotactile feature"},
                "skew": {
                    "lag_ms": -12.0,
                    "confidence": 0.81,
                    "reference": "observation.images.front",
                    "onsets": 3,
                    "resolution_ms": 5.0,
                    "flag": False,
                },
            }
        }
    },
}


class TestHealthReport:
    def test_example_round_trips_exactly(self) -> None:
        report = HealthReport.from_dict(HEALTH)
        assert report.to_dict() == HEALTH
        assert json.loads(json.dumps(report.to_dict())) == HEALTH
        checks = report.episodes["episode_000012"]["left_fingertip"]
        assert isinstance(checks, InstanceChecks) and isinstance(checks.drift, DriftReport)
        assert isinstance(checks.gel, Skipped) and checks.gel.skipped == "no visuotactile feature"
        assert checks.drift.frame_range == "not recorded" and checks.drift.rebaselines[0].sample_index == 1200
        assert [f for f in checks.to_dict()] == list(CHECK_NAMES)

    def test_skipped_is_told_apart_by_its_key(self) -> None:
        data = copy.deepcopy(HEALTH)
        block = data["episodes"]["episode_000012"]["left_fingertip"]
        block["integrity"]["raw"] = {"skipped": "no baseline stream"}
        block["drift"]["offset"] = {"skipped": "no session baseline"}
        report = HealthReport.from_dict(data)
        checks = report.episodes["episode_000012"]["left_fingertip"]
        assert isinstance(checks.integrity.raw, Skipped) and isinstance(checks.drift.offset, Skipped)
        assert report.to_dict() == data
        block["drift"] = {"skipped": "x", "flag": False}  # a flag on a skipped check is not a shape
        with pytest.raises(SchemaError, match="drift: unknown keys \\['flag'\\]"):
            HealthReport.from_dict(data)
        block["drift"] = {"no_contact_level": [0.0]}  # closest shape: the error names the missing key
        with pytest.raises(SchemaError, match="drift: missing required key 'level_change'"):
            HealthReport.from_dict(data)
        block["drift"] = {"skipped": ""}
        with pytest.raises(SchemaError, match="non-empty string"):
            HealthReport.from_dict(data)

    def test_frame_range_takes_a_pair_or_a_word(self) -> None:
        data = copy.deepcopy(HEALTH)
        drift = data["episodes"]["episode_000012"]["left_fingertip"]["drift"]
        for value in ([364, 545], "no seq", None):
            drift["frame_range"] = value
            assert HealthReport.from_dict(data).to_dict() == data
        drift["frame_range"] = 3.5
        with pytest.raises(SchemaError, match="fits none"):
            HealthReport.from_dict(data)

    def test_thresholds_defaults_and_enums(self) -> None:
        assert Thresholds().to_dict() == HEALTH["thresholds"]
        with pytest.raises(SchemaError, match="gel.kind"):
            GelReport(
                kind="voxels",
                dead_pct=0.0,
                largest_cluster_px=None,
                dead_elements=None,
                frames_used=1,
                flag=False,
            )
        data = copy.deepcopy(HEALTH)
        data["episodes"]["episode_000012"]["left_fingertip"]["saturation"]["limit"] = "guessed"
        with pytest.raises(SchemaError, match="saturation.limit"):
            HealthReport.from_dict(data)
        data = copy.deepcopy(HEALTH)
        data["instances"]["left_fingertip"]["metadata"] = "partial"
        with pytest.raises(SchemaError, match="instance.metadata"):
            HealthReport.from_dict(data)

    def test_unknown_keys_and_bad_episode_names_are_rejected(self) -> None:
        data = copy.deepcopy(HEALTH)
        data["episodes"]["episode_000012"]["left_fingertip"]["drift"]["trend"] = 1.0
        with pytest.raises(SchemaError, match="unknown keys"):
            HealthReport.from_dict(data)
        data = copy.deepcopy(HEALTH)
        data["episodes"]["ep12"] = data["episodes"].pop("episode_000012")
        with pytest.raises(SchemaError, match="episode_000012"):
            HealthReport.from_dict(data)
        with pytest.raises(SchemaError, match="unknown keys"):
            HealthReport.from_dict({**HEALTH, "notes": "x"})

    def test_path(self) -> None:
        assert str(health_report_path("outputs/name")) == "outputs/name/tactile/health.json"
