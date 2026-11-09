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
    Channel,
    ChannelReport,
    Clock,
    DriftReport,
    FrameCounts,
    GelReport,
    HealthReport,
    Indexing,
    InstanceChecks,
    Layout,
    Origin,
    Range,
    Region,
    SchemaError,
    SessionMeta,
    SessionRecord,
    Skipped,
    StreamInfo,
    TactileFamily,
    TactileInfo,
    Thresholds,
    Transport,
    check_device_indices,
    check_session_calibration,
    feature_key,
    health_report_path,
    read_info_lenient,
    read_session_meta_lenient,
    session_meta_path,
    without_core_keys,
)

N_TAXELS = 52


def positions(n: int) -> list[list[float]]:
    return [[1.5 * i, 0.0, 0.0] for i in range(n)]


def channels(unit: str, *names: str, source: str = "measured") -> list[dict]:
    return [{"name": name, "unit": unit, "source": source} for name in names]


# The four feature entries from SCHEMA.md, "Static metadata", with the "..." rows filled in.
LEFT_FINGERTIP = {
    "family": "taxel_array",
    "vendor": "PaXini",
    "model": "PX-6AX-GEN3 DP-S2015-Elite",
    "declared_rate_hz": 91,
    "channels": channels("N", "fx", "fy", "fz"),
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
    "device": {"id": "paxini_hub_0", "clock": "paxini_hub_0", "index": 0},
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

RIGHT_FINGERTIP_DISPLACEMENT = {
    "channels": channels("mm", "dx", "dy", "dz", source="derived"),
    "reduction": {"method": "mean", "window_ms": 33.3},
    "session_meta": "tactile/right_fingertip/meta.json",
}

PALM = {
    "family": "magnetic",
    "vendor": "NYU / AnySkin",
    "model": "AnySkin v1",
    "declared_rate_hz": 200,
    "channels": channels("uT", "bx", "by", "bz"),
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
            "transport": {"kind": "usb", "rate_hz": 91},
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


# The field-note fragments from SCHEMA.md, "Static metadata": a converted three-axis force kept as
# force_torque, and the PaXini GEN3-2 fingertip and palm as two regions.
CONVERTED_FORCE_CHANNELS = channels("N", "fx", "fy", "fz") + channels(
    "N.m", "mx", "my", "mz", source="unavailable"
)
GEN3_2_INDEXING = {
    "kind": "regions",
    "regions": [
        {
            "name": "fingertip",
            "shape": [8, 8],
            "pitch_mm": [1.75, 1.75],
            "first_element": 0,
            "area_mm2": 3.0625,
        },
        {"name": "palm", "shape": [16, 12], "pitch_mm": [3.0, 3.0], "first_element": 64, "area_mm2": 9.0},
    ],
}

FRAME = {k: PALM["layout"][k] for k in ("origin", "axes", "surface")}
WRENCH = {**PALM, "family": "force_torque", "channels": channels("N", "fx", "fy", "fz", "mx", "my", "mz")}


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

    def test_lerobot_keys_are_set_aside_on_a_strict_read(self) -> None:
        """LeRobot 0.6.1 writes has_audio and is_depth_map on every video feature, and audio.* when the
        file has sound, beside its video.* keys; the strict reader sets all of them aside."""
        core = {
            "video.fps": 30,
            "video.codec": "av1",
            "has_audio": False,
            "is_depth_map": False,
            "audio.codec": "aac",
            "audio.sample_rate": 48000,
        }
        assert TactileInfo.from_dict({**core, **RIGHT_FINGERTIP}).to_dict() == RIGHT_FINGERTIP
        assert without_core_keys({**core, "family": "visuotactile"}) == {"family": "visuotactile"}
        with pytest.raises(SchemaError, match=r"unknown keys \['has_video'\]"):
            TactileInfo.from_dict({**RIGHT_FINGERTIP, "has_video": True})

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
        assert TactileInfo.from_dict({**WRENCH, "layout": FRAME}).n_channels == 6

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
        with pytest.raises(
            SchemaError, match=f"positions_mm or layout.indexing is required for family '{family}'"
        ):
            TactileInfo.from_dict({**PALM, "family": family, "layout": FRAME})

    def test_positions_omitted_for_visuotactile(self) -> None:
        layout = {**RIGHT_FINGERTIP["layout"], "positions_mm": [[0, 0, 0]]}
        with pytest.raises(SchemaError, match="omitted for visuotactile"):
            TactileInfo.from_dict({**RIGHT_FINGERTIP, "layout": layout})

    def test_force_torque_has_at_most_one_position(self) -> None:
        with pytest.raises(SchemaError, match="one element; layout.positions_mm has 5 rows"):
            TactileInfo.from_dict(WRENCH)
        one = {**WRENCH, "layout": {**FRAME, "positions_mm": [[0, 0, 0]], "normals": None}}
        assert TactileInfo.from_dict(one).to_dict()["layout"] == one["layout"]

    def test_normals_only_beside_positions(self) -> None:
        assert Layout.from_dict(FRAME).to_dict() == FRAME
        with_positions = Layout.from_dict({**FRAME, "positions_mm": [[0, 0, 0]]}).to_dict()
        assert with_positions == {**FRAME, "positions_mm": [[0, 0, 0]], "normals": None}
        with pytest.raises(SchemaError, match="normals is given without positions_mm"):
            Layout.from_dict({**FRAME, "normals": [[0, 0, 1]]})


class TestStreams:
    def test_every_suffixed_stream_requires_derived_block(self) -> None:
        example = dict(RIGHT_FINGERTIP)
        del example["derived"]
        info = TactileInfo.from_dict(example)
        info.check_streams([])
        for suffixes in (["markers"], ["markers", "force"], ["displacement"]):
            with pytest.raises(SchemaError, match="require a 'derived' block"):
                info.check_streams(suffixes)

    def test_derived_block_satisfies_check(self) -> None:
        TactileInfo.from_dict(RIGHT_FINGERTIP).check_streams(["markers", "displacement", "force"])

    def test_array_families_have_no_suffixed_streams(self) -> None:
        with pytest.raises(SchemaError, match="visuotactile instances only"):
            TactileInfo.from_dict(PALM).check_streams(["force"])

    def test_unknown_suffix(self) -> None:
        with pytest.raises(SchemaError, match="unknown stream suffixes"):
            TactileInfo.from_dict(RIGHT_FINGERTIP).check_streams(["depth"])

    def test_stream_info_round_trips_and_is_strict(self) -> None:
        block = RIGHT_FINGERTIP_DISPLACEMENT
        assert StreamInfo.from_dict(block).to_dict() == block
        assert list(StreamInfo.from_dict(block).to_dict()) == ["channels", "reduction", "session_meta"]
        without_reduction = {k: v for k, v in block.items() if k != "reduction"}
        assert StreamInfo.from_dict(without_reduction).to_dict() == without_reduction
        with pytest.raises(SchemaError, match="unknown keys \\['family'\\]"):
            StreamInfo.from_dict({**block, "family": "visuotactile"})
        with pytest.raises(SchemaError, match="missing required key 'channels'"):
            StreamInfo.from_dict({"session_meta": "tactile/x/meta.json"})
        with pytest.raises(SchemaError, match="session_meta"):
            StreamInfo(session_meta="meta.json", channels=[Channel("dx", "mm", "derived")])

    def test_stream_channels_are_derived(self) -> None:
        measured = {**RIGHT_FINGERTIP_DISPLACEMENT, "channels": channels("mm", "dx", "dy", "dz")}
        with pytest.raises(SchemaError, match=r"channels \['dx', 'dy', 'dz'\] must be 'derived'"):
            StreamInfo.from_dict(measured)
        with pytest.raises(SchemaError, match="declares its channels"):
            StreamInfo(session_meta="tactile/x/meta.json", channels=[])


class TestLenientReader:
    def test_unknown_keys_are_reported_not_raised(self) -> None:
        layout = {**LEFT_FINGERTIP["layout"], "units": "mm"}
        data = {**LEFT_FINGERTIP, "serial": "PX3-00412", "layout": layout}
        info, unknown, missing, upgraded = read_info_lenient(data)
        assert info == TactileInfo.from_dict(LEFT_FINGERTIP)
        assert unknown == ["info.serial", "info.layout.units"]
        assert missing == [] and not upgraded

    def test_lerobot_keys_are_not_unknown(self) -> None:
        core = {"video.fps": 30, "video.codec": "av1", "has_audio": False, "is_depth_map": False}
        info, unknown, missing, upgraded = read_info_lenient({**RIGHT_FINGERTIP, **core})
        assert info == TactileInfo.from_dict(RIGHT_FINGERTIP)
        assert unknown == [] and missing == [] and not upgraded

    def test_wrong_types_still_raise(self) -> None:
        with pytest.raises(SchemaError, match="declared_rate_hz: expected a number"):
            read_info_lenient({**LEFT_FINGERTIP, "declared_rate_hz": "91"})
        with pytest.raises(SchemaError, match="missing required key 'session_meta'"):
            read_info_lenient({k: v for k, v in LEFT_FINGERTIP.items() if k != "session_meta"})


# LEFT_FINGERTIP as SCHEMA.md 0.3.1 wrote it: channel names, units per quantity, no device.index.
OLD_LEFT_FINGERTIP = {
    **{k: v for k, v in LEFT_FINGERTIP.items() if k != "channels"},
    "units": {"force": "N"},
    "channels": ["fx", "fy", "fz"],
    "device": {"id": "paxini_hub_0", "clock": "paxini_hub_0"},
}


class TestLenientUpgrade:
    def test_a_0_3_1_entry_reads_as_channel_objects_with_source_unknown(self) -> None:
        info, unknown, missing, upgraded = read_info_lenient(OLD_LEFT_FINGERTIP)
        assert upgraded and unknown == [] and missing == ["info.device.index"]
        assert [c.to_dict() for c in info.channels] == channels("N", "fx", "fy", "fz", source="unknown")
        assert info.device.index is None
        assert info.layout == TactileInfo.from_dict(LEFT_FINGERTIP).layout

    def test_the_strict_reader_rejects_the_0_3_form_and_the_upgrade(self) -> None:
        with pytest.raises(SchemaError, match="a list of names is the 0.3 form"):
            TactileInfo.from_dict(OLD_LEFT_FINGERTIP)
        info, _, _, _ = read_info_lenient(OLD_LEFT_FINGERTIP)
        with pytest.raises(SchemaError, match="source must be one of"):
            TactileInfo.from_dict(info.to_dict())
        with pytest.raises(SchemaError, match="missing required key 'index'"):
            TactileInfo.from_dict({**LEFT_FINGERTIP, "device": OLD_LEFT_FINGERTIP["device"]})

    def test_units_are_matched_by_quantity_and_never_guessed(self) -> None:
        old = {**WRENCH, "layout": FRAME, "units": {"force": "N", "torque": "N.m"}}
        for names in (["fx", "fy", "fz", "mx", "my", "mz"], ["fx", "fy", "fz", "tx", "ty", "tz"]):
            info, _, _, upgraded = read_info_lenient({**old, "channels": names})
            assert upgraded and [c.unit for c in info.channels] == ["N"] * 3 + ["N.m"] * 3
            assert {c.source for c in info.channels} == {"unknown"}
        info, _, _, _ = read_info_lenient(
            {**old, "channels": ["fx", "fy", "fz", "mx", "my", "mz"], "units": {"force": "N", "field": "uT"}}
        )
        assert [c.unit for c in info.channels] == ["N"] * 3 + [None] * 3
        no_block = {k: v for k, v in PALM.items() if k != "channels"} | {"channels": ["bx", "by", "bz"]}
        info, _, _, upgraded = read_info_lenient(no_block)
        assert upgraded and [c.unit for c in info.channels] == [None] * 3

    def test_a_units_block_on_an_image_entry_is_reported(self) -> None:
        info, unknown, _, upgraded = read_info_lenient({**RIGHT_FINGERTIP, "units": {"intensity": "counts"}})
        assert unknown == ["info.units"] and not upgraded and info.channels is None


class TestChannels:
    def test_channel_validation(self) -> None:
        for source in ("measured", "derived", "unavailable"):
            assert Channel("fx", "N", source).source == source
        assert Channel("p", "kPa", "measured").is_pressure and not Channel("fz", "N", "measured").is_pressure
        with pytest.raises(SchemaError, match="source must be one of"):
            Channel("fx", "N", "unknown")
        with pytest.raises(SchemaError, match="unit must be a non-empty string"):
            Channel("fx", None, "measured")
        with pytest.raises(SchemaError, match="channel.name"):
            Channel("", "N", "measured")

    def test_names_are_unique_and_channels_are_objects(self) -> None:
        with pytest.raises(SchemaError, match="unique names"):
            TactileInfo.from_dict({**PALM, "channels": channels("uT", "bx", "bx", "bz")})
        layout = Layout.from_dict(PALM["layout"])
        with pytest.raises(SchemaError, match="0.3 form"):
            TactileInfo("magnetic", 200, PALM["session_meta"], channels=["bx", "by", "bz"], layout=layout)

    def test_a_derived_channel_requires_the_derived_block(self) -> None:
        estimated = {**WRENCH, "layout": FRAME, "channels": channels("N", "fx", "fy", "fz", source="derived")}
        estimated["channels"] += channels("N.m", "mx", "my", "mz", source="derived")
        with pytest.raises(SchemaError, match="requires a 'derived' block"):
            TactileInfo.from_dict(estimated)
        named = {**estimated, "derived": {"method": "vendor-estimate", "version": "2.0"}}
        assert TactileInfo.from_dict(named).to_dict() == named

    def test_converted_force_keeps_six_channels_with_unavailable_moments(self) -> None:
        info = TactileInfo.from_dict({**WRENCH, "layout": FRAME, "channels": CONVERTED_FORCE_CHANNELS})
        assert [c.source for c in info.channels] == ["measured"] * 3 + ["unavailable"] * 3
        assert info.to_dict()["channels"] == CONVERTED_FORCE_CHANNELS

    def test_an_image_has_no_channels_and_only_an_image_is_processed(self) -> None:
        with pytest.raises(SchemaError, match="image entry has no channels"):
            TactileInfo.from_dict({**RIGHT_FINGERTIP, "channels": channels("counts", "gray")})
        processed = {**RIGHT_FINGERTIP, "processed": {"method": "deformation-map", "version": "2.1"}}
        out = TactileInfo.from_dict(processed).to_dict()
        assert out == processed and list(out).index("processed") == list(out).index("derived") + 1
        with pytest.raises(SchemaError, match="processed is defined for visuotactile"):
            TactileInfo.from_dict({**PALM, "processed": {"method": "x", "version": "1"}})
        with pytest.raises(SchemaError, match="processed.version"):
            TactileInfo.from_dict({**RIGHT_FINGERTIP, "processed": {"method": "x", "version": ""}})


def pressure_entry(layout: dict) -> dict:
    """A taxel_array with one channel in kPa, a pressure map, and the frame plus ``layout``."""
    return {**PALM, "family": "taxel_array", "channels": channels("kPa", "p"), "layout": {**FRAME, **layout}}


GRID_64 = [[1.75 * (i % 8), 1.75 * (i // 8), 0.0] for i in range(64)]


class TestIndexing:
    def test_schema_example_round_trips(self) -> None:
        entry = pressure_entry({"indexing": GEN3_2_INDEXING})
        info = TactileInfo.from_dict(entry)
        assert info.to_dict() == entry
        assert info.layout.n_elements == 256 and info.layout.has_area
        assert [r.n_elements for r in info.layout.indexing.parts] == [64, 192]

    def test_one_grid(self) -> None:
        grid = {"kind": "grid", "shape": [8, 8], "pitch_mm": [1.75, 1.75], "area_mm2": 3.0625}
        info = TactileInfo.from_dict(pressure_entry({"indexing": grid}))
        assert info.layout.indexing.parts == [Region("grid", [8, 8], [1.75, 1.75], 0, 3.0625)]
        assert info.layout.to_dict()["indexing"] == grid
        with pytest.raises(SchemaError, match="a grid has shape and pitch_mm, not regions"):
            Indexing.from_dict({**grid, "regions": GEN3_2_INDEXING["regions"]})
        with pytest.raises(SchemaError, match="pitch_mm must be two positive numbers"):
            Indexing.from_dict({"kind": "grid", "shape": [8, 8], "pitch_mm": [1.75]})
        with pytest.raises(SchemaError, match="shape must be two positive integers"):
            Indexing.from_dict({"kind": "grid", "shape": [8, 0], "pitch_mm": [1.0, 1.0]})
        with pytest.raises(SchemaError, match="indexing.kind"):
            Indexing.from_dict({"kind": "hexagonal"})

    def test_regions_cover_the_elements_once(self) -> None:
        tip, palm = GEN3_2_INDEXING["regions"]
        assert Indexing.from_dict({"kind": "regions", "regions": [palm, tip]}).n_elements == 256
        with pytest.raises(SchemaError, match="region 'palm' starts at element 65, expected 64"):
            Indexing.from_dict({"kind": "regions", "regions": [tip, {**palm, "first_element": 65}]})
        with pytest.raises(SchemaError, match="region 'fingertip' starts at element 1, expected 0"):
            Indexing.from_dict({"kind": "regions", "regions": [{**tip, "first_element": 1}, palm]})
        with pytest.raises(SchemaError, match="unique"):
            Indexing.from_dict({"kind": "regions", "regions": [tip, {**palm, "name": "fingertip"}]})
        with pytest.raises(SchemaError, match="at least one region"):
            Indexing.from_dict({"kind": "regions", "regions": []})
        with pytest.raises(SchemaError, match="each region carries its own"):
            Indexing.from_dict({**GEN3_2_INDEXING, "shape": [8, 8]})

    def test_a_pressure_channel_needs_element_area(self) -> None:
        bare = [{k: v for k, v in r.items() if k != "area_mm2"} for r in GEN3_2_INDEXING["regions"]]
        no_area = {"kind": "regions", "regions": bare}
        with pytest.raises(SchemaError, match=r"channels \['p'\] are pressures"):
            TactileInfo.from_dict(pressure_entry({"indexing": no_area}))
        newtons = {**pressure_entry({"indexing": no_area}), "channels": channels("N", "fz")}
        assert not TactileInfo.from_dict(newtons).layout.has_area
        for area in (3.0625, [3.0625] * 64):
            entry = pressure_entry({"positions_mm": GRID_64, "normals": None, "element_area_mm2": area})
            assert TactileInfo.from_dict(entry).to_dict() == entry
        for unit in ("Pa", "kPa", "N/mm2", "N/mm²", "psi"):
            with pytest.raises(SchemaError, match="are pressures"):
                TactileInfo.from_dict(
                    {**pressure_entry({"positions_mm": GRID_64}), "channels": channels(unit, "p")}
                )
        with pytest.raises(SchemaError, match="2 values for 64 positions"):
            TactileInfo.from_dict(pressure_entry({"positions_mm": GRID_64, "element_area_mm2": [1.0, 1.0]}))

    def test_where_indexing_and_area_belong(self) -> None:
        both = pressure_entry({"positions_mm": positions(256), "normals": None, "indexing": GEN3_2_INDEXING})
        assert TactileInfo.from_dict(both).to_dict() == both
        with pytest.raises(SchemaError, match="positions_mm has 255 rows, layout.indexing 256 elements"):
            TactileInfo.from_dict(
                pressure_entry({"positions_mm": positions(255), "indexing": GEN3_2_INDEXING})
            )
        with pytest.raises(SchemaError, match="element_area_mm2 is for positions_mm alone"):
            TactileInfo.from_dict(pressure_entry({"indexing": GEN3_2_INDEXING, "element_area_mm2": 3.0}))
        with pytest.raises(SchemaError, match="element_area_mm2 is given without positions_mm"):
            Layout.from_dict({**FRAME, "element_area_mm2": 3.0})
        grid = {"kind": "grid", "shape": [1, 1], "pitch_mm": [1.0, 1.0]}
        with pytest.raises(SchemaError, match="defined for taxel_array and magnetic"):
            TactileInfo.from_dict({**WRENCH, "layout": {**FRAME, "indexing": grid}})


class TestUncalibratedUnits:
    def test_counts_and_au_require_an_uncalibrated_session(self) -> None:
        record = SessionRecord.from_dict(LEFT_FINGERTIP_META["sessions"][0])
        counts = TactileInfo.from_dict({**PALM, "channels": channels("counts", "bx", "by", "bz")})
        match = r"channels in \['counts'\] require calibration.method 'uncalibrated', got 'vendor'"
        with pytest.raises(SchemaError, match=match):
            check_session_calibration(counts, record)
        record.calibration = Calibration(method="uncalibrated")
        check_session_calibration(counts, record)
        record.calibration = None
        au = TactileInfo.from_dict({**PALM, "channels": channels("a.u.", "bx", "by", "bz")})
        with pytest.raises(SchemaError, match=r"\['a.u.'\] require .* got None"):
            check_session_calibration(au, record)
        check_session_calibration(TactileInfo.from_dict(PALM), record)  # uT needs no method


class TestDeviceAndTransport:
    def test_device_index(self) -> None:
        assert TactileInfo.from_dict(LEFT_FINGERTIP).device.index == 0
        device = LEFT_FINGERTIP["device"]
        for bad in (-1, None):
            with pytest.raises(SchemaError, match="device.index must be a non-negative integer"):
                TactileInfo.from_dict({**LEFT_FINGERTIP, "device": {**device, "index": bad}})
        for bad in (1.5, True):
            with pytest.raises(SchemaError, match="expected an integer"):
                TactileInfo.from_dict({**LEFT_FINGERTIP, "device": {**device, "index": bad}})

    def test_indices_are_unique_on_a_device(self) -> None:
        left = TactileInfo.from_dict(LEFT_FINGERTIP)
        right = {**LEFT_FINGERTIP, "session_meta": "tactile/right/meta.json"}
        with pytest.raises(
            SchemaError, match="'left' and 'right' share device.index 0 on device 'paxini_hub_0'"
        ):
            check_device_indices({"left": left, "right": TactileInfo.from_dict(right)})
        second = TactileInfo.from_dict({**right, "device": {**LEFT_FINGERTIP["device"], "index": 1}})
        check_device_indices({"left": left, "right": second, "palm": TactileInfo.from_dict(PALM)})

    def test_transport(self) -> None:
        assert SessionMeta.from_dict(LEFT_FINGERTIP_META).sessions[0].transport == Transport("usb", 91)
        assert Transport("ble", None).to_dict() == {"kind": "ble", "rate_hz": None}
        with pytest.raises(SchemaError, match="transport.kind"):
            Transport("wifi", 100.0)
        with pytest.raises(SchemaError, match="transport.rate_hz"):
            Transport("usb", 0)
        with pytest.raises(SchemaError, match="missing required key 'rate_hz'"):
            Transport.from_dict({"kind": "usb"})
        older = copy.deepcopy(LEFT_FINGERTIP_META)
        del older["sessions"][0]["transport"]
        assert SessionMeta.from_dict(older).sessions[0].transport is None  # optional: a 0.3 record reads


class TestClockColumn:
    FEATURES = {
        "source_timestamp": {"dtype": "float64", "shape": [1]},
        "t_epoch": {"dtype": "float32", "shape": []},
        "observation.state": {"dtype": "float32", "shape": [6]},
    }

    def test_names_a_feature_of_one_number_per_frame(self) -> None:
        info = TactileInfo.from_dict({**PALM, "clock_column": "source_timestamp"})
        keys = list(info.to_dict())
        assert keys.index("clock_column") == keys.index("mount") - 1
        info.check_clock_column(self.FEATURES)
        TactileInfo.from_dict({**PALM, "clock_column": "t_epoch"}).check_clock_column(self.FEATURES)
        TactileInfo.from_dict(PALM).check_clock_column({})  # none declared
        with pytest.raises(SchemaError, match="'timestamp_ns' is not a feature"):
            TactileInfo.from_dict({**PALM, "clock_column": "timestamp_ns"}).check_clock_column(self.FEATURES)
        with pytest.raises(
            SchemaError, match=r"'observation.state' is float32 \[6\], not one number per frame"
        ):
            TactileInfo.from_dict({**PALM, "clock_column": "observation.state"}).check_clock_column(
                self.FEATURES
            )
        with pytest.raises(SchemaError, match="clock_column must be a non-empty string"):
            TactileInfo.from_dict({**PALM, "clock_column": ""})


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


# The report example from SCHEMA.md, "Health annotations", with its "..." rows filled in: the
# instance has 52 elements, and nan_pct has one value per element.
NO_NAN = [0.0] * 52
HEALTH = {
    "doctor_version": "0.1.0",
    "generated_at": "2026-11-05T14:03:05Z",
    "thresholds": {
        "drift": 0.05,
        "drift_units": None,
        "dropped": 0.02,
        "max_gap_ms": None,
        "gel": 0.05,
        "cluster_px": None,
        "cluster_pct": 0.005,
        "skew": 50.0,
        "confidence": 0.6,
        "saturated": 0.01,
        "held": 0.05,
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
            "channels": channels("N", "fx", "fy", "fz"),
            "upgraded": False,
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
                    "nan_pct": NO_NAN,
                    "flag": False,
                },
                "framerate": {
                    "effective_hz": 89.4,
                    "configured_hz": 91,
                    "declared_hz": 91,
                    "dropped_pct": 0.018,
                    "max_gap_ms": 31.0,
                    "max_gap_threshold_ms": 109.9,
                    "gap_histogram": {"edges_periods": [1.5, 2.5, 3.5], "counts": [600, 9, 2, 0]},
                    "frames": {"received": 30412, "dropped": 7, "overflowed": 0},
                    "source": "sidecar",
                    "timestamps": "measured",
                    "stalled": 0,
                    "clock": None,
                    "flag": False,
                },
                "saturation": {
                    "saturated_pct": [0.0, 0.0, 0.0],
                    "at_min_pct": [0.0, 0.0, 0.0],
                    "at_max_pct": [0.0, 0.0, 0.0],
                    "limit": "observed",
                    "min": [-0.4, -0.5, -0.1],
                    "max": [3.1, 2.9, 24.8],
                    "confidence": "low",
                    "source": "sidecar",
                    "nan_pct": NO_NAN,
                    "flag": False,
                },
                "gel": {
                    "kind": "elements",
                    "dead_pct": 0.0,
                    "largest_cluster_px": None,
                    "largest_cluster_pct": None,
                    "dead_elements": [],
                    "constant_elements": [],
                    "frames_used": 612,
                    "median_variance": None,
                    "rest_variance": None,
                    "episodes_checked": 14,
                    "nan_pct": NO_NAN,
                    "flag": False,
                },
                "skew": {
                    "lag_ms": -12.0,
                    "confidence": 0.81,
                    "reference": "observation.images.front",
                    "onsets": 3,
                    "resolution_ms": 5.0,
                    "flag": False,
                },
                "held": {
                    "held_pct": 0.0016,
                    "frames": 611,
                    "longest_run": 1,
                    "single_holds": 1,
                    "expected_pct": 0.0,
                    "groups": None,
                    "nan_pct": NO_NAN,
                    "source": "sidecar",
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
        assert isinstance(checks.gel, GelReport) and checks.gel.kind == "elements"
        assert checks.gel.dead_elements == [] and checks.gel.episodes_checked == 14
        assert checks.drift.frame_range == "not recorded" and checks.drift.rebaselines[0].sample_index == 1200
        assert [f for f in checks.to_dict()] == list(CHECK_NAMES)
        assert list(report.to_dict()["instances"]["left_fingertip"]) == list(
            HEALTH["instances"]["left_fingertip"]
        )
        assert list(checks.to_dict()["framerate"])[:3] == ["effective_hz", "configured_hz", "declared_hz"]

    def test_an_upgraded_instance_reads_back(self) -> None:
        data = copy.deepcopy(HEALTH)
        instance = data["instances"]["left_fingertip"]
        instance["channels"] = [{"name": "fx", "unit": None, "source": "unknown"}]
        instance["upgraded"] = True
        report = HealthReport.from_dict(data)
        assert report.instances["left_fingertip"].channels == [ChannelReport("fx", None, "unknown")]
        assert report.to_dict() == data
        instance["channels"][0]["source"] = "guessed"
        with pytest.raises(SchemaError, match="source must be one of"):
            HealthReport.from_dict(data)

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

    def test_a_report_from_before_the_added_keys_still_reads(self) -> None:
        """The keys added after 0.3.1 (saturation at_min_pct and at_max_pct, the gel block's cross-episode
        fields, nan_pct, the framerate's stalled, clock and configured_hz, the held block and its
        threshold, thresholds.cluster_pct, the instance's channels and upgraded) have defaults, so an
        older health.json still loads."""
        data = copy.deepcopy(HEALTH)
        data["thresholds"] = {k: v for k, v in data["thresholds"].items() if k not in ("cluster_pct", "held")}
        data["thresholds"]["cluster_px"] = 25
        del data["instances"]["left_fingertip"]["channels"], data["instances"]["left_fingertip"]["upgraded"]
        block = data["episodes"]["episode_000012"]["left_fingertip"]
        del (
            block["held"],
            block["drift"]["nan_pct"],
            block["framerate"]["stalled"],
            block["framerate"]["clock"],
            block["framerate"]["configured_hz"],
        )
        block["saturation"] = {
            k: v for k, v in block["saturation"].items() if not k.startswith("at_") and k != "nan_pct"
        }
        block["gel"] = {
            "kind": "elements",
            "dead_pct": 0.0,
            "largest_cluster_px": None,
            "dead_elements": [],
            "frames_used": 612,
            "flag": False,
        }
        report = HealthReport.from_dict(data)
        assert report.thresholds.cluster_px == 25 and report.thresholds.cluster_pct == 0.005
        assert report.thresholds.held == 0.05
        checks = report.episodes["episode_000012"]["left_fingertip"]
        assert checks.saturation.at_min_pct is None and checks.saturation.at_max_pct is None
        assert checks.gel.constant_elements is None and checks.gel.rest_variance is None
        assert checks.gel.episodes_checked == 1 and checks.held is None
        assert (
            checks.drift.nan_pct is None
            and checks.framerate.stalled is None
            and checks.framerate.clock is None
            and checks.framerate.configured_hz is None
        )
        instance = report.instances["left_fingertip"]
        assert instance.channels is None and not instance.upgraded
        out = report.to_dict()["episodes"]["episode_000012"]["left_fingertip"]
        assert "held" not in out and list(out) == list(CHECK_NAMES[:-1])
        assert list(out["saturation"])[:3] == ["saturated_pct", "at_min_pct", "at_max_pct"]
        assert list(out["drift"])[-2:] == ["nan_pct", "flag"]  # a check block's verdict stays last
        assert list(out["framerate"])[-3:] == ["stalled", "clock", "flag"]
        assert list(out["gel"]) == [
            "kind",
            "dead_pct",
            "largest_cluster_px",
            "largest_cluster_pct",
            "dead_elements",
            "constant_elements",
            "frames_used",
            "median_variance",
            "rest_variance",
            "episodes_checked",
            "nan_pct",
            "flag",
        ]

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
