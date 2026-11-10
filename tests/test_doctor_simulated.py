"""End to end: record with ``taxel record`` from simulated sensors with injected faults, run the doctor,
and check what it finds; then the same on a clean recording, a tampered copy, and a plain dataset."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.datasets import LeRobotDataset  # noqa: E402
from test_loader import plain_dataset  # noqa: E402

from taxel.cli.doctor import format_report  # noqa: E402
from taxel.cli.main import main  # noqa: E402
from taxel.dataset.features import feature_entry  # noqa: E402
from taxel.dataset.sidecar import episode_path, read_episode, read_session_meta, write_episode  # noqa: E402
from taxel.dataset.writer import TactileDatasetWriter, tactile_features  # noqa: E402
from taxel.doctor.checks import ENCODER_FLOOR, saturation_check  # noqa: E402
from taxel.doctor.loader import DoctorDataset  # noqa: E402
from taxel.doctor.report import UNAVAILABLE, run_doctor, write_report  # noqa: E402
from taxel.schema import (  # noqa: E402
    Axes,
    Channel,
    DriftReport,
    HealthReport,
    Indexing,
    IntegrityReport,
    Layout,
    Origin,
    Processed,
    Range,
    SchemaError,
    Skipped,
    TactileInfo,
    Thresholds,
    Transport,
    feature_key,
    session_meta_path,
)
from taxel.sensors import (  # noqa: E402
    SimulatedTactile,
    SimulatedTactileConfig,
    SimulatedVisuotactile,
    SimulatedVisuotactileConfig,
    TactileReading,
    host_clock,
)

INSTANCE = "left_fingertip"
BASE = {
    "type": "simulated",
    "declared_rate_hz": 500,
    "n_elements": 16,
    "seed": 1,
    "contact_schedule": [[0.6, 0.6, 8.0]],  # a held press early in episode 0, longer than a drift window
}
FAULTS = {
    **BASE,
    "drop_probability": 0.05,
    "drift_per_s": [0.0, 0.0, 0.8],
    "saturation_max": 6.0,  # the press clamps, the ramp does not
    "rebaseline_at_s": [3.6],  # inside the second episode, which starts between 2.5 s and 3.5 s
}
WINDOW_S = 0.25


def record(base: Path, sensor: dict) -> Path:
    args = [
        "record",
        "--robot.type=cameras_only",
        f"--robot.calibration_dir={base / 'calibration'}",
        "--tactile=" + json.dumps({INSTANCE: sensor}),
        "--dataset.repo_id=taxel/doctor_test",
        f"--dataset.root={base / 'ds'}",
        "--dataset.no_stamp=true",
        "--dataset.num_episodes=2",
        "--dataset.episode_time_s=2",
        "--dataset.reset_time_s=0.3",
        "--dataset.fps=30",
        "--dataset.single_task=press",
        "--baseline_s=0.2",
        "--apply_baseline=true",
        "--baseline_sample_every=50",
        "--session_id=run",
        "--play_sounds=false",
    ]
    assert main(args) == 0
    return base / "ds"


@pytest.fixture(scope="module")
def faulty(tmp_path_factory) -> Path:
    return record(tmp_path_factory.mktemp("faulty"), FAULTS)


LAPSES = {
    **BASE,
    "hold_schedule": [[1.0, 0.2], [1.6, 0.002]],  # in episode 0: 100 frames resent, then one
    "nan_schedule": [[3.0, 0.2]],  # in episode 1: elements 0 and 1 lost for 0.2 s
    "nan_elements": [0, 1],
}


@pytest.fixture(scope="module")
def lapses(tmp_path_factory) -> Path:
    return record(tmp_path_factory.mktemp("lapses"), LAPSES)


@pytest.fixture(scope="module")
def clean(tmp_path_factory) -> Path:
    return record(tmp_path_factory.mktemp("clean"), BASE)


def checks(root: Path, **kwargs):
    report = run_doctor(DoctorDataset(root), window_s=WINDOW_S, **kwargs)
    return report, [report.episodes[f"episode_{e:06d}"][INSTANCE] for e in (0, 1)]


def steady(root: Path, dest: Path) -> Path:
    """A copy of a recording whose native timestamps are on the ideal grid: each frame ``seq - seq_0``
    periods after the episode's first, the device clock keeping its offset.

    A live recording's gaps include every stall of the host running the test, and ten periods at
    500 Hz is 20 ms; on the grid a gap is a drop and nothing else, so the framerate check reads the
    same on every run. Integrity reduces the stored stream by these timestamps and no longer
    reproduces the main table from them, so it is left out of what is asserted on the copy.
    """
    shutil.copytree(root, dest)
    for e in (0, 1):
        path = episode_path(dest, INSTANCE, e)
        stream = read_episode(path)
        host = stream.host_ts[0] + (stream.seq - stream.seq[0]) / BASE["declared_rate_hz"]
        device = host + (stream.device_ts[0] - stream.host_ts[0])
        readings = [
            TactileReading(stream.data[i], float(host[i]), float(device[i]), int(stream.seq[i]))
            for i in range(len(stream))
        ]
        write_episode(path, e, readings, stream.shape)
    return dest


class TestFaultyRecording:
    def test_report_round_trips_and_is_written(self, faulty) -> None:
        report, _ = checks(faulty)
        path = write_report(report, faulty)
        assert path == faulty / "tactile" / "health.json"
        assert HealthReport.from_dict(json.loads(path.read_text())).to_dict() == report.to_dict()
        instance = report.instances[INSTANCE]
        assert instance.metadata == "info" and instance.sidecar and instance.range.max == 6.0
        assert report.thresholds == Thresholds() and list(report.episodes) == [
            "episode_000000",
            "episode_000001",
        ]

    def test_integrity_passes_on_all_three_parts(self, faulty) -> None:
        _, episodes = checks(faulty)
        for ep in episodes:
            assert isinstance(ep.integrity, IntegrityReport) and not ep.integrity.flag
            assert ep.integrity.copy.mismatches == 0 and ep.integrity.copy.frames >= 40
            assert ep.integrity.raw.mismatches == 0 and ep.integrity.raw.samples >= 10
            files = ep.integrity.files
            assert (
                files.native and files.baseline and files.capture_at_zero and files.origins and files.counters
            )

    def test_framerate_flags_the_source_drops(self, faulty, tmp_path) -> None:
        _, episodes = checks(faulty)
        for ep in episodes:
            fr = ep.framerate
            assert fr.source == "sidecar" and fr.timestamps == "measured" and fr.declared_hz == 500.0
            assert 0.02 < fr.dropped_pct < 0.1 and fr.flag  # drops come from seq, whatever the host clock did
            assert fr.frames is not None and fr.frames.dropped > 0 and fr.max_gap_threshold_ms == 20.0
        # On fixed timestamps the drops alone raise the flag, and the rate is what they leave of 500 Hz.
        _, episodes = checks(steady(faulty, tmp_path / "steady"), checks=["framerate"])
        for ep in episodes:
            fr = ep.framerate
            assert fr.flag and 0.02 < fr.dropped_pct < 0.1 and fr.max_gap_ms < fr.max_gap_threshold_ms
            assert fr.effective_hz == pytest.approx(500 * (1 - fr.dropped_pct), abs=0.1)

    def test_saturation_flags_the_clamped_press(self, faulty) -> None:
        _, (first, second) = checks(faulty)
        assert first.saturation.limit == "declared" and first.saturation.max == [6.0, 6.0, 6.0]
        assert all(p > 0.01 for p in first.saturation.saturated_pct) and first.saturation.flag
        assert second.saturation.saturated_pct == [0.0, 0.0, 0.0] and not second.saturation.flag

    def test_drift_finds_the_ramp_and_not_the_held_press(self, faulty) -> None:
        _, (first, _) = checks(faulty)
        drift = first.drift
        assert isinstance(drift, DriftReport)
        assert drift.baseline == "capture" and drift.source == "samples"
        assert drift.slope_per_s[2] == pytest.approx(0.8, abs=0.2) and abs(drift.slope_per_s[0]) < 0.1
        assert (
            drift.offset[2] > 0.5 and drift.offset_pct_range[2] > 0.05 and drift.flag
        )  # about 1.5 of a span of 12
        assert 2 <= drift.no_contact_windows < drift.windows  # the held press spans a window
        assert drift.rebaselines == [] and not drift.first_frame_zero
        assert isinstance(drift.frame_range, list) and len(drift.frame_range) == 2
        assert drift.baseline_age_s is not None and drift.baseline_age_s > 0

    def test_rebaseline_is_an_event_and_drift_after_it_still_finds_the_ramp(self, faulty) -> None:
        _, episodes = checks(faulty)
        events = [(ep.drift, e) for ep in episodes for e in ep.drift.rebaselines]
        assert len(events) == 1, [ep.drift.to_dict() for ep in episodes]
        drift, event = events[0]
        assert event.delta[2] > 2.0 and event.delta[0] < 0.05 and event.delta[1] < 0.05  # 0.8 N/s for ~3.5 s
        assert event.sample_index > 0
        assert drift.slope_per_s[2] == pytest.approx(0.8, abs=0.3)
        assert 0.0 < drift.offset[2] < 2.0  # measured against the new capture, not the session's
        assert drift.flag == (max(drift.offset_pct_range) > 0.05)


class TestCleanRecording:
    def test_nothing_is_flagged(self, clean, tmp_path) -> None:
        _, episodes = checks(clean)
        for ep in episodes:
            assert not ep.integrity.flag and not ep.saturation.flag
            assert ep.framerate.dropped_pct == 0.0
            assert ep.saturation.limit == "observed" and ep.saturation.confidence == "low"
            assert isinstance(ep.drift, DriftReport) and not ep.drift.flag
            assert all(abs(s) < 0.05 for s in ep.drift.slope_per_s)
            assert all(abs(o) < 0.05 for o in ep.drift.offset) and ep.drift.rebaselines == []
            assert ep.gel.kind == "elements" and isinstance(ep.skew, Skipped)
        first = episodes[0].drift
        assert first.no_contact_windows < first.windows  # the press is there, and is not drift
        # The gaps on fixed timestamps: on the live clock a stall of the host is a gap over 20 ms.
        _, episodes = checks(steady(clean, tmp_path / "steady"), checks=["framerate"])
        for ep in episodes:
            assert not ep.framerate.flag and ep.framerate.max_gap_ms == pytest.approx(2.0)


class TestLapses:
    """A recording whose sensor resent frames in episode 0 and lost readings in episode 1."""

    def test_held_frames_are_counted_and_flagged_on_the_native_stream(self, lapses) -> None:
        _, (first, second) = checks(lapses)
        held = first.held
        assert held.source == "sidecar" and held.expected_pct == 0.0 and held.groups is None
        assert held.longest_run == 100 and held.single_holds == 1
        assert held.held_pct == pytest.approx(101 / held.frames) and held.flag  # about 10%, over 5%
        assert first.framerate.stalled == 0  # the host clock kept advancing through the holds
        assert second.held.held_pct == 0.0 and not second.held.flag
        assert not first.integrity.flag  # a resent frame is stored as it came, and reproduces

    def test_nan_is_reported_and_left_out(self, lapses) -> None:
        _, (first, second) = checks(lapses)
        assert first.saturation.nan_pct == [0.0] * 16
        for block in (second.saturation, second.drift, second.held, second.gel):
            assert block.nan_pct[:2] == [pytest.approx(0.1, abs=0.01)] * 2 and block.nan_pct[2:] == [0.0] * 14
        assert second.held.frames < len(read_episode(episode_path(lapses, INSTANCE, 1))) - 1
        assert isinstance(second.drift, DriftReport) and abs(second.drift.slope_per_s[0]) < 0.05
        assert not second.integrity.flag  # NaN reproduced as NaN in the copy and the samples
        assert second.integrity.copy.mismatches == 0 and second.integrity.raw.mismatches == 0


class TestTamperedRecording:
    def test_raw_names_the_first_mismatching_sample(self, faulty, tmp_path) -> None:
        root = tmp_path / "tampered"
        shutil.copytree(faulty, root)
        path = episode_path(root, INSTANCE, 0)
        stream = read_episode(path)
        data = stream.data.copy()
        data[50, 0, 0] += 1.0  # a stored frame edited after the fact, at a sampled index
        readings = [
            TactileReading(data[i], float(stream.host_ts[i]), float(stream.device_ts[i]), int(stream.seq[i]))
            for i in range(len(stream))
        ]
        write_episode(path, 0, readings, stream.shape)
        _, (first, second) = checks(root)
        assert first.integrity.flag and first.integrity.raw.flag
        assert first.integrity.raw.mismatches == 1 and first.integrity.raw.first_mismatch == 50
        assert first.integrity.raw.max_abs_diff == pytest.approx(1.0, abs=1e-6)
        assert first.integrity.copy.flag  # the main-table copy no longer reproduces either
        assert not second.integrity.flag
        assert first.drift.rebaselines == []  # never counted as drift


class TestPlainDataset:
    def test_checks_degrade_and_say_why(self, tmp_path) -> None:
        root = tmp_path / "plain"
        plain_dataset(root, use_videos=False)
        report = run_doctor(DoctorDataset(root), window_s=0.1)
        probe = report.episodes["episode_000000"]["probe"]
        assert isinstance(probe.integrity, Skipped) and probe.integrity.skipped.startswith("no sidecar")
        assert probe.framerate.source == "main_table" and probe.framerate.timestamps == "nominal"
        assert not probe.framerate.flag and probe.framerate.dropped_pct is None
        assert probe.saturation.limit == "observed" and probe.saturation.source == "main_table"
        assert isinstance(probe.drift, DriftReport) and probe.drift.baseline == "none"
        assert probe.drift.source == "main_table" and isinstance(probe.drift.offset, Skipped)
        assert probe.drift.offset.skipped == "no session baseline" and not probe.drift.first_frame_zero
        assert probe.drift.confidence == "low" and not probe.drift.flag
        gel = report.episodes["episode_000000"]["gel"]
        assert isinstance(gel.integrity, Skipped) and isinstance(gel.drift, Skipped)
        assert gel.framerate.source == "main_table"
        assert report.instances["probe"].metadata == "none" and not report.instances["probe"].sidecar

    def test_first_frame_zero_is_named(self, tmp_path) -> None:
        from lerobot.datasets import LeRobotDataset

        root = tmp_path / "zero"
        features = {"observation.tactile.probe": {"dtype": "float32", "shape": [4, 3], "names": None}}
        dataset = LeRobotDataset.create("taxel/zero", fps=30, features=features, root=root, use_videos=False)
        rng = np.random.default_rng(0)
        for k in range(12):
            frame = np.zeros((4, 3), np.float32) if k == 0 else rng.normal(size=(4, 3)).astype(np.float32)
            dataset.add_frame({"observation.tactile.probe": frame, "task": "t"})
        dataset.save_episode()
        dataset.finalize()
        report = run_doctor(DoctorDataset(root), window_s=0.1)
        assert report.episodes["episode_000000"]["probe"].drift.first_frame_zero

    def test_unknown_episode_is_an_error(self, tmp_path) -> None:
        root = tmp_path / "plain"
        plain_dataset(root, use_videos=False)
        with pytest.raises(KeyError, match="no episodes \\[5\\]"):
            run_doctor(DoctorDataset(root), episodes=[5])
        with pytest.raises(KeyError, match="unknown checks \\['noise'\\]"):
            run_doctor(DoctorDataset(root), checks=["drift", "noise"])

    def test_no_instance_is_an_error(self, tmp_path) -> None:
        from lerobot.datasets import LeRobotDataset

        root = tmp_path / "none"
        features = {"observation.state": {"dtype": "float32", "shape": [3], "names": None}}
        dataset = LeRobotDataset.create("taxel/none", fps=30, features=features, root=root, use_videos=False)
        for _ in range(6):
            dataset.add_frame({"observation.state": np.zeros(3, np.float32), "task": "t"})
        dataset.save_episode()
        dataset.finalize()
        with pytest.raises(KeyError, match="no tactile instance.*--guess"):
            run_doctor(DoctorDataset(root))

    def test_skew_reference_is_the_scene_camera_unless_asked_for_all(self, tmp_path, monkeypatch) -> None:
        """With a camera in the dataset every instance correlates against it, so one stream serves them
        all; ``all`` searches every video stream for the most motion, which the two-sensor case needs."""
        from taxel.doctor import report as report_module

        root = tmp_path / "plain"
        plain_dataset(root, use_videos=False)
        asked: list[str] = []
        real = report_module.skew_check

        def spy(tactile_times, tactile_events, reference_times, reference_events, reference_key, *rest):
            asked.append(reference_key)
            return real(
                tactile_times, tactile_events, reference_times, reference_events, reference_key, *rest
            )

        monkeypatch.setattr(report_module, "skew_check", spy)
        run_doctor(DoctorDataset(root), window_s=0.1, checks=["skew"])
        assert asked == ["observation.images.cam"] * 4  # two instances, two episodes
        asked.clear()
        run_doctor(DoctorDataset(root), window_s=0.1, checks=["skew"], reference="all")
        assert len(asked) == 4 and set(asked) <= {"observation.images.cam", "observation.tactile.gel"}
        assert asked[1] == "observation.images.cam"  # the gel instance's only other image feature

    def test_dead_elements_need_every_checked_episode(self, tmp_path) -> None:
        """A zone nothing touched in an episode is constant there; a dead zone is constant in every episode.
        Zones 2, 3, 4 and 15 of a 17-zone glove never move, and two live zones idle in each episode."""
        from lerobot.datasets import LeRobotDataset

        root = tmp_path / "glove"
        features = {
            "observation.tactile.glove": {"dtype": "float32", "shape": [17, 1], "names": None},
            "observation.tactile.contact": {"dtype": "float32", "shape": [17], "names": None},
        }
        dataset = LeRobotDataset.create("taxel/glove", fps=30, features=features, root=root, use_videos=False)
        rng = np.random.default_rng(0)
        dead = [2, 3, 4, 15]
        for e in range(3):
            idle = [5 + e, 9 + e]
            for _ in range(30):
                frame = rng.normal(size=(17, 1)).astype(np.float32)
                frame[dead] = 0.5
                frame[idle] = 0.0
                contact = (rng.random(17) > 0.7).astype(np.float32)
                contact[dead] = 0.0
                dataset.add_frame(
                    {"observation.tactile.glove": frame, "observation.tactile.contact": contact, "task": "t"}
                )
            dataset.save_episode()
        dataset.finalize()
        report = run_doctor(DoctorDataset(root), window_s=0.1, checks=["gel"])
        for e in range(3):
            gel = report.episodes[f"episode_{e:06d}"]["glove"].gel
            assert gel.constant_elements == sorted(dead + [5 + e, 9 + e])
            assert gel.dead_elements == dead and gel.episodes_checked == 3
            assert gel.dead_pct == pytest.approx(4 / 17) and gel.flag
            contact = report.episodes[f"episode_{e:06d}"]["contact"].gel
            assert isinstance(contact, Skipped) and contact.skipped.startswith("binary feature")
        # One checked episode cannot tell the idle zones from the dead ones: dead stays undecided.
        one = run_doctor(DoctorDataset(root), window_s=0.1, checks=["gel"], episodes=[1])
        gel = one.episodes["episode_000001"]["glove"].gel
        assert gel.constant_elements == sorted(dead + [6, 10]) and gel.dead_elements is None
        assert gel.episodes_checked == 1 and gel.dead_pct == 0.0 and not gel.flag

    def test_a_clock_column_runs_framerate_and_skew(self, tmp_path) -> None:
        """A dataset that keeps ``timestamp`` nominal and its real clock in a column of its own."""
        root = tmp_path / "clocked"
        features = {
            "observation.tactile.probe": {"dtype": "float32", "shape": [4, 3], "names": None},
            "source_timestamp": {"dtype": "float64", "shape": [1], "names": ["seconds"]},
        }
        dataset = LeRobotDataset.create("taxel/clock", fps=30, features=features, root=root, use_videos=False)
        rng = np.random.default_rng(0)
        steps = np.full(89, 1 / 30)
        steps[[10, 40, 41]] = [0.06, 0.0, 0.07]  # two long steps and a stalled one
        clock = 1.7e9 + np.concatenate([[0.0], np.cumsum(steps)])
        for t in clock:
            frame = rng.normal(size=(4, 3)).astype(np.float32)
            dataset.add_frame(
                {"observation.tactile.probe": frame, "source_timestamp": np.array([t]), "task": "t"}
            )
        dataset.save_episode()
        dataset.finalize()
        report = run_doctor(DoctorDataset(root), window_s=0.5, clock="source_timestamp")
        framerate = report.episodes["episode_000000"]["probe"].framerate
        assert framerate.clock == "source_timestamp" and framerate.timestamps == "measured"
        assert framerate.gap_histogram.counts[1:] == [2, 0, 0] and framerate.stalled == 1
        assert framerate.max_gap_ms == pytest.approx(70.0, abs=0.01)  # epoch seconds resolve to about 0.2 us
        nominal = run_doctor(DoctorDataset(root), window_s=0.5).episodes["episode_000000"]["probe"].framerate
        assert nominal.timestamps == "nominal" and nominal.clock is None
        with pytest.raises(KeyError, match="clock column 'missing'"):
            run_doctor(DoctorDataset(root), clock="missing")
        with pytest.raises(SchemaError, match="not one number per frame"):
            run_doctor(DoctorDataset(root), clock="observation.tactile.probe")

    def test_held_groups_come_from_the_feature_names(self, tmp_path) -> None:
        """Two fingertips of six channels, named as T-Rex names them: a blackout filled on one tip is a
        held group, not a held frame."""
        root = tmp_path / "tips"
        names = [f"left_{tip}_f6_{i}" for tip in ("thumb", "index") for i in range(6)]
        features = {"observation.tactile_force": {"dtype": "float32", "shape": [12], "names": names}}
        dataset = LeRobotDataset.create("taxel/tips", fps=30, features=features, root=root, use_videos=False)
        rng = np.random.default_rng(0)
        previous = None
        for k in range(60):
            frame = rng.normal(size=12).astype(np.float32)
            if k in (20, 40) and previous is not None:
                frame[:6] = previous[:6]  # the thumb repeats a frame, twice in the episode
            dataset.add_frame({"observation.tactile_force": frame, "task": "t"})
            previous = frame
        dataset.save_episode()
        dataset.finalize()
        report = run_doctor(DoctorDataset(root), instances=["observation.tactile_force"], checks=["held"])
        held = report.episodes["episode_000000"]["observation.tactile_force"].held
        assert held.held_pct == 0.0 and held.source == "main_table" and not held.flag  # no declared rate
        assert set(held.groups) == {"left_thumb_f6", "left_index_f6"}
        assert held.groups["left_thumb_f6"].single_holds == 2 and held.groups["left_index_f6"].held_pct == 0.0

    def test_a_missing_video_file_is_skipped_with_the_reason(self, tmp_path) -> None:
        from test_loader import plain_dataset as plain

        root = tmp_path / "partial"
        plain(root, use_videos=True)
        for video in (root / "videos" / "observation.images.cam").rglob("*.mp4"):
            video.unlink()  # a partial download: the camera's files are not there
        report = run_doctor(DoctorDataset(root), window_s=0.1, checks=["gel", "skew"])
        block = report.episodes["episode_000000"]
        assert block["probe"].skew.reference == "observation.tactile.gel"  # the camera is passed over
        assert block["gel"].gel.kind == "pixels" or isinstance(block["gel"].gel, Skipped)
        for video in (root / "videos" / "observation.tactile.gel").rglob("*.mp4"):
            video.unlink()
        report = run_doctor(DoctorDataset(root), window_s=0.1, checks=["gel", "skew"])
        block = report.episodes["episode_000000"]
        assert block["gel"].gel.skipped.startswith("observation.tactile.gel: episode 0's video file videos/")
        assert block["gel"].gel.skipped.endswith("not found")
        assert block["probe"].skew == Skipped("no reference: the video files of 2 camera(s) are not on disk")

    def test_checks_selects_what_runs(self, tmp_path) -> None:
        root = tmp_path / "plain"
        plain_dataset(root, use_videos=False)
        report = run_doctor(DoctorDataset(root), window_s=0.1, checks=["framerate", "gel"])
        probe = report.episodes["episode_000001"]["probe"]
        assert probe.framerate.source == "main_table" and probe.gel.kind == "elements"
        for name in ("integrity", "drift", "saturation", "skew"):
            assert getattr(probe, name) == Skipped("not selected")
        gel = report.episodes["episode_000001"]["gel"]
        assert gel.gel.kind == "pixels" and gel.skew == Skipped("not selected")


# --- Two sensors sharing a press schedule: gel and skew ------------------------------------------------

GEL_KEY = feature_key("gel")
ARRAY_KEY = feature_key(INSTANCE)
EPISODE_STARTS_S = (1.0, 3.5)
"""Sensor time of each episode's first frame, from the array's ``connect()``. The fixture waits for
it, so an episode holds the same images on every run; the first start leaves the setup and the
baseline capture time to finish, the second half a second to save the first episode."""
PRESSES = [(1.4, 0.4), (2.2, 0.4), (3.9, 0.4), (4.7, 0.4)]
"""Two presses in each 2 s episode, 0.4 s and 1.2 s after its start. One press is an onset and a
release quantized to two video-frame phases, and the skew confidence of a clean pair then sits near
the floor on an unlucky recording; two give four events."""
IMAGE_RATE = 60.0
DELAY_FRAMES = 2  # the image sensor shows the press two of its frames late: 33 ms at 60 Hz
DEAD_BLOCK = (10, 20, 6, 6)  # 36 pixels, 1.2% of the 48x64 image, over the 0.5% cluster default


def two_sensor_dataset(root: Path, *, faults: bool) -> Path:
    """An array sensor through the writer and an image sensor fed to ``add_frame`` by hand, two episodes.

    The image of frame ``k`` is the image sensor's frame at the frame's sensor time, taken by its
    ``seq``. A live read returns whichever frame comes next, so the frames written, and their noise,
    would change with the host's timing from run to run; the noise is what sets the variance of the
    pixels at the dead block's edge after encoding, and with it the cluster the gel check finds. The
    encoder mixes the live neighbours' noise into the block, and on about half of the frame sets with
    these presses it lifts one to four of its pixels over the dead-pixel floor; the image seed is one
    whose frames leave the whole block under it in both episodes on arm64 and on x86_64, whose builds
    of the encoder encode the same frames differently.
    """
    array = SimulatedTactile(
        SimulatedTactileConfig(
            instance_id=INSTANCE,
            declared_rate_hz=500,
            n_elements=16,
            seed=1,
            contact_schedule=[(s, d, 4.0) for s, d in PRESSES],
            dead_elements=[2, 5] if faults else None,
        )
    )
    image = SimulatedVisuotactile(
        SimulatedVisuotactileConfig(
            instance_id="gel",
            declared_rate_hz=IMAGE_RATE,
            seed=20,
            noise_std=6.0,  # camera noise well above the codec's flicker, as on a real gel image
            contact_schedule=[(s, d, 80.0) for s, d in PRESSES],
            dead_pixels=DEAD_BLOCK if faults else None,
            transport_delay_s=DELAY_FRAMES / IMAGE_RATE if faults else 0.0,
        )
    )
    zero = host_clock()  # sensor time 0 of both: the image runs the press schedule from here too
    array.connect()
    features = {
        **tactile_features([array], fps=30),
        GEL_KEY: feature_entry(image.metadata(), image.config.shape, use_videos=True),
    }
    dataset = LeRobotDataset.create(
        "taxel/two_sensors", fps=30, features=features, root=root, use_videos=True
    )
    writer = TactileDatasetWriter(
        dataset, [array], baseline_s=0.2, apply_baseline=True, baseline_sample_every=50
    )
    writer.start_session("two")
    for start_s in EPISODE_STARTS_S:
        while (left := zero + start_s - host_clock()) > 0:
            writer.idle()
            time.sleep(min(left, 0.01))
        writer.start_episode()
        for k in range(60):
            values = writer.frame()  # at the frame's nominal time
            frame = image._frame(round((start_s + k / 30) * IMAGE_RATE))
            dataset.add_frame({**values, GEL_KEY: frame, "task": "press"})
        writer.end_episode()
        dataset.save_episode()
    array.disconnect()
    writer.finalize()
    dataset.finalize()
    return root


@pytest.fixture(scope="module")
def two_faulty(tmp_path_factory) -> Path:
    return two_sensor_dataset(tmp_path_factory.mktemp("two_faulty") / "ds", faults=True)


@pytest.fixture(scope="module")
def two_clean(tmp_path_factory) -> Path:
    return two_sensor_dataset(tmp_path_factory.mktemp("two_clean") / "ds", faults=False)


PERIOD_MS = 1000 / 30


class TestTwoSensors:
    def test_instances_and_references(self, two_faulty) -> None:
        report = run_doctor(DoctorDataset(two_faulty), window_s=WINDOW_S)
        assert sorted(report.instances) == ["gel", INSTANCE]
        gel = report.instances["gel"]
        assert gel.family == "visuotactile" and gel.metadata == "info" and not gel.sidecar
        assert gel.unknown_keys == []  # LeRobot's has_audio and is_depth_map are not the tactile block's
        for e in (0, 1):
            block = report.episodes[f"episode_{e:06d}"]
            assert block[INSTANCE].skew.reference == GEL_KEY and block["gel"].skew.reference == ARRAY_KEY
            assert isinstance(block["gel"].integrity, Skipped) and isinstance(block["gel"].drift, Skipped)

    def test_gel_finds_the_dead_cluster_and_the_dead_elements(self, two_faulty) -> None:
        report = run_doctor(DoctorDataset(two_faulty), window_s=WINDOW_S)
        for e in (0, 1):
            block = report.episodes[f"episode_{e:06d}"]
            pixels = block["gel"].gel
            assert pixels.kind == "pixels" and pixels.frames_used == 60 and pixels.episodes_checked == 2
            # Another machine's encoder build can lift a pixel or two of the block's edge over the floor.
            assert abs(pixels.largest_cluster_px - 36) <= 2 and pixels.flag
            assert pixels.largest_cluster_pct == pytest.approx(36 / (48 * 64), abs=2 / (48 * 64))
            assert pixels.dead_pct == pytest.approx(36 / (48 * 64), abs=0.002)
            assert ENCODER_FLOOR <= pixels.rest_variance <= pixels.median_variance
            elements = block[INSTANCE].gel
            assert elements.kind == "elements" and elements.constant_elements == [2, 5]
            assert elements.dead_elements == [2, 5] and elements.episodes_checked == 2
            assert elements.dead_pct == 0.125 and elements.flag

    def test_skew_recovers_the_transport_delay_within_a_frame_period(self, two_faulty) -> None:
        report = run_doctor(DoctorDataset(two_faulty), window_s=WINDOW_S)
        delay_ms = DELAY_FRAMES / IMAGE_RATE * 1000
        for e in (0, 1):
            block = report.episodes[f"episode_{e:06d}"]
            skew = block[INSTANCE].skew
            assert skew.confidence > Thresholds().confidence and skew.onsets >= 1
            assert skew.lag_ms == pytest.approx(-delay_ms, abs=PERIOD_MS + 5)  # the tactile stream leads
            assert skew.resolution_ms == pytest.approx(PERIOD_MS, abs=1.0)
            assert skew.flag == (abs(skew.lag_ms) > Thresholds().skew)
            # The image instance has no sidecar, so its own check runs on the nominal timeline against the
            # main table's mean-reduced copy of the array stream: both sides quantized, two frames of slack.
            mirrored = block["gel"].skew
            assert mirrored.lag_ms == pytest.approx(delay_ms, abs=2 * PERIOD_MS + 5)

    def test_explicit_reference(self, two_faulty) -> None:
        report = run_doctor(DoctorDataset(two_faulty), reference=GEL_KEY, episodes=[0], window_s=WINDOW_S)
        block = report.episodes["episode_000000"]
        assert block[INSTANCE].skew.reference == GEL_KEY
        assert isinstance(block["gel"].skew, Skipped) and "itself" in block["gel"].skew.skipped
        with pytest.raises(KeyError, match="not a feature"):
            run_doctor(DoctorDataset(two_faulty), reference="observation.images.none")

    def test_clean_pair_raises_nothing(self, two_clean) -> None:
        report = run_doctor(DoctorDataset(two_clean), window_s=WINDOW_S)
        for e in (0, 1):
            block = report.episodes[f"episode_{e:06d}"]
            pixels = block["gel"].gel
            assert pixels.dead_pct < 0.005 and pixels.largest_cluster_px <= 4 and not pixels.flag
            assert pixels.episodes_checked == 2 and pixels.rest_variance >= ENCODER_FLOOR
            elements = block[INSTANCE].gel
            assert elements.constant_elements == [] and elements.dead_elements == [] and not elements.flag
            for name, slack in ((INSTANCE, PERIOD_MS + 5), ("gel", 2 * PERIOD_MS + 5)):
                skew = block[name].skew
                assert skew.confidence > Thresholds().confidence and abs(skew.lag_ms) <= slack
                assert not skew.flag
            array = block[INSTANCE]
            assert not array.integrity.flag and not array.drift.flag and not array.saturation.flag


class TestNoVideo:
    def test_gel_reports_elements_and_skew_is_skipped(self, faulty) -> None:
        _, episodes = checks(faulty)
        for ep in episodes:
            assert ep.gel.kind == "elements" and ep.gel.dead_elements == [] and not ep.gel.flag
            assert isinstance(ep.skew, Skipped)
            assert ep.skew.skipped == "no reference: the dataset has no camera and no other tactile instance"


# --- SCHEMA 0.4: channel sources, a clock column in info, processed images, 0.3 metadata --------------

GRID = Layout(
    Origin("corner", "cell 0"),
    Axes("along a row", "down a column", "out of the skin"),
    "skin",
    indexing=Indexing("grid", [2, 4], [5.0, 5.0]),
)


def converted(
    root: Path, info: TactileInfo, shape: tuple[int, ...], frames: list[list[np.ndarray]], **extra
) -> Path:
    """A dataset as a converter writes it: one instance with an ``info`` block and no sidecar, one list of
    frames per episode, and ``extra`` main-table features with their per-frame values."""
    key = feature_key("probe")
    features = {key: feature_entry(info, shape)}
    features.update({name: {"dtype": "float64", "shape": [1], "names": None} for name in extra})
    dataset = LeRobotDataset.create("taxel/converted", fps=30, features=features, root=root, use_videos=False)
    for e, episode in enumerate(frames):
        for k, frame in enumerate(episode):
            values = {name: np.array([columns[e][k]]) for name, columns in extra.items()}
            dataset.add_frame({key: frame, **values, "task": "t"})
        dataset.save_episode()
    dataset.finalize()
    return root


def glove_frames(rng: np.random.Generator, n_episodes: int = 2, n: int = 60) -> list[list[np.ndarray]]:
    """An eight-cell glove at rest near 1000 counts on fz; fx and fy are kept as zeros."""
    out = []
    for _ in range(n_episodes):
        episode = []
        for _ in range(n):
            frame = np.zeros((8, 3), np.float32)
            frame[:, 2] = 1000.0 + rng.normal(0.0, 5.0, 8)
            episode.append(frame)
        out.append(episode)
    return out


class TestSchema04:
    def test_an_unavailable_channel_is_never_flagged(self, tmp_path) -> None:
        """A glove converted with fx and fy kept beside its normal force: the two are unavailable, stored as
        0.0. Against the converter's range [0, 4095] those zeros sit at the lower limit, so a saturation check
        that read them would flag every frame."""
        info = TactileInfo(
            family="taxel_array",
            declared_rate_hz=30,
            session_meta=session_meta_path("probe"),
            channels=[
                Channel("fx", "counts", "unavailable"),
                Channel("fy", "counts", "unavailable"),
                Channel("fz", "counts", "measured"),
            ],
            range=Range(min=0.0, max=4095.0),
            layout=GRID,
        )
        frames = glove_frames(np.random.default_rng(0))
        report = run_doctor(DoctorDataset(converted(tmp_path / "glove", info, (8, 3), frames)), window_s=0.5)
        for e in (0, 1):
            checks = report.episodes[f"episode_{e:06d}"]["probe"]
            for block in (checks.drift, checks.saturation, checks.gel, checks.held):
                assert not isinstance(block, Skipped) and not block.flag
            assert checks.saturation.limit == "declared"
            assert checks.saturation.saturated_pct == [None, None, 0.0]
            assert checks.drift.level_change[:2] == [None, None] and checks.drift.level_change[2] is not None
        unmasked = saturation_check(np.stack(frames[0]), info.range.limits(3), "main_table", Thresholds())
        assert unmasked.flag and unmasked.saturated_pct[:2] == [1.0, 1.0]
        assert [c.source for c in report.instances["probe"].channels] == [
            "unavailable",
            "unavailable",
            "measured",
        ]
        assert "channels fx, fy (counts, unavailable, not checked); fz (counts, measured)" in format_report(
            report
        )

    def test_an_instance_with_no_measured_channel_is_not_checked(self, tmp_path) -> None:
        names = ("fx", "fy", "fz", "mx", "my", "mz")
        info = TactileInfo(
            family="force_torque",
            declared_rate_hz=30,
            session_meta=session_meta_path("probe"),
            channels=[Channel(n, "N" if n[0] == "f" else "N.m", "unavailable") for n in names],
            layout=Layout(Origin("mount", "the flange"), Axes("x", "y", "out of the tool face"), "vendor"),
        )
        zeros = [[np.zeros((1, 6), np.float32)] * 30]
        report = run_doctor(DoctorDataset(converted(tmp_path / "blank", info, (1, 6), zeros)), window_s=0.5)
        checks = report.episodes["episode_000000"]["probe"]
        for block in (checks.drift, checks.saturation, checks.gel, checks.held, checks.skew):
            assert block == Skipped(UNAVAILABLE)
        assert not isinstance(checks.framerate, Skipped)  # the timing is still there to check

    def test_info_clock_column_is_the_default_clock_and_clock_overrides_it(self, tmp_path) -> None:
        steps = np.full(89, 1 / 30)
        steps[[10, 40, 41]] = [0.06, 0.0, 0.07]  # two long steps and a stalled one
        source = 1.7e9 + np.concatenate([[0.0], np.cumsum(steps)])
        host = 50.0 + np.arange(90) / 30
        info = TactileInfo(
            family="taxel_array",
            declared_rate_hz=30,
            session_meta=session_meta_path("probe"),
            channels=[Channel(n, "N", "measured") for n in ("fx", "fy", "fz")],
            layout=GRID,
            clock_column="source_timestamp",
        )
        rng = np.random.default_rng(1)
        frames = [[rng.normal(size=(8, 3)).astype(np.float32) for _ in range(90)]]
        root = converted(
            tmp_path / "clocked", info, (8, 3), frames, source_timestamp=[source], host_seconds=[host]
        )

        def framerate(**kwargs):
            report = run_doctor(DoctorDataset(root), window_s=0.5, checks=["framerate"], **kwargs)
            return report.episodes["episode_000000"]["probe"].framerate

        default = framerate()
        assert default == framerate(clock="source_timestamp")
        assert (
            default.clock == "source_timestamp" and default.stalled == 1 and default.timestamps == "measured"
        )
        override = framerate(clock="host_seconds")
        assert override.clock == "host_seconds" and override.stalled == 0

    def test_a_clock_column_that_names_no_clock_is_reported(self, tmp_path) -> None:
        info = TactileInfo(
            family="taxel_array",
            declared_rate_hz=30,
            session_meta=session_meta_path("probe"),
            channels=[Channel("fz", "N", "measured")],
            layout=GRID,
            clock_column="t_epoch",
        )
        frames = [[np.random.default_rng(2).normal(size=(8, 1)).astype(np.float32) for _ in range(30)]]
        report = run_doctor(
            DoctorDataset(converted(tmp_path / "bad", info, (8, 1), frames)), checks=["framerate"]
        )
        assert "info.clock_column: clock column 't_epoch' is not a feature" in report.instances["probe"].error
        framerate = report.episodes["episode_000000"]["probe"].framerate
        assert framerate.clock is None and framerate.timestamps == "nominal"

    def test_a_processed_image_skips_gel_with_the_reason(self, tmp_path) -> None:
        info = TactileInfo(
            family="visuotactile",
            declared_rate_hz=30,
            session_meta=session_meta_path("probe"),
            layout=Layout(
                Origin("corner", "the top-left pixel"), Axes("right", "down", "out of the gel"), "gel"
            ),
            processed=Processed("difference-image", "1.0"),
        )
        root = tmp_path / "diff"
        key = feature_key("probe")
        features = {key: feature_entry(info, (24, 32, 3), use_videos=False)}
        dataset = LeRobotDataset.create("taxel/diff", fps=30, features=features, root=root, use_videos=False)
        rng = np.random.default_rng(3)
        for _ in range(10):
            dataset.add_frame({key: rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), "task": "t"})
        dataset.save_episode()
        dataset.finalize()
        gel = run_doctor(DoctorDataset(root), checks=["gel"]).episodes["episode_000000"]["probe"].gel
        assert gel == Skipped(
            "processed image (difference-image 1.0): dead pixels are defined on raw gel frames"
        )

    def test_a_0_3_1_dataset_reads_through_the_lenient_path(self, clean, tmp_path, capsys) -> None:
        root = tmp_path / "old"
        shutil.copytree(clean, root)
        path = root / "meta" / "info.json"
        info = json.loads(path.read_text())
        block = info["features"][feature_key(INSTANCE)]["info"]
        block["channels"], block["units"] = [c["name"] for c in block["channels"]], {"force": "N"}
        path.write_text(json.dumps(info))
        report = run_doctor(DoctorDataset(root), window_s=WINDOW_S)
        instance = report.instances[INSTANCE]
        assert instance.metadata == "info" and instance.upgraded and instance.error is None
        assert [(c.name, c.unit, c.source) for c in instance.channels] == [
            ("fx", "N", "unknown"),
            ("fy", "N", "unknown"),
            ("fz", "N", "unknown"),
        ]
        assert HealthReport.from_dict(report.to_dict()) == report  # the report still reads back strictly
        first = report.episodes["episode_000000"][INSTANCE]
        assert isinstance(first.integrity, IntegrityReport) and not first.integrity.flag
        assert isinstance(first.drift, DriftReport)  # an unknown channel is checked as data
        assert main(["doctor", str(root), "--no-write", "--window-s", str(WINDOW_S)]) == 0
        out = capsys.readouterr().out
        assert out.count("SCHEMA 0.3 form") == 1 and "fx, fy, fz (N, source unknown)" in out

    def test_the_configured_rate_is_reported_beside_the_effective_rate(self, clean) -> None:
        assert read_session_meta(clean, INSTANCE).sessions[0].transport == Transport("other", 500.0)
        report, blocks = checks(clean)
        for block in blocks:
            assert block.framerate.configured_hz == 500.0 and block.framerate.effective_hz > 0
        instance = report.instances[INSTANCE]
        assert not instance.upgraded and {(c.unit, c.source) for c in instance.channels} == {
            ("N", "measured")
        }
