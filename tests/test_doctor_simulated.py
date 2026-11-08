"""End to end: record with ``taxel record`` from simulated sensors with injected faults, run the doctor,
and check what it finds; then the same on a clean recording, a tampered copy, and a plain dataset."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.datasets import LeRobotDataset  # noqa: E402
from test_loader import plain_dataset  # noqa: E402

from taxel.cli.main import main  # noqa: E402
from taxel.dataset.features import feature_entry  # noqa: E402
from taxel.dataset.sidecar import episode_path, read_episode, write_episode  # noqa: E402
from taxel.dataset.writer import TactileDatasetWriter, tactile_features  # noqa: E402
from taxel.doctor.checks import ENCODER_FLOOR  # noqa: E402
from taxel.doctor.loader import DoctorDataset  # noqa: E402
from taxel.doctor.report import run_doctor, write_report  # noqa: E402
from taxel.schema import DriftReport, HealthReport, IntegrityReport, Skipped, Thresholds, feature_key  # noqa: E402
from taxel.sensors import (  # noqa: E402
    SimulatedTactile,
    SimulatedTactileConfig,
    SimulatedVisuotactile,
    SimulatedVisuotactileConfig,
    TactileReading,
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


@pytest.fixture(scope="module")
def clean(tmp_path_factory) -> Path:
    return record(tmp_path_factory.mktemp("clean"), BASE)


def checks(root: Path, **kwargs):
    report = run_doctor(DoctorDataset(root), window_s=WINDOW_S, **kwargs)
    return report, [report.episodes[f"episode_{e:06d}"][INSTANCE] for e in (0, 1)]


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

    def test_framerate_flags_the_source_drops(self, faulty) -> None:
        _, episodes = checks(faulty)
        for ep in episodes:
            fr = ep.framerate
            assert fr.source == "sidecar" and fr.timestamps == "measured" and fr.declared_hz == 500.0
            assert 0.02 < fr.dropped_pct < 0.1 and fr.flag
            assert fr.frames is not None and fr.frames.dropped > 0 and fr.max_gap_threshold_ms == 20.0
            assert 400 < fr.effective_hz < 500

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
    def test_nothing_is_flagged(self, clean) -> None:
        _, episodes = checks(clean)
        for ep in episodes:
            assert not ep.integrity.flag and not ep.framerate.flag and not ep.saturation.flag
            assert ep.framerate.dropped_pct == 0.0
            assert ep.saturation.limit == "observed" and ep.saturation.confidence == "low"
            assert isinstance(ep.drift, DriftReport) and not ep.drift.flag
            assert all(abs(s) < 0.05 for s in ep.drift.slope_per_s)
            assert all(abs(o) < 0.05 for o in ep.drift.offset) and ep.drift.rebaselines == []
            assert ep.gel.kind == "elements" and isinstance(ep.skew, Skipped)
        first = episodes[0].drift
        assert first.no_contact_windows < first.windows  # the press is there, and is not drift


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
        # One checked episode cannot tell the idle zones from the dead ones, and does not flag.
        one = run_doctor(DoctorDataset(root), window_s=0.1, checks=["gel"], episodes=[1])
        gel = one.episodes["episode_000001"]["glove"].gel
        assert gel.dead_elements == sorted(dead + [6, 10]) and gel.episodes_checked == 1 and not gel.flag

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
PRESSES = [(0.6, 0.4), (1.4, 0.4), (2.7, 0.4), (3.5, 0.4)]
"""Two presses in each 2 s episode (the first runs from about 0.2 s, the second from about 2.3 s of sensor
time). One press is an onset and a release quantized to two video-frame phases, and the skew
confidence of a clean pair then sits near the floor on an unlucky recording; two give four events."""
IMAGE_RATE = 60.0
DELAY_FRAMES = 2  # the image sensor shows the press two of its frames late: 33 ms at 60 Hz
DEAD_BLOCK = (10, 20, 6, 6)  # 36 pixels, 1.2% of the 48x64 image, over the 0.5% cluster default


def two_sensor_dataset(root: Path, *, faults: bool) -> Path:
    """An array sensor through the writer and an image sensor fed to ``add_frame`` by hand, two episodes."""
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
            seed=2,
            noise_std=6.0,  # camera noise well above the codec's flicker, as on a real gel image
            contact_schedule=[(s, d, 80.0) for s, d in PRESSES],
            dead_pixels=DEAD_BLOCK if faults else None,
            transport_delay_s=DELAY_FRAMES / IMAGE_RATE if faults else 0.0,
        )
    )
    array.connect()
    image.connect()
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
    for _ in range(2):
        writer.start_episode()
        for _ in range(60):
            values = writer.frame()  # at the frame's nominal time
            frame = image.read()  # the next image after it: fresh to within one image period
            dataset.add_frame({**values, GEL_KEY: frame.data, "task": "press"})
        writer.end_episode()
        dataset.save_episode()
        writer.idle()
    array.disconnect()
    image.disconnect()
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
            # Encoder output differs between machines: a pixel or two of the block's edge crosses the floor.
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
