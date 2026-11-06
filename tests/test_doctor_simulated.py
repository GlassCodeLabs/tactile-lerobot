"""End to end: record with ``taxel record`` from simulated sensors with injected faults, run the doctor,
and check what it finds; then the same on a clean recording, a tampered copy, and a plain dataset."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from test_loader import plain_dataset  # noqa: E402

from taxel.cli.main import main  # noqa: E402
from taxel.dataset.sidecar import episode_path, read_episode, write_episode  # noqa: E402
from taxel.doctor.loader import DoctorDataset  # noqa: E402
from taxel.doctor.report import run_doctor, write_report  # noqa: E402
from taxel.schema import DriftReport, HealthReport, IntegrityReport, Skipped, Thresholds  # noqa: E402
from taxel.sensors import TactileReading  # noqa: E402

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
            assert isinstance(ep.gel, Skipped) and isinstance(ep.skew, Skipped)
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
