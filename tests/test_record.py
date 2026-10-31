"""``taxel record`` with a simulated sensor, no cameras, no robot, headless."""

import json

import pytest

pytest.importorskip("lerobot")

from tactile_checks import check_recording  # noqa: E402

from taxel.cli.main import main  # noqa: E402
from taxel.dataset.sidecar import read_session_meta  # noqa: E402
from taxel.sensors import SimulatedTactile, SimulatedTactileConfig  # noqa: E402

REPO_ID = "taxel/record_test"
SENSOR = {
    "type": "simulated",
    "declared_rate_hz": 500,
    "n_elements": 16,
    "drop_probability": 0.05,
    "clock_offset_s": 0.25,
    "seed": 1,
}


def is_dropped(seq: int) -> bool:
    config = SimulatedTactileConfig(
        instance_id="left_fingertip", **{k: v for k, v in SENSOR.items() if k != "type"}
    )
    return SimulatedTactile(config)._is_dropped(seq)


def tactile_arg(**overrides) -> str:
    return "--tactile=" + json.dumps({"left_fingertip": {**SENSOR, **overrides}})


def record_args(tmp_path, *extra: str) -> list[str]:
    return [
        "record",
        "--robot.type=cameras_only",
        f"--robot.calibration_dir={tmp_path / 'calibration'}",
        tactile_arg(),
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.root={tmp_path / 'ds'}",
        "--dataset.no_stamp=true",
        "--dataset.num_episodes=2",
        "--dataset.episode_time_s=1",
        "--dataset.reset_time_s=0.3",
        "--dataset.fps=30",
        "--dataset.single_task=press",
        "--baseline_s=0.2",
        "--play_sounds=false",
        *extra,
    ]


class TestUsage:
    def test_top_level(self, capsys) -> None:
        assert main([]) == 2
        assert "usage: taxel <command>" in capsys.readouterr().err
        assert main(["--help"]) == 0
        assert "record" in capsys.readouterr().out
        assert main(["--version"]) == 0
        assert capsys.readouterr().out.startswith("taxel ")
        assert main(["doctor"]) == 2
        assert "unknown command 'doctor'" in capsys.readouterr().err

    def test_record_help_is_scoped_to_the_robot_type(self, capsys) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["record", "--robot.type=cameras_only", "--help"])
        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        assert "--robot.cameras" in out and "--tactile" in out and "--dataset.push_to_hub" in out

    def test_option_errors(self, tmp_path, capsys) -> None:
        assert main(record_args(tmp_path, "--reduction=nearest")) == 2
        assert "--reduction must be mean or last" in capsys.readouterr().err
        args = [a for a in record_args(tmp_path) if not a.startswith("--tactile=")]
        assert main(args) == 2
        assert "at least one tactile sensor" in capsys.readouterr().err
        assert main(record_args(tmp_path, tactile_arg(instance_id="thumb"))) == 2
        assert capsys.readouterr().err == "error: --tactile key 'left_fingertip' is not instance_id 'thumb'\n"
        assert not (tmp_path / "ds").exists()


class TestRecording:
    def test_two_episodes_headless(self, tmp_path) -> None:
        assert main(record_args(tmp_path)) == 0
        root = tmp_path / "ds"
        dataset, info, record = check_recording(
            root, REPO_ID, "left_fingertip", is_dropped=is_dropped, device_offset_s=0.25
        )
        assert dataset.num_episodes == 2 and record.episodes == (0, 1)
        assert dataset.meta.robot_type == "cameras_only"
        assert set(dataset.meta.features) == {
            "observation.tactile.left_fingertip",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        }
        for e in range(2):
            assert 25 <= int(dataset.meta.episodes[e]["length"]) <= 35  # one second at 30 fps
        assert info.reduction.method == "mean" and info.reduction.window_ms == 33.333
        assert record.baseline is not None and record.baseline.duration_s == 0.2
        assert record.adapter.name == "taxel.simulated"
        assert dataset.meta.tasks is not None and "press" in dataset.meta.tasks.index

    def test_resume_appends_a_session(self, tmp_path) -> None:
        assert main(record_args(tmp_path, "--session_id=first")) == 0
        assert (
            main(record_args(tmp_path, "--resume=true", "--session_id=second", "--dataset.num_episodes=1"))
            == 0
        )
        root = tmp_path / "ds"
        dataset, _, record = check_recording(
            root, REPO_ID, "left_fingertip", is_dropped=is_dropped, session=-1
        )
        assert dataset.num_episodes == 3 and record.session_id == "second" and record.episodes == (2, 2)
        sessions = read_session_meta(root, "left_fingertip").sessions
        assert [(s.session_id, s.episodes) for s in sessions] == [("first", (0, 1)), ("second", (2, 2))]

    def test_resume_refuses_a_different_setup(self, tmp_path, capsys) -> None:
        assert main(record_args(tmp_path, "--dataset.num_episodes=1")) == 0
        assert main(record_args(tmp_path, "--resume=true", "--dataset.fps=25")) == 2
        assert "recorded at 30 fps, not 25" in capsys.readouterr().err
        assert main(record_args(tmp_path, "--resume=true", tactile_arg(declared_rate_hz=400))) == 2
        assert "does not match the metadata" in capsys.readouterr().err
