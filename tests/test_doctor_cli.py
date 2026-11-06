"""``taxel doctor``: option parsing, the table, the report file and the exit codes."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from taxel.cli.doctor import (
    UsageError,
    format_report,
    main,
    parse_episodes,
    parse_fail_on,
    read_thresholds,
)
from taxel.schema import HealthReport, Thresholds

needs_lerobot = pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="needs taxel[lerobot]")


class TestParsing:
    def test_episodes(self) -> None:
        assert parse_episodes("0,3-5") == [0, 3, 4, 5]
        assert parse_episodes("7") == [7] and parse_episodes("2, 1-2") == [1, 2]
        for bad in ("a", "5-3", "-1", ",", "1-"):
            with pytest.raises(UsageError, match="--episodes"):
                parse_episodes(bad)

    def test_fail_on(self) -> None:
        thresholds, checks = parse_fail_on("drift=0.1,integrity,cluster_px=30", Thresholds())
        assert thresholds.drift == 0.1 and thresholds.cluster_px == 30 and thresholds.dropped == 0.02
        assert checks == {"drift", "integrity", "gel"}
        thresholds, checks = parse_fail_on("drift_units=0.2", Thresholds())
        assert thresholds.drift_units == 0.2 and checks == {"drift"}
        assert parse_fail_on("dropped,max_gap_ms=40", Thresholds())[1] == {"framerate"}
        assert parse_fail_on("", Thresholds()) == (Thresholds(), set())
        with pytest.raises(UsageError, match="unknown check 'noise'"):
            parse_fail_on("noise=1", Thresholds())
        with pytest.raises(UsageError, match="not a number"):
            parse_fail_on("drift=high", Thresholds())
        with pytest.raises(UsageError, match="takes no value"):
            parse_fail_on("integrity=1", Thresholds())

    def test_thresholds_file(self, tmp_path) -> None:
        path = tmp_path / "t.json"
        path.write_text('{"saturated": 0.5, "max_gap_ms": 40}')
        thresholds = read_thresholds(path)
        assert thresholds.saturated == 0.5 and thresholds.max_gap_ms == 40 and thresholds.drift == 0.05
        path.write_text('{"noise": 1}')
        with pytest.raises(UsageError, match="unknown keys"):
            read_thresholds(path)
        path.write_text("[1, 2]")
        with pytest.raises(UsageError, match="JSON object"):
            read_thresholds(path)
        with pytest.raises(UsageError, match="cannot read"):
            read_thresholds(tmp_path / "missing.json")

    def test_table_from_the_schema_example(self) -> None:
        from test_schema import HEALTH

        text = format_report(HealthReport.from_dict(HEALTH))
        assert text.rstrip().endswith(
            "1 episode(s), 1 instance(s), 0 flag(s); ! marks a flag, - a skipped check"
        )
        assert "left_fingertip  observation.tactile.left_fingertip  taxel_array" in text
        assert "missing keys meta.sessions[0].baseline.frame_range" in text
        assert "000012" in text and "ok copy,raw,files" in text and ", 1 rebaseline" in text
        assert "skipped gel: no visuotactile feature (1 episode)" in text
        plain = copy.deepcopy(HEALTH)
        drift = plain["episodes"]["episode_000012"]["left_fingertip"]["drift"]
        drift["offset"] = {"skipped": "no session baseline"}
        drift["first_frame_zero"] = True
        text = format_report(HealthReport.from_dict(plain))
        assert "level change +0.002, 1 rebaseline, first frame zero" in text
        drift["confidence"] = "low"
        text = format_report(HealthReport.from_dict(plain))
        assert "level change +0.002, 1 rebaseline, first frame zero, low confidence" in text
        plain["episodes"]["episode_000013"] = copy.deepcopy(plain["episodes"]["episode_000012"])
        for episode, value in (("episode_000012", "0.41"), ("episode_000013", "0.37")):
            plain["episodes"][episode]["left_fingertip"]["skew"] = {
                "skipped": f"no onset matched in both signals: confidence {value} below 0.5"
            }
        text = format_report(HealthReport.from_dict(plain))
        assert "skipped skew: no onset matched in both signals: confidence below 0.5 (2 episodes)" in text
        assert text.rstrip().endswith(
            "2 episode(s), 1 instance(s), 0 flag(s); ! marks a flag, - a skipped check"
        )

    def test_usage(self, capsys) -> None:
        assert main([]) == 2
        assert main(["--help"]) == 0 and "--fail-on" in capsys.readouterr().out
        assert main(["/nowhere/at/all", "--no-write"]) == 2
        assert "does not look like a Hub repo id" in capsys.readouterr().err
        assert main([".", "--episodes", "x", "--no-write"]) == 2
        assert main([".", "--window-s", "0", "--no-write"]) == 2


@pytest.fixture(scope="module")
def faulty(tmp_path_factory) -> Path:
    pytest.importorskip("lerobot")
    from test_doctor_simulated import FAULTS, record

    return record(tmp_path_factory.mktemp("cli"), FAULTS)


@needs_lerobot
class TestRun:
    def test_writes_the_report_and_prints_the_table(self, faulty, capsys) -> None:
        assert main([str(faulty)]) == 0
        out = capsys.readouterr().out
        assert (
            "left_fingertip  observation.tactile.left_fingertip  taxel_array  metadata: info  sidecar: yes"
            in out
        )
        assert "000000" in out and "000001" in out and " !" in out
        assert (
            "skipped skew: no reference: the dataset has no camera and no other tactile instance (2 episodes)"
            in out
        )
        assert f"written {faulty / 'tactile' / 'health.json'}" in out
        report = HealthReport.from_dict(json.loads((faulty / "tactile" / "health.json").read_text()))
        assert list(report.episodes) == ["episode_000000", "episode_000001"]
        assert report.episodes["episode_000000"]["left_fingertip"].framerate.flag

    def test_no_write_and_json_path(self, faulty, tmp_path, capsys) -> None:
        (faulty / "tactile" / "health.json").unlink(missing_ok=True)
        out_path = tmp_path / "report.json"
        assert main([str(faulty), "--no-write", "--json", str(out_path), "--episodes", "1"]) == 0
        assert not (faulty / "tactile" / "health.json").exists()
        report = HealthReport.from_dict(json.loads(out_path.read_text()))
        assert list(report.episodes) == ["episode_000001"]
        assert "written" in capsys.readouterr().out

    def test_fail_on_sets_the_exit_code(self, faulty, capsys) -> None:
        assert main([str(faulty), "--no-write", "--fail-on", "dropped"]) == 1
        out = capsys.readouterr().out
        assert "fail: episode_000000 left_fingertip framerate" in out
        assert (
            main([str(faulty), "--no-write", "--fail-on", "dropped=0.5"]) == 0
        )  # threshold raised above the drops
        assert main([str(faulty), "--no-write", "--fail-on", "drift,saturated,integrity"]) == 1
        assert "fail: episode_000000 left_fingertip saturation" in capsys.readouterr().out
        assert main([str(faulty), "--no-write", "--fail-on", "integrity"]) == 0
        assert (
            main([str(faulty), "--no-write", "--fail-on", "gel,skew"]) == 0
        )  # skipped and clean checks never fail

    def test_thresholds_file_is_recorded(self, faulty, tmp_path) -> None:
        path = tmp_path / "t.json"
        path.write_text('{"saturated": 0.5, "dropped": 0.5}')
        out_path = tmp_path / "report.json"
        assert main([str(faulty), "--no-write", "--thresholds", str(path), "--json", str(out_path)]) == 0
        report = HealthReport.from_dict(json.loads(out_path.read_text()))
        assert report.thresholds.saturated == 0.5 and report.thresholds.dropped == 0.5
        first = report.episodes["episode_000000"]["left_fingertip"]
        assert not first.saturation.flag and not first.framerate.flag and first.drift.flag

    def test_selection_errors(self, faulty, capsys) -> None:
        assert main([str(faulty), "--no-write", "--instances", "thumb"]) == 2
        assert "neither a tactile instance" in capsys.readouterr().err
        assert main([str(faulty), "--no-write", "--episodes", "7"]) == 2
        assert "no episodes [7]" in capsys.readouterr().err
        assert main([str(faulty), "--no-write", "--reference", "observation.images.none"]) == 2
        assert "not a feature" in capsys.readouterr().err
