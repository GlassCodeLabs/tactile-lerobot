"""``taxel doctor`` on a public dataset, gated behind ``TAXEL_NETWORK_TESTS=1``.

The dataset is ``RoboSynapse/univtac-isaac51_franka-panda_lift-can-v4``: a LeRobot v3.0 re-collection
of the UniVTAC lift-can task in simulation, 100 episodes at 60 fps, with a 20-value tactile descriptor
per frame (ten numbers per GelSight Mini finger, derived from the marker field) and two robot cameras.
It has no sidecar, so the checks degrade to what the main table allows. Its README documents that the
marker displacement reference is the first frame of each episode because the simulator's built-in
reference markers are stale: a per-episode re-zero the data cannot undo, which the drift check names
as ``first_frame_zero``.
"""

import json
import os

import pytest

pytest.importorskip("lerobot")

from taxel.cli.doctor import main  # noqa: E402
from taxel.schema import HealthReport, Skipped  # noqa: E402

REPO_ID = "RoboSynapse/univtac-isaac51_franka-panda_lift-can-v4"
CAMERAS = {"observation.images.head", "observation.images.wrist"}

pytestmark = pytest.mark.skipif(
    os.environ.get("TAXEL_NETWORK_TESTS") != "1", reason="set TAXEL_NETWORK_TESTS=1 to download the dataset"
)


def test_univtac_lift_can(tmp_path, capsys) -> None:
    out_path = tmp_path / "health.json"
    assert main([REPO_ID, "--no-write", "--json", str(out_path)]) == 0
    table = capsys.readouterr().out
    report = HealthReport.from_dict(json.loads(out_path.read_text()))

    assert list(report.instances) == ["tactile"]
    instance = report.instances["tactile"]
    assert instance.key == "observation.tactile" and instance.family == "array"
    assert instance.metadata == "none" and not instance.sidecar and instance.declared_rate_hz is None
    assert len(report.episodes) == 100

    first_frame_zero = 0
    lags = []
    for block in report.episodes.values():
        checks = block["tactile"]
        assert isinstance(checks.integrity, Skipped) and checks.integrity.skipped.startswith("no sidecar")
        assert checks.framerate.source == "main_table" and checks.framerate.timestamps == "nominal"
        assert not checks.framerate.flag and checks.framerate.dropped_pct is None
        assert checks.saturation.limit == "observed" and checks.saturation.confidence == "low"
        assert not isinstance(checks.drift, Skipped)
        assert checks.drift.baseline == "none" and isinstance(checks.drift.offset, Skipped)
        assert checks.drift.confidence == "low" and not checks.drift.flag  # numbers in unknown units, no flag
        first_frame_zero += checks.drift.first_frame_zero
        assert checks.gel.kind == "elements"
        if not isinstance(checks.skew, Skipped):
            assert checks.skew.reference in CAMERAS
            lags.append(checks.skew.lag_ms)
    assert first_frame_zero == 100  # the first-frame reference, in every episode
    assert "first frame zero" in table
    print(f"skew measured in {len(lags)} of 100 episodes; lags {sorted(lags)[:3]} .. {sorted(lags)[-3:]} ms")
