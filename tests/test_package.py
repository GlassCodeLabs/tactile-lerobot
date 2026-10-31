import importlib.util
import subprocess
import sys
from importlib.metadata import version

import pytest

import taxel


def test_installed_version_matches_module() -> None:
    assert version("taxel") == taxel.__version__


def test_core_packages_import_without_optional_dependencies() -> None:
    """Importing the sensors, the schema and the package roots of the optional parts must not import
    LeRobot, draccus, torch or pyarrow, so the numpy-only install stays numpy-only at runtime."""
    script = (
        "import sys, taxel.schema, taxel.sensors, taxel.dataset, taxel.cli; "
        "print(sorted(m.split('.')[0] for m in sys.modules "
        "if m.split('.')[0] in {'lerobot', 'draccus', 'torch', 'pyarrow', 'datasets'}))"
    )
    out = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "[]"


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is not None, reason="LeRobot is installed")
def test_record_without_the_extra_names_it(capsys) -> None:
    from taxel.cli.main import main

    assert main(["record", "--robot.type=cameras_only"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("taxel record: LeRobot is not installed") and "pip install 'taxel[lerobot]'" in err
