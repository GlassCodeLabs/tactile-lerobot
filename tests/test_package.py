import subprocess
import sys
from importlib.metadata import version

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
