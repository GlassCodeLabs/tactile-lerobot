from importlib.metadata import version

import taxel


def test_installed_version_matches_module() -> None:
    assert version("taxel") == taxel.__version__
