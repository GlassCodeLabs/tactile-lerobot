"""Writing tactile streams into a LeRobot dataset in the ``SCHEMA.md`` layout.

The core package depends on numpy only. LeRobot and pyarrow come with the
``taxel[lerobot]`` extra, and every module here that needs them imports them
inside the functions that use them, or behind a module-level ``try`` that
raises an ``ImportError`` naming the extra. ``import taxel.dataset`` therefore
never pulls in LeRobot, and a missing dependency fails at the first call that
needs it with the install command in the message.
"""

LEROBOT_EXTRA = "taxel[lerobot]"
"""The extra that installs LeRobot and pyarrow: ``pip install 'taxel[lerobot]'``."""


def missing_dependency(name: str, cause: ImportError) -> ImportError:
    """Return the error raised when an optional dependency of this package is not installed."""
    return ImportError(
        f"{name} is not installed; it comes with the '{LEROBOT_EXTRA}' extra: pip install '{LEROBOT_EXTRA}'"
    ).with_traceback(cause.__traceback__)
