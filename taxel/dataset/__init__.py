"""Writing tactile streams into a LeRobot dataset in the ``SCHEMA.md`` layout.

The core package depends on numpy only. pyarrow comes with the ``taxel[doctor]``
extra and LeRobot with ``taxel[lerobot]``, and every module here that needs them
imports them inside the functions that use them, or behind a module-level ``try``
that raises an ``ImportError`` naming the extra. ``import taxel.dataset`` therefore
never pulls in LeRobot or pyarrow, and a missing dependency fails at the first call
that needs it with the install command in the message.
"""

LEROBOT_EXTRA = "taxel[lerobot]"
"""The extra that installs LeRobot and pyarrow: ``pip install 'taxel[lerobot]'``."""

DOCTOR_EXTRA = "taxel[doctor]"
"""The extra that installs pyarrow and huggingface_hub, enough to read datasets without torch."""


def missing_dependency(name: str, cause: ImportError, extra: str = LEROBOT_EXTRA) -> ImportError:
    """Return the error raised when an optional dependency of this package is not installed.

    The message names the extra that brings it; for a ``[doctor]`` dependency it also
    names ``[lerobot]``, which includes everything ``[doctor]`` does.
    """
    also = f" (and with '{LEROBOT_EXTRA}')" if extra != LEROBOT_EXTRA else ""
    return ImportError(
        f"{name} is not installed; it comes with the '{extra}' extra{also}: pip install '{extra}'"
    ).with_traceback(cause.__traceback__)
