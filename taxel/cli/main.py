"""Entry point of the ``taxel`` command: ``taxel <subcommand> [options]``.

Subcommands live in their own modules and are imported only when invoked, so
``taxel --help`` costs nothing and a missing optional dependency is reported by
the subcommand that needs it.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

from taxel import __version__

SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "record": ("taxel.cli.record", "record a LeRobot dataset with tactile sensors beside a robot or cameras"),
}


def usage() -> str:
    lines = ["usage: taxel <command> [options]", "", "commands:"]
    lines += [f"  {name:<10} {summary}" for name, (_, summary) in SUBCOMMANDS.items()]
    lines += ["", "run `taxel <command> --help` for a command's options"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to a subcommand and return its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(usage(), file=sys.stderr)
        return 2
    if args[0] in ("-h", "--help"):
        print(usage())
        return 0
    if args[0] == "--version":
        print(f"taxel {__version__}")
        return 0
    name, rest = args[0], args[1:]
    if name not in SUBCOMMANDS:
        print(f"taxel: unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    try:
        module = importlib.import_module(SUBCOMMANDS[name][0])
    except ImportError as e:
        # A missing optional dependency; the message names the extra to install.
        print(f"taxel {name}: {e}", file=sys.stderr)
        return 2
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    sys.exit(main())
