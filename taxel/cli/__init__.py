"""The ``taxel`` command line: ``taxel record`` and the subcommands that join it later.

Everything here depends on the ``taxel[lerobot]`` extra: the record command is
built on LeRobot's robot, camera and dataset code and parses its options with
draccus, as ``lerobot-record`` does. Those imports live in the subcommand
modules behind a module-level ``try`` that names the extra, so importing this
package is free and ``taxel.sensors`` stays independent of LeRobot.
"""
