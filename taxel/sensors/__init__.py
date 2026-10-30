"""Sensor backends: the ``TactileSensor`` contract and its implementations."""

from taxel.sensors.base import (
    TactileReading,
    TactileSensor,
    TactileSensorConfig,
    host_clock,
    make_tactile_sensor,
)

__all__ = [
    "TactileReading",
    "TactileSensor",
    "TactileSensorConfig",
    "host_clock",
    "make_tactile_sensor",
]
