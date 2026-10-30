"""Sensor backends: the ``TactileSensor`` contract and its implementations."""

from taxel.sensors.base import (
    TactileReading,
    TactileSensor,
    TactileSensorConfig,
    host_clock,
    make_tactile_sensor,
)
from taxel.sensors.simulated import SimulatedTactile, SimulatedTactileConfig

__all__ = [
    "SimulatedTactile",
    "SimulatedTactileConfig",
    "TactileReading",
    "TactileSensor",
    "TactileSensorConfig",
    "host_clock",
    "make_tactile_sensor",
]
