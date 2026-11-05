"""Sensor backends: the ``TactileSensor`` contract and its implementations."""

from taxel.sensors.base import (
    TactileReading,
    TactileSensor,
    TactileSensorConfig,
    host_clock,
    make_tactile_sensor,
)
from taxel.sensors.simulated import (
    SimulatedTactile,
    SimulatedTactileConfig,
    SimulatedVisuotactile,
    SimulatedVisuotactileConfig,
)

__all__ = [
    "SimulatedTactile",
    "SimulatedTactileConfig",
    "SimulatedVisuotactile",
    "SimulatedVisuotactileConfig",
    "TactileReading",
    "TactileSensor",
    "TactileSensorConfig",
    "host_clock",
    "make_tactile_sensor",
]
