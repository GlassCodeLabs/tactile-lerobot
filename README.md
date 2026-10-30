# tactile-lerobot

The Python package is `taxel`; the repository name describes its role as the LeRobot tactile plugin.

An open-source LeRobot plugin for tactile sensors, plus a health-check CLI for tactile recordings.

**What it does**

- One LeRobot feature key per tactile sensor *instance*, with the sensor's geometry, serial, gel or skin ID and calibration version carried in dataset metadata
- Native-rate storage: a 1 kHz taxel array and a 30 Hz camera keep their own timestamps
- Adapters for common sensors, each under 200 lines, with a recipe for adding your own
- `tactile doctor`: scores every episode for baseline drift, dead pixels and gel condition, dropped frames, saturation and clock skew against the video stream, and writes a JSON report you can diff across sessions
- MCAP export so Foxglove and Rerun open tactile data alongside video and joints

**Status:** the `taxel` package contains the schema dataclasses ([SCHEMA.md](SCHEMA.md)), the `TactileSensor` base class and a simulated backend with injectable faults ([docs/sensors.md](docs/sensors.md)); the dataset writer is the next component.

**Why**

Tactile data goes into training sets with no record of which sensor produced it, where each taxel sat, whether the gel was worn, or which calibration converted the numbers. Drift and wear produce plausible-looking data that quietly corrupts policies. Every lab and startup rebuilds the same ingestion and QA plumbing. Several single-sensor integrations already exist — Columbia's [LeFlexiTac](https://github.com/TNA001-AI/lerobot_tactile) fork for FlexiTac, Xense's [LeRobot fork](https://github.com/XenseRobotics-AI/xense-taccap-lerobot), a community [PaXini plugin](https://github.com/Jingyi-Z/robotics_notes) — each solving it for one sensor in its own way. This project is the neutral, sensor-agnostic layer that lets those live side by side as plugins, built to converge with the LeRobot tactile work in [PR #3175](https://github.com/huggingface/lerobot/pull/3175) and [PR #4469](https://github.com/huggingface/lerobot/pull/4469) and the format conversion in [TLabel](https://github.com/liesliy/tlabel), not to compete with any of them.

**Feedback wanted**

If you record tactile data and the schema would break for your sensor or your pipeline, open an issue. Sensor owners willing to test an adapter: same.

MIT licence.
