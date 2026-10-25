# tactile-lerobot

*Working name. Will be renamed when the package is published.*

An open-source LeRobot plugin for tactile sensors, plus a health-check CLI for tactile recordings.

**What it does**

- One LeRobot feature key per tactile sensor *instance*, with the sensor's geometry, serial, gel or skin ID and calibration version carried in dataset metadata
- Native-rate storage: a 1 kHz taxel array and a 30 Hz camera keep their own timestamps
- Adapters for common sensors, each under 200 lines, with a recipe for adding your own
- `tactile doctor`: scores every episode for baseline drift, dead pixels and gel condition, dropped frames, saturation and clock skew against the video stream, and writes a JSON report you can diff across sessions
- MCAP export so Foxglove and Rerun open tactile data alongside video and joints

**Status (November 2026):** schema drafting. See [SCHEMA.md](SCHEMA.md). First adapters planned: PaXini PX-6AX GEN3, AnySkin, DIGIT / GelSight Mini, plus a simulated backend for CI.

**Why**

Tactile data goes into training sets with no record of which sensor produced it, where each taxel sat, whether the gel was worn, or which calibration converted the numbers. Drift and wear produce plausible-looking data that quietly corrupts policies. Every lab and startup rebuilds the same ingestion and QA plumbing. This project is the neutral, sensor-agnostic version of that plumbing, built to converge with the LeRobot tactile work in [PR #4469](https://github.com/huggingface/lerobot/pull/4469) and the format conversion in [TLabel](https://github.com/liesliy/tlabel), not to compete with them.

**Feedback wanted**

If you record tactile data and the schema would break for your sensor or your pipeline, open an issue. Sensor owners willing to test an adapter: same.

MIT licence.
