# Tactile data schema for LeRobot datasets — draft 0.2

*Status: draft for feedback, November 2026. Nothing here is final. The goal is to converge with [LeRobot PR #4469](https://github.com/huggingface/lerobot/pull/4469), [TLabel](https://github.com/liesliy/tlabel) and [Multi-Frequency-LeRobot](https://github.com/RoboCoin-BAAI/Multi-Frequency-LeRobot), not to add a fourth incompatible layout. Open questions are listed at the end; those are the ones I'd most like answered.*

## The problem this solves

A tactile feature in a LeRobot dataset today is typically an unnamed array: 116 floats per frame, or an image. Months later nobody can answer which sensor produced it, where each element sat on the fingertip, whether the gel was new or worn, what calibration converted raw readings to newtons, or what the sensor read with nothing touching it. Two sensors with different taxel counts cannot share one feature. Streams at different rates get resampled to the camera rate and lose their timing. None of this is visible until a model trained on the data fails on a fresh sensor.

## Six principles

1. **Per instance, not per type.** A dataset with two fingertips has two features, not one feature with a stacked dimension. Instances have stable IDs across sessions. Five fingertips on one hub are five instances that share a `device`.
2. **Metadata once, data per frame.** Everything static about a sensor instance (geometry, serial, gel ID, calibration, declared rate) lives in dataset metadata. Per-frame data is only what changes.
3. **Native rate, own timestamps.** When a sensor's rate differs from the dataset fps, its native-rate stream with device and host timestamps is required. A copy at the dataset fps is allowed for training code, provided the metadata says how it was made. Native timestamps are never discarded at write time.
4. **Provenance keys always exist.** `calibration`, `baseline` and `adapter` are present on every instance even when a value is `null`, so their absence is a visible fact rather than a missing key.
5. **Raw is always recoverable.** Any transformation applied at record time (baseline subtraction, rate reduction) is recorded with enough information to undo it or to know exactly what was lost.
6. **Health is an annotation, not a filter.** Quality scores are written alongside the data, per episode, and never silently drop frames or episodes.

## Feature keys

```
observation.tactile.<instance_id>
```

`instance_id` is a short slug chosen by the recorder and stable across sessions: `left_fingertip`, `right_fingertip`, `palm_0`, `thumb`. It must match a key under `tactile_sensors` in the metadata.

For visuotactile sensors the image stream is stored the way camera streams are (video-encoded), under the same key. The image is the ground-truth stream. Optional streams derived from it use a suffix:

```
observation.tactile.<instance_id>.markers
observation.tactile.<instance_id>.displacement
observation.tactile.<instance_id>.force
```

`.displacement` and `.force` are allowed only when the instance's metadata carries a non-null `derived` block naming the method that produced them.

## Metadata block

Added to the dataset's `info.json` (or wherever LeRobot core settles for sensor metadata) as `tactile_sensors`, one entry per instance.

```json
"tactile_sensors": {
  "left_fingertip": {
    "family": "taxel_array",
    "vendor": "PaXini",
    "model": "PX-6AX-GEN3 DP-S1813-Elite",
    "serial": "PX3-00412",
    "firmware": null,
    "device": {"id": "paxini_hub_0", "serial": "PXH-0193", "firmware": "1.4.2", "clock": "paxini_hub_0"},
    "adapter": {"name": "tactile-lerobot.paxini", "version": "0.1.0"},
    "declared_rate_hz": 1000,
    "units": {"force": "N"},
    "channels": ["fx", "fy", "fz"],
    "layout": {
      "frame": "sensor",
      "positions_mm": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], "..."],
      "normals": [[0, 0, 1], [0, 0, 1], "..."]
    },
    "calibration": {
      "version": "vendor-factory-2026-03",
      "method": "vendor",
      "date": "2026-03-14",
      "reference": null
    },
    "baseline": {
      "captured_at": "2026-11-21T14:03:11Z",
      "duration_s": 2.0,
      "applied": true,
      "values": [[0.01, -0.02, 0.00], "..."]
    },
    "reduction": {"method": "mean", "window_ms": 33.3},
    "mount": {"body": "so101_follower", "link": "gripper_left_finger", "pose": null}
  },

  "right_fingertip": {
    "family": "visuotactile",
    "vendor": "GelSight",
    "model": "DIGIT",
    "serial": "D20491",
    "firmware": null,
    "device": null,
    "adapter": {"name": "tactile-lerobot.digit", "version": "0.1.0"},
    "declared_rate_hz": 60,
    "image": {"width": 320, "height": 240, "channels": 3},
    "gel": {"id": "gel-2026-09-b", "installed_at": "2026-09-30", "cycles_at_install": 0},
    "markers": {"count": 63, "grid": [7, 9]},
    "derived": {"method": "marker-flow-depth", "version": "0.1.0"},
    "calibration": {
      "version": null,
      "method": null,
      "date": null,
      "reference": null
    },
    "baseline": {
      "captured_at": "2026-11-21T14:03:11Z",
      "duration_s": 2.0,
      "applied": false,
      "reference_frame": "episode_000000/frame_0000"
    },
    "mount": {"body": "so101_follower", "link": "gripper_right_finger", "pose": null}
  },

  "palm_0": {
    "family": "magnetic",
    "vendor": "NYU / AnySkin",
    "model": "AnySkin v1",
    "serial": null,
    "firmware": "qtpy-anyskin-0.3",
    "device": null,
    "adapter": {"name": "tactile-lerobot.anyskin", "version": "0.1.0"},
    "declared_rate_hz": 200,
    "units": {"field": "uT"},
    "channels": ["bx", "by", "bz"],
    "layout": {
      "frame": "sensor",
      "positions_mm": [[0, 0, 0], [6, 0, 0], [-6, 0, 0], [0, 6, 0], [0, -6, 0]],
      "normals": null
    },
    "skin": {"id": "skin-07", "installed_at": "2026-11-10"},
    "calibration": {"version": null, "method": "uncalibrated-by-design", "date": null, "reference": null},
    "baseline": {
      "captured_at": "2026-11-21T14:03:11Z",
      "duration_s": 2.0,
      "applied": false,
      "values": [[412.0, -88.5, 1203.2], "..."]
    },
    "mount": {"body": "so101_follower", "link": "palm", "pose": null}
  }
}
```

Field notes:

- `family` is one of `taxel_array`, `visuotactile`, `magnetic`, `force_torque`. Adding a family is a schema change; adding a vendor is not.
- `device` groups instances that share one physical connection, such as several fingertips on one serial hub. `id` is a slug unique in the dataset; `serial` and `firmware` are the hub's; `clock` names the timebase, and instances with the same `clock` value have directly comparable `device_ts`. `device` is `null` for a sensor with its own connection. Per-fingertip identity (serial, gel or skin, calibration, wear) stays on the instance, so replacing one fingertip does not change the identity of the others.
- `adapter` names the code that turned the sensor's output into the stored values, and its version. A bug found in an adapter can then be traced to the datasets it affected.
- `layout.positions_mm` is the physical position of each element in the sensor's own frame. For taxel arrays and magnetic sensors its length equals the per-frame element count. For visuotactile sensors the image itself is the layout; `markers.grid` gives the marker arrangement.
- `gel` and `skin` are the replaceable-part identifiers. They are what make "this sensor degraded 30% faster than that one" answerable later.
- `calibration.method` is free text for now (`vendor`, `3dcal`, `nano17-press`, `uncalibrated-by-design`). A controlled vocabulary can come later once there are a few real values.
- `baseline` is what the sensor read with nothing touching it at session start. For taxel and magnetic families it is stored as values; for visuotactile it is a pointer to a reference frame. `applied` says whether the values were subtracted at record time. When `applied` is `true`, stored data plus `baseline.values` must reconstruct the raw readings exactly. `applied: true` is valid only where the baseline is stored as values.
- `reduction` is present only when the main table carries a copy of this instance at the dataset fps alongside a native-rate stream. `method` is one of `nearest`, `mean`, `last`; `window_ms` is the span of native samples each dataset frame was reduced from.
- `derived` is required when `.displacement` or `.force` streams are present, and names the algorithm (`method`) and its `version`. Point count and order are defined by that method.
- `mount` records where on the robot the sensor sits. `pose` is a 4×4 or (xyz, quat) in the link frame if known, else `null`.

## Per-frame data

| Family | dtype | Shape per frame | Notes |
| --- | --- | --- | --- |
| `taxel_array` | float32 | `(n_elements, n_channels)` | Never uint8. The PaXini resultant mode caps Fz at 25.5 N in uint8; store distributed mode, or scale correctly, or both |
| `visuotactile` | uint8 | `(H, W, 3)` video-encoded | Encoded like camera streams |
| `visuotactile` `.markers` | float32 | `(n_markers, 2)` | Optional; marker pixel coordinates |
| `visuotactile` `.displacement` | float32 | `(N, 3)` | Optional; dx, dy, dz; requires `derived` |
| `visuotactile` `.force` | float32 | `(N, 3)` | Optional; fx, fy, fz; requires `derived` |
| `magnetic` | float32 | `(n_sensors, 3)` | Raw field. Baseline-subtracted view is computed at read time, not stored |
| `force_torque` | float32 | `(6,)` | fx, fy, fz, mx, my, mz in the sensor frame |

Every family's per-frame row also carries:

| Column | dtype | Meaning |
| --- | --- | --- |
| `device_ts` | float64 or null | Timestamp from the sensor's own clock, seconds, if the device provides one |
| `host_ts` | float64 | Host receive time, seconds, monotonic clock |
| `seq` | int64 or null | Device sequence number if provided; gaps mean dropped frames at the source |

## Timestamps and alignment

When a sensor's rate equals the dataset fps, its main-table column is the native stream and nothing else is needed.

When the rates differ, the native-rate stream with `device_ts`, `host_ts` and `seq` is required. Where it lives is not fixed yet: a per-sensor table (the [Multi-Frequency-LeRobot](https://github.com/RoboCoin-BAAI/Multi-Frequency-LeRobot) approach) or a block inside the main table. The schema requires the stream, not the mechanism; the choice will follow a write-path test against `LeRobotDataset`.

A copy at the dataset fps in the main table is optional and recommended, since most training code reads one table. When present, the instance's `reduction` block says how it was made. A reader that needs the timing aligns from the native stream by nearest `host_ts` or by a window; the alignment policy lives in the loader, not the file.

A dataset-rate copy cannot show dropped frames or clock skew. Those are only visible in the native stream, which is why it is never optional when the rates differ.

## Health annotations

`tactile doctor` writes a per-episode block the dataset can carry without changing any data:

```json
"tactile_health": {
  "episode_000012": {
    "left_fingertip": {
      "drift": {"offset_pct_range": 0.031, "flag": false},
      "framerate": {"effective_hz": 982.4, "dropped_pct": 0.018, "max_gap_ms": 14.0, "flag": false},
      "saturation": {"saturated_pct": 0.0, "flag": false},
      "skew": {"lag_ms": -12.0, "confidence": 0.81, "flag": false}
    },
    "right_fingertip": {
      "gel": {"dead_pct": 0.062, "largest_cluster_px": 41, "flag": true},
      "framerate": {"effective_hz": 29.1, "dropped_pct": 0.51, "max_gap_ms": 220.0, "flag": true},
      "skew": {"lag_ms": 35.0, "confidence": 0.67, "flag": false}
    },
    "doctor_version": "0.1.0",
    "thresholds": "default"
  }
}
```

Thresholds are conservative defaults and are recorded with the result so a later run with different thresholds is distinguishable.

## Interoperability

- **PR #4469.** The `TactileSensor` base class in this project follows that PR's method surface (`connect`, `read`, `async_read`, `read_latest`, `disconnect`) and config registry, so a merge or dependency is possible. Type mapping: `taxel_array` → `force`, `force_torque` → `wrench`; the PR's `displacement` and `force` point sets correspond to the optional `.displacement` and `.force` streams here. `magnetic` and visuotactile images have no PR type. The PR's `read()` returns no timestamps or per-sensor metadata and the PR defines no dataset layer; those are what this schema adds. A recording made with the PR's record-time tare is compliant if the tare values are stored with `baseline.applied: true`.
- **TLabel.** TLabel is an annotation layer: per-frame labels (contact, slip, force magnitude, manipulation phase and others) computed from raw data. It sits above this schema, not beside it. Overlap is in the sensor description: TLabel's `manufacturer`, `model`, `sensor_type`, `frame_rate` and `provenance` (firmware, calibration date, sampling rate) map to `vendor`, `model`, `family`, `declared_rate_hz`, `firmware` and `calibration.date`. A converter would feed these recordings to TLabel's adapters, one instance at a time, and could store TLabel labels back alongside the raw data per instance.
- **MCAP.** One channel per instance, schema per family, `host_ts` as the log time, `device_ts` as publish time where present. Camera and joint channels exported alongside so Foxglove opens the whole episode.

## Open questions (please answer any of these)

1. Per-instance keys: does any training code need a stacked per-hub view (five fingertips as one array), and if so, should the loader build it from the `device` grouping rather than storing it?
2. Is `layout.positions_mm` the right level of geometry, or do people need full per-element frames (position + orientation)? Does anyone actually use this in a policy today?
3. For visuotactile sensors, is a reference frame pointer enough for `baseline`, or should a small reference image be stored in metadata directly?
4. What does your pipeline do with the no-contact baseline today? Subtract at record time, at load time, or never?
5. Multi-rate: would a native-rate stream plus an optional dataset-rate copy in the main table work for your loader? Which reduction do you use when you downsample (nearest, mean, last, something else)?
6. What calibration methods are in use that should be in a controlled vocabulary?
7. For derived point-set streams (`.displacement`, `.force`), should point positions be stored, and where: per frame, or once in metadata?
8. What is missing that would make you refuse to use this?

## Changelog

- 0.2 (2026-10-29):
  - Timing: a native-rate stream with device and host timestamps is required whenever the sensor rate differs from the dataset fps; an optional dataset-rate copy in the main table must declare its `reduction` (`method`, `window_ms`).
  - Baseline: `baseline.applied` added. Record-time subtraction is allowed if the baseline values are stored; stored data plus baseline must reconstruct raw.
  - `device` block (`id`, `serial`, `firmware`, `clock`) groups instances that share a hub; each fingertip remains its own instance.
  - Torque channels renamed `tx/ty/tz` → `mx/my/mz`, matching PR #4469 and TLabel.
  - Optional derived visuotactile streams `.displacement` and `.force`, float32 `(N, 3)`, allowed only with a `derived` block (`method`, `version`).
  - `adapter` block (`name`, `version`) added to every instance as a provenance key.
  - New principle 5, "raw is always recoverable".
  - Interoperability rewritten against PR #4469's code and TLabel's spec. Open questions 1, 5 and 7 revised.
  - Pending a write-path test against `LeRobotDataset`: where the native-rate stream is stored, and whether the sibling `.displacement` / `.force` keys load cleanly. If sibling keys do not, those streams move to 0.3.
- 0.1 (Nov 2026): first draft for feedback.
