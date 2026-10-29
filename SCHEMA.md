# Tactile data schema for LeRobot datasets — draft 0.2

*Status: draft for feedback, October 2026. Nothing here is final. The goal is to converge with [LeRobot PR #4469](https://github.com/huggingface/lerobot/pull/4469), [TLabel](https://github.com/liesliy/tlabel) and [Multi-Frequency-LeRobot](https://github.com/RoboCoin-BAAI/Multi-Frequency-LeRobot), not to add a fourth incompatible layout. Open questions are listed at the end; those are the ones I'd most like answered.*

Everything below targets LeRobot 0.6.1 (dataset format v3.0) and was checked against it: an `(N, C)` float32 tactile feature with per-instance metadata in its `info` entry writes, reloads and resumes through the public `LeRobotDataset` API with no change to LeRobot core. [`examples/write_tactile_feature.py`](examples/write_tactile_feature.py) shows the round trip.

## The problem this solves

A tactile feature in a LeRobot dataset today is typically an unnamed array: 116 floats per frame, or an image. Months later nobody can answer which sensor produced it, where each element sat on the fingertip, whether the gel was new or worn, what calibration converted raw readings to newtons, or what the sensor read with nothing touching it. Two sensors with different taxel counts cannot share one feature. Streams at different rates get resampled to the camera rate and lose their timing. None of this is visible until a model trained on the data fails on a fresh sensor.

## Six principles

1. **Per instance, not per type.** A dataset with two fingertips has two features, not one feature with a stacked dimension. Instances have stable IDs across sessions. Five fingertips on one hub are five instances that share a `device`.
2. **Static metadata in the feature, per-session facts beside it.** What is true of an instance in every session (family, geometry, channels, declared rate) lives in its feature entry's `info`. What can change between sessions (serial after a swap, calibration, baseline, gel or skin) lives in a per-instance sidecar. Per-frame data is only what changes per frame.
3. **Native rate, own timestamps.** When a sensor's rate differs from the dataset fps, its native-rate stream with device and host timestamps is required (for visuotactile images, see the `reduction` note under Static metadata). A copy at the dataset fps is allowed for training code, provided the metadata says how it was made. Native timestamps are never discarded at write time, except the frames a video encoder drops, which `reduction` accounts for.
4. **Provenance keys always exist.** `calibration`, `baseline` and `adapter` are present in every session record even when a value is `null`, so their absence is a visible fact rather than a missing key.
5. **Raw is always recoverable.** Any transformation applied at record time (baseline subtraction, rate reduction) is recorded with enough information to undo it or to know exactly what was lost.
6. **Health is an annotation, not a filter.** Quality scores are written alongside the data, per episode, and never silently drop frames or episodes.

## Feature keys

```
observation.tactile.<instance_id>
```

`instance_id` is a short slug chosen by the recorder and stable across sessions: `left_fingertip`, `right_fingertip`, `palm_0`, `thumb`. Feature names may contain dots but not `/`.

For visuotactile sensors the image stream is stored the way camera streams are (video-encoded), under the same key. The image is the ground-truth stream. Optional streams derived from it use a suffix:

```
observation.tactile.<instance_id>.markers
observation.tactile.<instance_id>.displacement
observation.tactile.<instance_id>.force
```

`.displacement` and `.force` are allowed only when the instance's `info` carries a `derived` block naming the method that produced them. In 0.6.1 these sibling keys and a single `(N, 6)` key behave the same on every check (loading, stats, metadata, `delta_timestamps`, file size); siblings are used here because they keep the image as the instance's main stream.

A suffixed key is its own feature entry. Its `info` carries only `reduction`, when its main-table column is a dataset-rate copy of a faster native stream, and `session_meta`, pointing at the instance's sidecar. Everything else (`family`, `derived`, `mount`, `device`) is read from the instance's entry.

## Static metadata: the feature entry's `info`

Each instance's static metadata goes in its feature entry in `meta/info.json`, under `info`, the slot LeRobot core already uses for its own per-feature metadata (video and depth parameters). LeRobot 0.6.1 keeps extra keys there through `finalize`, reload, `resume` and the edit tools, and uploads them unchanged with `push_to_hub` (checked on array features; video features untested, see question 6). It drops unknown top-level keys in `info.json`, so a top-level `tactile_sensors` block does not work.

Only facts that hold for every session go here. `lerobot-edit-dataset merge` requires non-video feature entries to be identical, `info` included, so anything that varies between sessions in `info` would make two sessions impossible to merge.

```json
"observation.tactile.left_fingertip": {
  "dtype": "float32",
  "shape": [52, 3],
  "info": {
    "family": "taxel_array",
    "vendor": "PaXini",
    "model": "PX-6AX-GEN3 DP-S2015-Elite",
    "declared_rate_hz": 91,
    "units": {"force": "N"},
    "channels": ["fx", "fy", "fz"],
    "layout": {
      "frame": "sensor",
      "positions_mm": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], "..."],
      "normals": [[0, 0, 1], [0, 0, 1], "..."]
    },
    "reduction": {"method": "mean", "window_ms": 33.3},
    "mount": {"body": "so101_follower", "link": "gripper_left_finger", "pose": null},
    "device": {"id": "paxini_hub_0", "clock": "paxini_hub_0"},
    "session_meta": "tactile/left_fingertip/meta.json"
  }
},

"observation.tactile.right_fingertip": {
  "dtype": "video",
  "shape": [240, 320, 3],
  "info": {
    "video.fps": 30,
    "...": "core's own video.* keys",
    "family": "visuotactile",
    "vendor": "GelSight",
    "model": "DIGIT",
    "declared_rate_hz": 60,
    "reduction": {"method": "nearest", "window_ms": 33.3},
    "markers": {"count": 63, "grid": [7, 9]},
    "derived": {"method": "marker-flow-depth", "version": "0.1.0"},
    "mount": {"body": "so101_follower", "link": "gripper_right_finger", "pose": null},
    "device": null,
    "session_meta": "tactile/right_fingertip/meta.json"
  }
},

"observation.tactile.palm_0": {
  "dtype": "float32",
  "shape": [5, 3],
  "info": {
    "family": "magnetic",
    "vendor": "NYU / AnySkin",
    "model": "AnySkin v1",
    "declared_rate_hz": 200,
    "units": {"field": "uT"},
    "channels": ["bx", "by", "bz"],
    "layout": {
      "frame": "sensor",
      "positions_mm": [[0, 0, 0], [6, 0, 0], [-6, 0, 0], [0, 6, 0], [0, -6, 0]],
      "normals": null
    },
    "reduction": {"method": "mean", "window_ms": 33.3},
    "mount": {"body": "so101_follower", "link": "palm", "pose": null},
    "device": null,
    "session_meta": "tactile/palm_0/meta.json"
  }
}
```

Core keys (`dtype`, `shape`, `names`) follow LeRobot's rules; only `info` is added to. Field notes:

- `family` is one of `taxel_array`, `visuotactile`, `magnetic`, `force_torque`. Adding a family is a schema change; adding a vendor is not.
- `channels` is required for `taxel_array`, `magnetic` and `force_torque`, and fixes the last dimension: per-frame shape is `(n_elements, len(channels))`. A force-torque sensor is one element, `(1, 6)`, so every array family shares this shape and the key stays its own feature instead of folding into `observation.state` as a 1-D array would. Order is the declared order, so `["fx", "fy", "fz", "dx", "dy", "dz"]` is as valid as `["fx", "fy", "fz"]`.
- `layout.positions_mm` is the physical position of each element in the sensor's own frame. For taxel arrays and magnetic sensors its length equals `n_elements`. For visuotactile sensors the image itself is the layout; `markers.grid` gives the marker arrangement.
- `reduction` is present only when the main-table column is a copy at the dataset fps of a faster native stream, or, for a video feature, when the sensor runs faster than the dataset fps. `method` is one of `nearest`, `mean`, `last`; `window_ms` is the span of native samples each dataset frame was reduced from.
- `derived` is required when `.displacement` or `.force` streams are present, and names the algorithm (`method`) and its `version`. Point count and order are defined by that method.
- `mount` records where on the robot the sensor sits. `pose` is a 4×4 or (xyz, quat) in the link frame if known, else `null`. Moving a sensor to another link makes it a different instance.
- `device` groups instances that share one physical connection, such as several fingertips on one serial hub. `id` is a slug unique in the dataset; `clock` names the timebase, and instances with the same `clock` have directly comparable `device_ts`. `device` is `null` for a sensor with its own connection. The grouping is structural, so it lives here: five keys with the same `device.id` are one hand. The hub's serial and firmware can change and live in the session record.
- `session_meta` points to the instance's sidecar `meta.json`, relative to the dataset root.
- On video features, the tactile keys sit beside core's own `video.*` keys and never use that prefix.
- For visuotactile sensors the image stream is stored at the dataset fps. When the sensor runs faster, `info.reduction` records `{"method": "nearest", "window_ms": …}` and the native-rate requirement applies to the instance's array streams (`.markers`, `.displacement`, `.force`), when present, not to the image; storing raw frames outside the video encoder is out of scope for 0.2.

## Per-session metadata: `tactile/<instance_id>/meta.json`

One file per instance, holding one record per recording session and the episodes it covers. LeRobot's own `merge` does not see this file; a merge tool that carries the sidecar concatenates the `sessions` lists and offsets their episode ranges.

```json
{
  "instance_id": "left_fingertip",
  "sessions": [
    {
      "session_id": "2026-11-21a",
      "episodes": [0, 41],
      "serial": "PX3-00412",
      "firmware": null,
      "device": {"serial": "PXH-0193", "firmware": "1.4.2"},
      "adapter": {"name": "tactile-lerobot.paxini", "version": "0.1.0"},
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
      "gel": null,
      "skin": null
    }
  ]
}
```

Field notes:

- `episodes` is an inclusive range of episode indices recorded in that session.
- `device` holds the hub's `serial` and `firmware` for that session, and is `null` when the instance's `info.device` is `null`. Per-fingertip identity (serial, gel or skin, calibration) stays with the instance, so replacing one fingertip does not change the identity of the others.
- `adapter` names the code that turned the sensor's output into the stored values, and its version. A bug found in an adapter can then be traced to the datasets it affected.
- `calibration.method` is free text for now (`vendor`, `3dcal`, `nano17-press`, `uncalibrated-by-design`). A controlled vocabulary can come later once there are a few real values.
- `baseline` is what the sensor read with nothing touching it at session start. For taxel and magnetic families it is stored as `values`; for visuotactile it is a `reference_frame` pointer (`"episode_000000/frame_0000"`). `applied` says whether the values were subtracted at record time. When `applied` is `true`, stored data plus `baseline.values` must reconstruct the raw readings exactly. `applied: true` is valid only where the baseline is stored as values.
- `gel` (`{"id", "installed_at", "cycles_at_install"}`) and `skin` (`{"id", "installed_at"}`) are the replaceable-part identifiers. They are what make "this sensor degraded 30% faster than that one" answerable later.

## Per-frame data

| Family | dtype | Shape per frame | Notes |
| --- | --- | --- | --- |
| `taxel_array` | float32 | `(n_elements, len(channels))` | Never uint8. The PaXini resultant mode caps Fz at 25.5 N in uint8; store distributed mode, or scale correctly, or both |
| `visuotactile` | uint8 | `(H, W, 3)` video-encoded | Encoded like camera streams |
| `visuotactile` `.markers` | float32 | `(n_markers, 2)` | Optional; marker pixel coordinates |
| `visuotactile` `.displacement` | float32 | `(N, 3)` | Optional; dx, dy, dz; requires `derived` |
| `visuotactile` `.force` | float32 | `(N, 3)` | Optional; fx, fy, fz; requires `derived` |
| `magnetic` | float32 | `(n_sensors, len(channels))` | Raw field. Baseline-subtracted view is computed at read time, not stored |
| `force_torque` | float32 | `(1, 6)` | `channels` `["fx", "fy", "fz", "mx", "my", "mz"]` in the sensor frame; one element, so the key stays standalone |

Parquet stores the array features as Hugging Face `Array2D` columns, readable with plain pyarrow or pandas, without LeRobot.

**Normalization stats are per channel, not per element.** LeRobot computes `stats.json` over the last dimension, so a `(52, 3)` feature gets stats of shape `(3,)` and a policy normalizer scales every taxel by the same three numbers. Per-taxel baselines and drift are computed from the data by `tactile doctor`, never read from `stats.json`. For the same reason, main-table layouts that pad a fixed-size block of samples per row are avoided: NaN padding makes the stats NaN, and zero padding biases them.

## Native-rate streams: `tactile/<instance_id>/episode-{index:06d}.parquet`

When a sensor's rate equals the dataset fps, its main-table column is the native stream and nothing else is needed.

When the rates differ, the native-rate stream is required (except the visuotactile image itself, see the `reduction` note under Static metadata) and lives beside the dataset, one file per instance per episode:

```
<dataset root>/
├── meta/  data/  videos/          # LeRobot's own layout, unchanged
└── tactile/
    ├── left_fingertip/
    │   ├── meta.json
    │   ├── episode-000000.parquet
    │   └── episode-000001.parquet
    └── right_fingertip/
        ├── meta.json
        └── markers/               # a suffixed stream keeps its own files
            └── episode-000000.parquet
```

| Column | dtype | Meaning |
| --- | --- | --- |
| `episode_index` | int64 | Episode this sample belongs to |
| `sample_index` | int64 | Sample number within the episode, from 0 |
| `host_ts` | float64 | Host receive time, seconds, monotonic clock |
| `device_ts` | float64 or null | Timestamp from the sensor's own clock, seconds, if the device provides one |
| `seq` | int64 or null | Device sequence number if provided; gaps mean dropped frames at the source |
| `data` | float32 | One sample, same shape as the feature's per-frame shape |

A suffixed stream keeps its own files: `tactile/<instance_id>/<suffix>/episode-{index:06d}.parquet` (for example `tactile/right_fingertip/markers/`), with the same columns.

A copy at the dataset fps in the main table is optional and recommended, since most training code reads one table. When present, the feature's `info.reduction` says how it was made. A reader that needs the timing aligns from the sidecar by nearest `host_ts` or by a window; the alignment policy lives in the loader, not the file.

A dataset-rate copy cannot show dropped frames or clock skew. Those are only visible in the native stream, which is why it is never optional when the rates differ. A visuotactile image at the dataset fps carries only what the video shows; its suffixed streams, when present, carry the native timing.

What core does and doesn't do with sidecars in 0.6.1: a full load ignores them and works normally, and `push_to_hub` uploads them with the rest of the folder. `lerobot-edit-dataset` copies only `meta/`, `data/` and `videos/`, and a partial load (`episodes=[…]`) downloads only the listed files, so both leave the sidecar behind. Tools in this project carry it; core tools do not yet.

## Writing tactile data

The public dataset API (`LeRobotDataset.create` → `add_frame` → `save_episode` → `finalize`) writes and reloads these features in LeRobot 0.6.1 with no core change, and `delta_timestamps` works on tactile keys.

Stock `lerobot-record` does not. A robot that returns a `(52, 3)` array from `get_observation()` is rejected by `hw_to_dataset_features`, which treats any non-1-D shape as a camera and expects `(H, W, C)`, and `build_dataset_frame` passes through only 1-D float arrays and images. Folding the taxels into `observation.state` or declaring a `(52, 3, 1)` "depth camera" both load, and both lose what the data is. Plugin discovery offers no other route.

This project therefore records through the public API with its own `tactile-record` command, built on `make_robot_from_config`, `build_dataset_frame` and `LeRobotDataset.add_frame`. A small upstream change to `lerobot/utils/feature_utils.py` (accept an N-D numeric spec in `hw_to_dataset_features`, pass N-D arrays through in `build_dataset_frame`), open as [huggingface/lerobot#5149](https://github.com/huggingface/lerobot/pull/5149), would let `lerobot-record` do the same once merged.

## Health annotations

`tactile doctor` writes a per-episode report to `tactile/health.json`, beside the data and never inside `info.json`:

```json
{
  "episode_000012": {
    "left_fingertip": {
      "drift": {"offset_pct_range": 0.031, "flag": false},
      "framerate": {"effective_hz": 89.4, "dropped_pct": 0.018, "max_gap_ms": 31.0, "flag": false},
      "saturation": {"saturated_pct": 0.0, "flag": false},
      "skew": {"lag_ms": -12.0, "confidence": 0.81, "flag": false}
    },
    "right_fingertip": {
      "gel": {"dead_pct": 0.062, "largest_cluster_px": 41, "flag": true},
      "framerate": {"effective_hz": 29.1, "dropped_pct": 0.03, "max_gap_ms": 220.0, "flag": true},
      "skew": {"lag_ms": 35.0, "confidence": 0.67, "flag": false}
    },
    "doctor_version": "0.1.0",
    "thresholds": "default"
  }
}
```

Thresholds are conservative defaults and are recorded with the result so a later run with different thresholds is distinguishable. For a video feature with `reduction`, `framerate` is measured against the dataset fps; the native rate is checked from the suffixed streams when present.

## Interoperability

- **PR #4469.** The `TactileSensor` base class in this project follows that PR's method surface (`connect`, `read`, `async_read`, `read_latest`, `disconnect`) and config registry, so a merge or dependency is possible. Type mapping: `taxel_array` → `force`, `force_torque` → `wrench`; the PR's `displacement` and `force` point sets correspond to the optional `.displacement` and `.force` streams here, and its 6-D `full` layout to a single key with six declared `channels`. `magnetic` and visuotactile images have no PR type. The PR's `read()` returns no timestamps or per-sensor metadata and the PR defines no dataset layer; those are what this schema adds. A recording made with the PR's record-time tare is compliant if the tare values are stored with `baseline.applied: true`.
- **TLabel.** TLabel is an annotation layer: per-frame labels (contact, slip, force magnitude, manipulation phase and others) computed from raw data. It sits above this schema, not beside it. Overlap is in the sensor description: TLabel's `manufacturer`, `model`, `sensor_type`, `frame_rate` and `provenance` (firmware, calibration date, sampling rate) map to `vendor`, `model`, `family`, `declared_rate_hz`, `firmware` and `calibration.date`. A converter would feed these recordings to TLabel's adapters, one instance at a time, and could store TLabel labels back alongside the raw data per instance.
- **MCAP.** One channel per instance, schema per family, `host_ts` as the log time, `device_ts` as publish time where present, read from the native-rate sidecar where one exists. Camera and joint channels exported alongside so Foxglove opens the whole episode.

## Open questions (please answer any of these)

0.1's questions on per-instance keys vs a ragged dimension, multi-rate storage, and TLabel field overlap are answered in 0.2; see Feature keys, Native-rate streams, and Interoperability.

1. Per-instance keys: does any training code need a stacked per-hub view (five fingertips as one array), and if so, should the loader build it from the `device` grouping rather than storing it?
2. Static vs per-session: which facts change between your recording sessions? A fact placed in `info` that varies makes `merge` fail; one placed in the sidecar that never varies costs nothing.
3. Sidecars: core edit tools and partial loads leave `tactile/` behind. Would you accept that until core knows about it, or would you rather keep native-rate data in the main table despite the padding cost to stats?
4. Normalization: is per-channel normalization acceptable for tactile policies, or do you normalize per taxel, and where?
5. Is `layout.positions_mm` the right level of geometry, or do people need full per-element frames (position + orientation)? Does anyone actually use this in a policy today?
6. For visuotactile sensors, is a reference frame pointer enough for `baseline`, or should a small reference image be stored directly? And does core's own write of video `info` keep the extra keys on a video feature? (Checked for parquet features, not yet for video.)
7. What does your pipeline do with the no-contact baseline today? Subtract at record time, at load time, or never?
8. What calibration methods are in use that should be in a controlled vocabulary?
9. For derived point-set streams (`.displacement`, `.force`), should point positions be stored, and where: per frame, or once in `info`?
10. What is missing that would make you refuse to use this?

## Changelog

- 0.2 (2026-10-29):
  - Timing: a native-rate stream with device and host timestamps is required whenever the sensor rate differs from the dataset fps; an optional dataset-rate copy in the main table must declare its `reduction` (`method`, `window_ms`).
  - Baseline: `baseline.applied` added. Record-time subtraction is allowed if the baseline values are stored; stored data plus baseline must reconstruct raw.
  - `device` grouping for instances that share a hub: `id` and `clock` in `info`, the hub's `serial` and `firmware` per session; each fingertip remains its own instance.
  - Torque channels renamed `tx/ty/tz` → `mx/my/mz`, matching PR #4469 and TLabel.
  - Optional derived visuotactile streams `.displacement` and `.force`, float32 `(N, 3)`, allowed only with a `derived` block (`method`, `version`).
  - `adapter` block (`name`, `version`) added as a provenance key.
  - New principle 5, "raw is always recoverable".
  - Static metadata moved from a top-level `tactile_sensors` block, which LeRobot 0.6.1 drops, to each feature entry's `info`.
  - Static and per-session metadata split: serial, firmware, the hub's serial and firmware, `adapter`, `calibration`, `baseline`, `gel` and `skin` move to `tactile/<instance_id>/meta.json`, since `merge` requires identical feature entries.
  - `channels` required for `taxel_array`, `magnetic` and `force_torque`; per-frame shape is `(n_elements, len(channels))`, with `force_torque` as `(1, 6)`. Stats are per channel; per-taxel statistics are computed by `tactile doctor`.
  - Native-rate sidecar layout defined: `tactile/<instance_id>/episode-{index:06d}.parquet` with `episode_index`, `sample_index`, `host_ts`, `device_ts`, `seq`, `data`. Health reports move to `tactile/health.json`.
  - Record path: the public dataset API writes tactile features with no core change; stock `lerobot-record` does not, so recording goes through `tactile-record`, with the upstream change opened as [huggingface/lerobot#5149](https://github.com/huggingface/lerobot/pull/5149).
  - Sibling `.displacement` / `.force` keys checked against 0.6.1: they load, get stats and carry `info` exactly as a single key does, so they stay.
  - Interoperability rewritten against PR #4469's code and TLabel's spec. Open questions revised.
- 0.1 (Oct 2026): first draft for feedback.
