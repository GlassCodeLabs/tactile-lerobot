# Tactile data schema for LeRobot datasets — draft 0.3.1

*Status: draft for feedback, November 2026. Nothing here is final. The goal is to converge with [LeRobot PR #4469](https://github.com/huggingface/lerobot/pull/4469), [TLabel](https://github.com/liesliy/tlabel) and [Multi-Frequency-LeRobot](https://github.com/RoboCoin-BAAI/Multi-Frequency-LeRobot), not to add a fourth incompatible layout. 0.3 applies the answers given on #4469 ([comment](https://github.com/huggingface/lerobot/pull/4469#issuecomment-6555461272)) to 0.2's open questions. The questions still open are listed at the end; those are the ones I'd most like answered.*

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

A suffixed key is its own feature entry. Its `info` carries only `reduction`, when its main-table column is a dataset-rate copy of a faster native stream, and `session_meta`, pointing at the instance's sidecar. Everything else (`family`, `derived`, `layout`, `mount`, `device`) is read from the instance's entry.

## Static metadata: the feature entry's `info`

Each instance's static metadata goes in its feature entry in `meta/info.json`, under `info`, the slot LeRobot core already uses for its own per-feature metadata (video and depth parameters). LeRobot 0.6.1 keeps extra keys there through `finalize`, reload, `resume` and the edit tools, and uploads them unchanged with `push_to_hub` (checked on array features, and for video features core merges its probed `video.*` keys into the existing `info` rather than replacing it). It drops unknown top-level keys in `info.json`, so a top-level `tactile_sensors` block does not work.

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
      "origin": {"at": "corner", "description": "taxel 0, the corner of the array nearest the connector"},
      "axes": {
        "x": "along the long side of the array, away from the connector",
        "y": "across the array, from taxel 0 toward taxel 8",
        "z": "out of the sensing surface, toward the contacting object"
      },
      "surface": "pad",
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
    "layout": {
      "origin": {"at": "corner", "description": "the top-left pixel of the image"},
      "axes": {
        "x": "along an image row, toward the right",
        "y": "down an image column",
        "z": "out of the gel, toward the contacting object"
      },
      "surface": "gel"
    },
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
      "origin": {"at": "element", "description": "the central magnetometer"},
      "axes": {
        "x": "across the board, from the central magnetometer toward the second",
        "y": "across the board, from the central magnetometer toward the fourth",
        "z": "out of the skin, toward the contacting object"
      },
      "surface": "die",
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
- `range` (0.3.1) is optional and gives the rated range of the readings in `units`: `{"min": …, "max": …}`, each a number for the whole instance or a list with one value per channel, in channel order. A reading at or beyond a limit is saturated. It is defined for the array families; absent means unknown, and a health check then falls back to the observed extremes and says so.
- `layout` is required for every family and declares the sensor's own coordinate frame, because two sensors can both publish `positions_mm` and be mutually uninterpretable if one measures from a top-left origin and the other from a bottom-left one, or one from the gel surface and the other from the sensor die. `origin.at` says where (0, 0, 0) sits on the sensor body: `center` or `corner` of the sensing area, `element` for a sensing element, or `mount` for the mounting face; `origin.description` says which corner or element in words. `axes.x`, `axes.y` and `axes.z` say in words where +x, +y and +z point, in the sensor's own terms. The frame is right-handed, and +z out of the sensing surface toward the contacting object is the convention; an adapter that keeps a vendor frame with another convention says so in `axes.z`. `surface` says which surface the frame and the positions describe: `gel` (the outer face of a transparent elastomer), `die` (the plane of the sensing elements, under whatever covers them), `skin` (the outer face of a replaceable skin), `pad` (the outer face of a fixed contact pad over the elements), or `vendor` (positions copied from the vendor's documentation, whose own frame definition applies; the document is named in the session record's `calibration.reference`). Any other value is an error.
- `layout.positions_mm` is the physical position of each element in that frame, in millimetres, one `[x, y, z]` row per element. It is required for `taxel_array` and `magnetic`, where its length equals `n_elements`; optional for `force_torque`, one row if given; and omitted for `visuotactile`, where the image is the element arrangement and `markers.grid` the marker arrangement, while `layout` still declares the frame the `.displacement` and `.force` streams are expressed in and the surface the image shows. `layout.normals` is optional: one unit vector per position.
- `reduction` is present only when the main-table column is a copy at the dataset fps of a faster native stream, or, for a video feature, when the sensor runs faster than the dataset fps. `method` is one of `nearest`, `mean`, `last`; `window_ms` is the span of native samples each dataset frame was reduced from.
- `derived` is required when `.displacement` or `.force` streams are present, and names the algorithm (`method`) and its `version`. Point count and order are defined by that method.
- `mount` records where on the robot the sensor sits. `pose` is a 4×4 or (xyz, quat) in the link frame if known, else `null`. Moving a sensor to another link makes it a different instance.
- `device` groups instances that share one physical connection, such as several fingertips on one serial hub. `id` is a slug unique in the dataset; `clock` names the timebase, and instances with the same `clock` have directly comparable `device_ts`. `device` is `null` for a sensor with its own connection. The grouping is structural, so it lives here: five keys with the same `device.id` are one hand. The hub's serial and firmware can change and live in the session record.
- `session_meta` points to the instance's sidecar `meta.json`, relative to the dataset root.
- On video features, the tactile keys sit beside core's own `video.*` keys and never use that prefix.
- For visuotactile sensors the image stream is stored at the dataset fps. When the sensor runs faster, `info.reduction` records `{"method": "nearest", "window_ms": …}` and the native-rate requirement applies to the instance's array streams (`.markers`, `.displacement`, `.force`), when present, not to the image; storing raw frames outside the video encoder is out of scope for 0.3.

## Per-session metadata: `tactile/<instance_id>/meta.json`

One file per instance, holding one record per recording session and the episodes it covers. LeRobot's own `merge` does not see this file; a merge tool that carries the sidecar concatenates the `sessions` lists and offsets their episode ranges.

```json
{
  "instance_id": "left_fingertip",
  "sessions": [
    {
      "session_id": "2026-11-21a",
      "episodes": [0, 41],
      "clock": {
        "wall_time": "2026-11-21T14:03:05.117Z",
        "host_clock": 10234.117,
        "episode_origins": [10240.004, 10301.772, "..."]
      },
      "frames": {"received": 30412, "dropped": 7, "overflowed": 0},
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
        "frame_range": [364, 545],
        "values": [[0.01, -0.02, 0.00], "..."]
      },
      "baseline_stream": {"sample_every": 100},
      "gel": null,
      "skin": null
    }
  ]
}
```

Field notes:

- `episodes` is an inclusive range of episode indices recorded in that session.
- `clock` anchors the session's host clock. Sidecar `host_ts` values come from a monotonic clock with no epoch, and LeRobot's own `timestamp` column is `frame_index / fps`, restarting at zero in every episode. `wall_time` is the UTC time when the session starts, right after `connect()`, and `host_clock` the host clock value at the same instant, so any `host_ts` maps to the calendar as `wall_time + (host_ts − host_clock)`. `episode_origins` holds one value per episode in `episodes`, in order: the host clock value when the episode's first dataset frame was built. Dataset frame `k` of that episode is at `origin + k / fps` on the host clock, which is also the target time its dataset-rate copy was reduced at.
- `frames` holds the sensor's frame counters since `connect()`, so per session: `received` by the host, `dropped` at the source (gaps in `seq`), `overflowed` on the host because the recorder did not drain the sensor in time. They come from the sensor's counters, not from the episode files, and they count every frame since `connect()`, including the baseline capture and the pauses between episodes, which the files do not hold. So `received` is at least the number of native rows in the session's episode files and, with `seq` present, `dropped` is at least the sum of the `seq` gaps inside them.
- `device` holds the hub's `serial` and `firmware` for that session, and is `null` when the instance's `info.device` is `null`. Per-fingertip identity (serial, gel or skin, calibration) stays with the instance, so replacing one fingertip does not change the identity of the others.
- `adapter` names the code that turned the sensor's output into the stored values, and its version. A bug found in an adapter can then be traced to the datasets it affected.
- `calibration.method` is free text for now (`vendor`, `3dcal`, `nano17-press`, `uncalibrated-by-design`). A controlled vocabulary can come later once there are a few real values. `calibration.reference` names the document the calibration or, for `layout.surface: "vendor"`, the element positions were taken from.
- `baseline` is what the sensor read with nothing touching it at session start. For taxel and magnetic families it is stored as `values`; for visuotactile it is a `reference_frame` pointer (`"episode_000000/frame_0000"`). `frame_range` is `[first_seq, last_seq]` of the native frames averaged into `values`, `null` when the sensor has no `seq`; it lets a consumer audit what the baseline absorbed, since a capture that starts with the sensor already loaded absorbs the load and offsets every later reading low. `applied` says whether the values were subtracted at record time. When `applied` is `true`, stored data plus the baseline in force must reconstruct the raw readings to within float32 rounding; the baseline in force is read from the baseline stream, since a backend may re-baseline after this capture, and `baseline` here is only the capture at session start. `applied: true` is valid only where the baseline is stored as values.
- `baseline_stream` declares the sampling period of the instance's baseline stream (see Baseline stream): one raw sample every `sample_every` native samples. It is `null` when the session wrote no baseline stream.
- `gel` (`{"id", "installed_at", "cycles_at_install"}`) and `skin` (`{"id", "installed_at"}`) are the replaceable-part identifiers. They are what make "this sensor degraded 30% faster than that one" answerable later.

## Per-frame data

| Family | dtype | Shape per frame | Notes |
| --- | --- | --- | --- |
| `taxel_array` | float32 | `(n_elements, len(channels))` | Never uint8. The PaXini resultant mode caps Fz at 25.5 N in uint8; store distributed mode, or scale correctly, or both |
| `visuotactile` | uint8 | `(H, W, 3)` video-encoded | Encoded like camera streams |
| `visuotactile` `.markers` | float32 | `(n_markers, 2)` | Optional; marker pixel coordinates |
| `visuotactile` `.displacement` | float32 | `(N, 3)` | Optional; dx, dy, dz; requires `derived` |
| `visuotactile` `.force` | float32 | `(N, 3)` | Optional; fx, fy, fz; requires `derived` |
| `magnetic` | float32 | `(n_sensors, len(channels))` | Raw field unless `baseline.applied`; the no-contact baseline lives in the session record and the baseline stream |
| `force_torque` | float32 | `(1, 6)` | `channels` `["fx", "fy", "fz", "mx", "my", "mz"]` in the sensor frame; one element, so the key stays standalone |

Parquet stores the array features as Hugging Face `Array2D` columns, readable with plain pyarrow or pandas, without LeRobot.

**The format stores physical units and normalizes nothing.** `units` names the unit of each quantity, and no scaling is applied at record time: different policies want different statistics (unit variance, the sensor's rated range, none at all), and the format is not the place to choose. Normalization is a load-time step, per channel, derived from LeRobot's `stats.json`, which is per channel already: LeRobot computes stats over the last dimension, so a `(52, 3)` feature gets stats of shape `(3,)` and a policy normalizer scales every taxel by the same three numbers. That keeps the spatial gradient across taxels that distinguishes a flat press from a corner contact, which per-taxel normalization would erase, and it stays reversible. Per-taxel baselines and drift are computed from the data by `taxel doctor`, never read from `stats.json`. For the same reason, main-table layouts that pad a fixed-size block of samples per row are avoided: NaN padding makes the stats NaN, and zero padding biases them.

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
    │   ├── episode-000001.parquet
    │   └── baseline/              # the baseline stream, one file per episode
    │       ├── episode-000000.parquet
    │       └── episode-000001.parquet
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

`data` is stored as an Arrow fixed-shape tensor column (the `fixed_shape_tensor` canonical extension type), which pyarrow reads back as one `(n_samples, …)` array; the other columns are plain.

A suffixed stream keeps its own files: `tactile/<instance_id>/<suffix>/episode-{index:06d}.parquet` (for example `tactile/right_fingertip/markers/`), with the same columns. The directory name `baseline` under an instance is reserved for the baseline stream and is not a stream suffix.

A copy at the dataset fps in the main table is optional and recommended, since most training code reads one table. When present, the feature's `info.reduction` says how it was made. A reader that needs the timing aligns from the sidecar by nearest `host_ts` or by a window; the alignment policy lives in the loader, not the file. The host time of dataset frame `k` of an episode is the episode's origin from the session record's `clock` plus `k / fps`.

A dataset-rate copy cannot show dropped frames or clock skew. Those are only visible in the native stream, which is why it is never optional when the rates differ. A visuotactile image at the dataset fps carries only what the video shows; its suffixed streams, when present, carry the native timing.

What core does and doesn't do with sidecars in 0.6.1: a full load ignores them and works normally, and `push_to_hub` uploads them with the rest of the folder. `lerobot-edit-dataset` copies only `meta/`, `data/` and `videos/`, and a partial load (`episodes=[…]`) downloads only the listed files, so both leave the sidecar behind. Tools in this project carry it; core tools do not yet. `lerobot-edit-dataset merge` also fails on any dataset with a 2-D feature in 0.6.x, tactile or not, because it rewrites the data parquet through pandas and pyarrow cannot convert a 2-D array column from a DataFrame; merging a tactile dataset therefore needs a tool in this project.

## Baseline stream: `tactile/<instance_id>/baseline/episode-{index:06d}.parquet`

A baseline captured once at session start and subtracted at record time loses whatever the sensor drifts to afterwards (thermal, gel creep, hysteresis), and a backend that re-baselines internally loses the drift for good: the stored frames return to zero while the sensor has not. The baseline stream keeps both recoverable. It is defined for the array families (a visuotactile instance's baseline is its `reference_frame` and has no stream in 0.3), written for every array-family instance whether or not a baseline is applied, one file per instance per episode, with two kinds of row:

| Column | dtype | `capture` row | `sample` row |
| --- | --- | --- | --- |
| `episode_index` | int64 | the episode | the episode |
| `sample_index` | int64 | first native sample of the episode the baseline applies to | the sampled native sample |
| `kind` | string | `capture` | `sample` |
| `seq` | int64 or null | that sample's `seq` | that sample's `seq` |
| `host_ts` | float64 | host clock when the baseline took effect | the sample's receive time |
| `frame_range` | 2 × int64 or null | `[first_seq, last_seq]` of the raw frames averaged, `null` without `seq` | `null` |
| `values` | float32 | the baseline, in raw terms | the unsubtracted sample |

`values` is a fixed-shape tensor column of the feature's per-frame shape, as `data` is in the native stream.

- A **capture row** records a baseline that was subtracted from the stored frames from `sample_index` on. Every file opens with a capture row at `sample_index` 0 for the baseline in force at the episode's first frame, repeated from where it was first captured (its `host_ts` may precede the episode), so each episode reconstructs on its own. A baseline that takes effect during the episode, whether a backend re-baselined or the recorder did, adds a capture row at the first sample it applies to. Clearing a baseline is a capture row of zeros with `frame_range` `null`.
- A **sample row** is the unsubtracted frame at every native sample whose `sample_index` is a multiple of the session record's `baseline_stream.sample_every`, so sample 0 is always sampled and every episode opens with a raw reading. One rule covers sensors with and without `seq`. The default period is 100: one raw sample per second at 100 Hz, one percent of the native stream.

**Reconstruction (principle 5).** The raw value of native sample `i` is its stored `data` plus the `values` of the latest capture row with `sample_index <= i`; when there is no such row, the stored data is raw. The sum is float32, so it matches the sensor's reading to within one unit in the last place; a sample row's `values` are computed the same way and must equal the reconstruction at its index exactly, which is the integrity check a validator runs. When nothing was applied, the file has no capture rows and its samples equal the stored frames.

**Why both kinds.** Capture rows are what reconstruction needs, exact to the sample. Sample rows are ground truth that does not depend on the subtractor reporting honestly: a firmware that re-zeroes without telling the host, a filter that tracks slowly, or an adapter bug all show up as a sample that disagrees with stored data plus the capture in force, which a validator reports as an integrity failure. Drift is read from the samples: with nothing touching the sensor, a sample minus the baseline in force should be zero, and its growth over an episode is the sensor drifting, visible even when the stored frames sit at zero because a backend re-baselined. That is what lets a health check tell "sensor drifted" from "policy is pressing".

A backend may subtract only what it can report. A device-side tare whose values the host cannot read back is not allowed under principle 5; the backend subtracts on the host instead, or declares the device's values, so that every stored frame carries the baseline that was taken from it ([docs/sensors.md](docs/sensors.md)).

## Writing tactile data

The public dataset API (`LeRobotDataset.create` → `add_frame` → `save_episode` → `finalize`) writes and reloads these features in LeRobot 0.6.1 with no core change, and `delta_timestamps` works on tactile keys.

Stock `lerobot-record` does not. A robot that returns a `(52, 3)` array from `get_observation()` is rejected by `hw_to_dataset_features`, which treats any non-1-D shape as a camera and expects `(H, W, C)`, and `build_dataset_frame` passes through only 1-D float arrays and images. Folding the taxels into `observation.state` or declaring a `(52, 3, 1)` "depth camera" both load, and both lose what the data is. Plugin discovery offers no other route.

This project therefore records through the public API with its own `taxel record` command ([docs/recording.md](docs/recording.md)), built on `make_robot_from_config`, `build_dataset_frame` and `LeRobotDataset.add_frame`. A small upstream change to `lerobot/utils/feature_utils.py` (accept an N-D numeric spec in `hw_to_dataset_features`, pass N-D arrays through in `build_dataset_frame`), open as [huggingface/lerobot#5149](https://github.com/huggingface/lerobot/pull/5149), would let `lerobot-record` do the same once merged.

## Health annotations

`taxel doctor` writes one report per dataset to `tactile/health.json`, beside the data and never inside `info.json`. It carries the doctor version and the thresholds the run used, so two runs are distinguishable; what the run found out about each instance before checking any episode; and, per episode and per instance, the integrity check and the five health checks, each with its numbers and a `flag`:

```json
{
  "doctor_version": "0.1.0",
  "generated_at": "2026-11-05T14:03:05Z",
  "thresholds": {"drift": 0.05, "dropped": 0.02, "max_gap_ms": null, "gel": 0.05,
                 "cluster_px": 25, "skew": 50.0, "confidence": 0.5, "saturated": 0.01, "integrity": true},
  "instances": {
    "left_fingertip": {"key": "observation.tactile.left_fingertip", "family": "taxel_array", "metadata": "info",
                       "sidecar": true, "declared_rate_hz": 91, "range": null,
                       "unknown_keys": [], "missing_keys": ["meta.sessions[0].baseline.frame_range"]}
  },
  "episodes": {
    "episode_000012": {
      "left_fingertip": {
        "integrity": {"copy": {"frames": 612, "mismatches": 0, "first_mismatch": null, "max_abs_diff": 0.0, "flag": false},
                      "raw": {"samples": 7, "mismatches": 0, "first_mismatch": null, "max_abs_diff": 0.0, "flag": false},
                      "files": {"native": true, "baseline": true, "capture_at_zero": true, "origins": true, "counters": true, "flag": false},
                      "flag": false},
        "drift": {"no_contact_level": [0.011, 0.008, 0.029], "level_change": [0.001, 0.001, 0.002],
                  "offset": [0.012, 0.009, 0.031], "offset_pct_range": null, "slope_per_s": [0.001, 0.0, 0.002],
                  "baseline": "capture", "source": "samples", "baseline_age_s": 3.2, "frame_range": "not recorded",
                  "windows": 20, "no_contact_windows": 14, "candidates": 91, "first_frame_zero": false,
                  "rebaselines": [{"sample_index": 1200, "host_ts": 10301.1, "delta": [0.0, 0.0, 0.03]}],
                  "flag": false},
        "framerate": {"effective_hz": 89.4, "declared_hz": 91, "dropped_pct": 0.018, "max_gap_ms": 31.0,
                      "max_gap_threshold_ms": 109.9,
                      "gap_histogram": {"edges_periods": [1.5, 2.5, 3.5], "counts": [600, 9, 2, 0]},
                      "frames": {"received": 30412, "dropped": 7, "overflowed": 0},
                      "source": "sidecar", "timestamps": "measured", "flag": false},
        "saturation": {"saturated_pct": [0.0, 0.0, 0.0], "limit": "observed", "min": [-0.4, -0.5, -0.1],
                       "max": [3.1, 2.9, 24.8], "confidence": "low", "source": "sidecar", "flag": false},
        "gel": {"skipped": "no visuotactile feature"},
        "skew": {"lag_ms": -12.0, "confidence": 0.81, "reference": "observation.images.front",
                 "onsets": 3, "resolution_ms": 5.0, "flag": false}
      }
    }
  }
}
```

A check that cannot run because its input is missing is `{"skipped": "<reason>"}` and nothing else, so an absent `flag` means "not evaluated", never "passed". The `integrity` check has three parts, `copy`, `raw` and `files`, each run or skipped on its own; on a dataset without a sidecar all three are skipped. `thresholds` records the values used, not a name; `max_gap_ms` `null` means ten declared periods, resolved per instance into the framerate block's `max_gap_threshold_ms`, and `confidence` is the cross-correlation peak a skew lag needs before it can be flagged. `instances` records how each instance's metadata was read (`info`, `none` for a plain LeRobot dataset, `invalid` with an `error`) and the lenient readers' unknown and missing keys. `gel` on an array instance reports `"kind": "elements"` and the zero-variance elements instead of pixel clusters. A flag is an annotation (principle 6); the exit code follows the thresholds the caller names.

For a video feature with `reduction`, `framerate` is measured against the dataset fps; the native rate is checked from the suffixed streams when present. For array families `drift` is read from the baseline stream: the raw samples against the baseline in force, which separates sensor drift from a sustained press, and a capture row inside an episode is reported as a `rebaseline` event rather than scored. On a dataset without a sidecar the checks degrade to what the main table allows and say so in `source`.

## Interoperability

- **PR #4469.** The `TactileSensor` base class in this project follows that PR's method surface (`connect`, `read`, `async_read`, `read_latest`, `disconnect`) and config registry, so a merge or dependency is possible. Type mapping: `taxel_array` → `force`, `force_torque` → `wrench`; the PR's `displacement` and `force` point sets correspond to the optional `.displacement` and `.force` streams here, and its 6-D `full` layout to a single key with six declared `channels`. `magnetic` and visuotactile images have no PR type. The PR's `read()` returns no timestamps or per-sensor metadata and the PR defines no dataset layer; those are what this schema adds. A recording made with the PR's record-time tare is compliant if the tare values are stored with `baseline.applied: true` and the baseline stream carries the capture.
- **TLabel.** TLabel is two things. It is a storage format: a `TLabelData` document, sensor-agnostic and lossless to round-trip, with `sensor_info` (the sensor and the adapter that produced the document), `capabilities` (which of its fourteen per-frame dimensions the producer actually emits) and `episodes` whose per-frame annotations carry `confidence` and a compliance level `L1` to `L4` (the #4469 discussion refers to these as its `provenance` and `data_quality` blocks). Adapters convert vendor formats into that document and exporters convert it back out. On top of the format it is a label schema: `contact`, `contact_centroid`, `slip_event`, `manipulation_phase` and the rest. So TLabel is a sibling format with a bidirectional mapping, not a layer above this schema: a recording in this layout is a source for TLabel documents, and TLabel's dimensions are a specification for what labels stored beside a recording mean. The converter lives in `tlabel` as an adapter, following the rule that conversion belongs to the package that owns the target format ([#4469 comment](https://github.com/huggingface/lerobot/pull/4469#issuecomment-6555461272)). This side of the contract is `taxel record` output, kept stable and listed in [docs/recording.md](docs/recording.md#interop-contract): per-instance `(N, C)` keys with channels declared in `info`, the native-rate sidecar, the baseline stream and `meta.json` for the session. The other side is one TLabel document per instance; back-conversion writes TLabel labels beside the raw data, per instance. Field overlap: `sensor_info.manufacturer`, `model`, `sensor_type`, `frame_rate`, `adapter_name` and `adapter_version` map to `vendor`, `model`, `family`, `declared_rate_hz` and the session record's `adapter`. TLabel models one sensor stream per document; instances that share a hub are grouped here by `device.id` and `device.clock`, which stay stable so documents from one hand can be grouped the same way.
- **MCAP.** One channel per instance, schema per family, `host_ts` as the log time, `device_ts` as publish time where present, read from the native-rate sidecar where one exists. Camera and joint channels exported alongside so Foxglove opens the whole episode.

## Open questions (please answer any of these)

0.1's questions on per-instance keys vs a ragged dimension, multi-rate storage, and TLabel field overlap are answered in 0.2; see Feature keys, Native-rate streams, and Interoperability. Questions 3, 4, 5 and 7 were answered on [#4469](https://github.com/huggingface/lerobot/pull/4469#issuecomment-6555461272) and are resolved in 0.3; they keep their numbers and carry their answers so the numbering stays stable.

1. Per-instance keys: does any training code need a stacked per-hub view (five fingertips as one array), and if so, should the loader build it from the `device` grouping rather than storing it?
2. Static vs per-session: which facts change between your recording sessions? A fact placed in `info` that varies makes `merge` fail; one placed in the sidecar that never varies costs nothing.
3. Sidecars: settled. Core edit tools and partial loads leave `tactile/` behind, and that is accepted until core knows about the directory, because padding the main table to protect native-rate data would cost the stats (see Per-frame data). The mitigation is a validator, `taxel doctor`, that detects a dataset whose sidecar is missing or incomplete and checks that the stored copy and the raw readings are reproducible from it, which is what makes principle 5 checkable rather than a promise.
4. Normalization: resolved. Per channel, at load time, never inside the format; see Per-frame data.
5. Geometry: resolved. `positions_mm` is the right level for storage, and its consumer is spatial alignment and multi-finger fusion rather than the policy, which reads a derived contact centroid. The gap was not orientation but a defined coordinate frame, which `layout` now declares; see Static metadata.
6. For visuotactile sensors, is a reference frame pointer enough for `baseline`, or should a small reference image be stored directly? (Core's own write of video `info` keeps extra keys in 0.6.1: it merges the probed `video.*` values into the existing dict.)
7. Baseline: resolved. Subtraction at record time is allowed provided what was subtracted is stored and raw stays reconstructable: `baseline.frame_range` says which frames a capture absorbed, and the baseline stream records every capture applied and a periodic raw sample, so drift is auditable even after a backend re-baselines; see Baseline stream.
8. What calibration methods are in use that should be in a controlled vocabulary?
9. For derived point-set streams (`.displacement`, `.force`), should point positions be stored, and where: per frame, or once in `info`?
10. Coverage: a float32 array cannot say "not measured", so a channel a sensor does not measure is filled with zero and "measured 0 N" and "not measured" become the same number. `derived.method` names how a stream was produced (a GelSight `.force` reconstructed from displacement looks like a PaXini `.force` measured directly) but not whether a number is a measurement at all, and a consumer should not have to parse a method string to learn that. Two designs: a per-channel `measured: true|false` beside `info.channels`, or a `capabilities` block on the instance mirroring TLabel's. Which would a converter find easier to emit and to check? The choice waits on that answer.

## Changelog

- 0.3.1 (2026-11-05):
  - `range` added to the static metadata, optional, array families only: the rated range of the readings per channel, so a saturation check has a limit to measure against instead of the observed extremes.
  - Health report shape fixed: one document with `doctor_version`, `generated_at`, `thresholds` as the values used, an `instances` block with how each instance's metadata was read and the lenient readers' findings, and `episodes` keyed `episode_{index:06d}`; the `integrity` check with its three parts; the `{"skipped": reason}` form for a check that could not run; `gel` on array instances with `"kind": "elements"`.
  - `frames` field note corrected: the counters count every frame since `connect()`, so `received` is at least the native rows in the session's files and `dropped` at least the `seq` gaps inside them, not equal to them.
- 0.3 (2026-11-01), applying the answers to 0.2's questions given on [#4469](https://github.com/huggingface/lerobot/pull/4469#issuecomment-6555461272):
  - `layout` made explicit and required for every family: `origin` (`at` plus `description`), `axes` (`x`, `y`, `z` in words) and `surface` (`gel`, `die`, `skin`, `pad`, `vendor`) declare the coordinate frame `positions_mm` is expressed in. `positions_mm` is required for `taxel_array` and `magnetic`, optional for `force_torque`, omitted for `visuotactile`. `layout.frame` removed: it only ever said `sensor`, and `origin` and `axes` now say what that frame is. (Q5)
  - `baseline.frame_range` added: `[first_seq, last_seq]` of the native frames a capture averaged, so a baseline that absorbed a load can be found. (Q7)
  - Baseline stream `tactile/<instance_id>/baseline/episode-{index:06d}.parquet` added, with a capture row for every baseline applied and a raw sample every `baseline_stream.sample_every` native samples; stored data plus the baseline in force reconstructs raw, and the samples make that checkable. The session record's `baseline` is the capture at session start; later captures live in the stream. (Q7, principle 5)
  - Normalization stated normatively: the format stores physical units and applies no scaling; normalization is per channel at load time from `stats.json`. (Q4)
  - TLabel described as a sibling storage format with a bidirectional mapping, not an annotation layer above this schema. The converter lives in `tlabel`; this side's stable surfaces are listed in docs/recording.md, Interop contract.
  - Open questions: 3 settled with the validator as the mitigation; 4, 5 and 7 resolved; 10 rewritten as the choice between a per-channel `measured` flag and a `capabilities` block, left open.
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
  - `channels` required for `taxel_array`, `magnetic` and `force_torque`; per-frame shape is `(n_elements, len(channels))`, with `force_torque` as `(1, 6)`. Stats are per channel; per-taxel statistics are computed by `taxel doctor`.
  - Native-rate sidecar layout defined: `tactile/<instance_id>/episode-{index:06d}.parquet` with `episode_index`, `sample_index`, `host_ts`, `device_ts`, `seq`, `data`. Health reports move to `tactile/health.json`.
  - Record path: the public dataset API writes tactile features with no core change; stock `lerobot-record` does not, so recording goes through `taxel record`, with the upstream change opened as [huggingface/lerobot#5149](https://github.com/huggingface/lerobot/pull/5149).
  - Sibling `.displacement` / `.force` keys checked against 0.6.1: they load, get stats and carry `info` exactly as a single key does, so they stay.
  - Interoperability rewritten against PR #4469's code and TLabel's spec. Open questions revised.
  - 2026-10-30: `clock` (`wall_time`, `host_clock`, `episode_origins`) and `frames` (`received`, `dropped`, `overflowed`) added to the session record. The host clock has no epoch and LeRobot's timestamps restart in every episode; the anchor and the per-episode origins make sidecar `host_ts` comparable with both, and the counters record losses as the sensor counted them.
- 0.1 (Oct 2026): first draft for feedback.
