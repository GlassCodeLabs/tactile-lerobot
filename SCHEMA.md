# Tactile data schema for LeRobot datasets — draft 0.1

*Status: draft for feedback, November 2026. Nothing here is final. The goal is to converge with [LeRobot PR #4469](https://github.com/huggingface/lerobot/pull/4469), [TLabel](https://github.com/liesliy/tlabel) and [Multi-Frequency-LeRobot](https://github.com/RoboCoin-BAAI/Multi-Frequency-LeRobot), not to add a fourth incompatible layout. Open questions are listed at the end; those are the ones I'd most like answered.*

## The problem this solves

A tactile feature in a LeRobot dataset today is typically an unnamed array: 116 floats per frame, or an image. Months later nobody can answer which sensor produced it, where each element sat on the fingertip, whether the gel was new or worn, what calibration converted raw readings to newtons, or what the sensor read with nothing touching it. Two sensors with different taxel counts cannot share one feature. Streams at different rates get resampled to the camera rate and lose their timing. None of this is visible until a model trained on the data fails on a fresh sensor.

## Five principles

1. **Per instance, not per type.** A dataset with two fingertips has two features, not one feature with a stacked dimension. Instances have stable IDs across sessions.
2. **Metadata once, data per frame.** Everything static about a sensor instance (geometry, serial, gel ID, calibration, declared rate) lives in dataset metadata. Per-frame data is only what changes.
3. **Native rate, own timestamps.** Tactile frames keep the rate and timestamps they were captured at. Alignment with video or joint streams happens at read time, never at write time.
4. **Provenance keys always exist.** `calibration` and `baseline` are present on every instance even when their value is `null`, so their absence is a visible fact rather than a missing key.
5. **Health is an annotation, not a filter.** Quality scores are written alongside the data, per episode, and never silently drop frames or episodes.

## Feature keys

```
observation.tactile.<instance_id>
```

`instance_id` is a short slug chosen by the recorder and stable across sessions: `left_fingertip`, `right_fingertip`, `palm_0`, `thumb`. It must match a key under `tactile_sensors` in the metadata.

For visuotactile sensors the image stream is stored the way camera streams are (video-encoded), under the same key. An optional marker stream uses a suffix:

```
observation.tactile.<instance_id>.markers
```

## Metadata block

Added to the dataset's `info.json` (or wherever LeRobot core settles for sensor metadata) as `tactile_sensors`, one entry per instance.

```json
"tactile_sensors": {
  "left_fingertip": {
    "family": "taxel_array",
    "vendor": "PaXini",
    "model": "PX-6AX-GEN3 DP-S1813-Elite",
    "serial": "PX3-00412",
    "firmware": "1.4.2",
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
      "values": [[0.01, -0.02, 0.00], "..."]
    },
    "mount": {"body": "so101_follower", "link": "gripper_left_finger", "pose": null}
  },

  "right_fingertip": {
    "family": "visuotactile",
    "vendor": "GelSight",
    "model": "DIGIT",
    "serial": "D20491",
    "firmware": null,
    "declared_rate_hz": 60,
    "image": {"width": 320, "height": 240, "channels": 3},
    "gel": {"id": "gel-2026-09-b", "installed_at": "2026-09-30", "cycles_at_install": 0},
    "markers": {"count": 63, "grid": [7, 9]},
    "calibration": {
      "version": null,
      "method": null,
      "date": null,
      "reference": null
    },
    "baseline": {
      "captured_at": "2026-11-21T14:03:11Z",
      "duration_s": 2.0,
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
      "values": [[412.0, -88.5, 1203.2], "..."]
    },
    "mount": {"body": "so101_follower", "link": "palm", "pose": null}
  }
}
```

Field notes:

- `family` is one of `taxel_array`, `visuotactile`, `magnetic`, `force_torque`. Adding a family is a schema change; adding a vendor is not.
- `layout.positions_mm` is the physical position of each element in the sensor's own frame. For taxel arrays and magnetic sensors its length equals the per-frame element count. For visuotactile sensors the image itself is the layout; `markers.grid` gives the marker arrangement.
- `gel` and `skin` are the replaceable-part identifiers. They are what make "this sensor degraded 30% faster than that one" answerable later.
- `calibration.method` is free text for now (`vendor`, `3dcal`, `nano17-press`, `uncalibrated-by-design`). A controlled vocabulary can come later once there are a few real values.
- `baseline` is what the sensor read with nothing touching it at session start. For taxel and magnetic families it is stored as values; for visuotactile it is a pointer to a reference frame.
- `mount` records where on the robot the sensor sits. `pose` is a 4×4 or (xyz, quat) in the link frame if known, else `null`.

## Per-frame data

| Family | dtype | Shape per frame | Notes |
| --- | --- | --- | --- |
| `taxel_array` | float32 | `(n_elements, n_channels)` | Never uint8. The PaXini resultant mode caps Fz at 25.5 N in uint8; store distributed mode, or scale correctly, or both |
| `visuotactile` | uint8 | `(H, W, 3)` video-encoded; optional `.markers` float32 `(n_markers, 2)` | Encoded like camera streams; markers are pixel coordinates |
| `magnetic` | float32 | `(n_sensors, 3)` | Raw field. Baseline-subtracted view is computed at read time, not stored |
| `force_torque` | float32 | `(6,)` | fx, fy, fz, tx, ty, tz in the sensor frame |

Every family's per-frame row also carries:

| Column | dtype | Meaning |
| --- | --- | --- |
| `device_ts` | float64 or null | Timestamp from the sensor's own clock, seconds, if the device provides one |
| `host_ts` | float64 | Host receive time, seconds, monotonic clock |
| `seq` | int64 or null | Device sequence number if provided; gaps mean dropped frames at the source |

## Timestamps and alignment

Tactile streams are stored at their native rate in their own table (following the Multi-Frequency-LeRobot approach of a per-sensor parquet file rather than one merged table). The dataset's main `timestamp` column stays at the dataset fps. A reader aligns tactile to a video frame by nearest-`host_ts` or by a window; the policy for that lives in the loader, not the file.

If LeRobot core adopts a different multi-rate layout, this schema follows it. The non-negotiable part is that native timestamps are never discarded at write time.

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

- **PR #4469 types.** `taxel_array` maps to its point-set type with per-point force; `force_torque` maps to its wrench type; the `TactileSensor` base class in this project mirrors that PR's surface so a merge or dependency is possible.
- **TLabel.** A converter in both directions is planned: TLabel's semantic fields populate `tactile_sensors` where they overlap, and this metadata exports to TLabel's schema for annotation. Which fields overlap is one of the open questions below.
- **MCAP.** One channel per instance, schema per family, `host_ts` as the log time, `device_ts` as publish time where present. Camera and joint channels exported alongside so Foxglove opens the whole episode.

## Open questions (please answer any of these)

1. Per-instance feature keys vs. one feature with a ragged or padded dimension: is there a reason the ragged approach is better for training code?
2. Is `layout.positions_mm` the right level of geometry, or do people need full per-element frames (position + orientation)? Does anyone actually use this in a policy today?
3. For visuotactile sensors, is a reference frame pointer enough for `baseline`, or should a small reference image be stored in metadata directly?
4. What does your pipeline do with the no-contact baseline today? Subtract at record time, at load time, or never?
5. Multi-rate: would you accept a per-sensor parquet layout, or does your loader need a single table?
6. What calibration methods are in use that should be in a controlled vocabulary?
7. Which of these fields does TLabel's schema already carry under another name?
8. What is missing that would make you refuse to use this?

## Changelog

- 0.1 (Nov 2026): first draft for feedback.
