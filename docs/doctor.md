# Checking a recording with `taxel doctor`

A tactile sensor degrades quietly. The no-contact reading creeps (drift), a visuotactile gel wears and pixels stop responding (gel), the device or the host drops frames (framerate), the sensor's clock and the camera's clock disagree so touch and video are misaligned (skew), and readings pin at the top of the range so the force is unknown (saturation). None of this is an error at record time; it shows up later as a policy that trains badly. `taxel doctor` makes it visible per episode, with numbers, before anyone trains on the data. It also checks that a dataset is what it claims to be: that the stored frames can be reproduced from the sidecar, which [SCHEMA.md](../SCHEMA.md) promises under principle 5.

```shell
taxel doctor <dataset_root_or_repo_id> [--episodes 0,3-5] [--instances left_fingertip] \
    [--reference observation.images.front] [--json path] \
    [--fail-on drift=0.05,dropped=0.02,gel=0.05,skew=50,saturated=0.01,integrity] \
    [--thresholds file.json] [--no-write]
```

The report goes to `tactile/health.json` inside the dataset, beside the data and never inside `info.json`; `--no-write` prints only, `--json` writes a second copy. A per-instance table is printed either way. The exit code is 0 when no `--fail-on` threshold is hit, 1 when one is, 2 on a usage error. Without `--fail-on` a flag is an annotation and nothing more (principle 6): the exit code is 0 and no frame or episode is ever dropped. A Hub repo id is downloaded into LeRobot's cache directory. Reading a dataset needs the `taxel[doctor]` extra (pyarrow and huggingface_hub); the image checks decode video through LeRobot and need `taxel[lerobot]`.

## What it reads

For every tactile instance and every episode: the main table's copy of the stream, and, when the dataset was recorded with `taxel record`, the native stream `tactile/<id>/episode-*.parquet`, the baseline stream `tactile/<id>/baseline/episode-*.parquet` and the session record `tactile/<id>/meta.json` ([recording.md](recording.md)). Video frames are decoded only when a check asks for them. Metadata goes through the lenient readers: an unknown key in `info` or in the session record, or a key a later schema version added, is listed in the report's `instances` block, never raised.

A plain LeRobot dataset with tactile-looking features and no sidecar, which is every public dataset today, loads too. A feature looks tactile when its key is `observation.tactile` or `observation.tactile.<id>`; `--instances` can name any instance id or any feature key, so a visuotactile stream stored under a camera key can be checked by name. Every check then degrades to what the main table allows and says what it skipped and why; the sections below say how. An instance with no `info` block is reported with `metadata: none`, one whose block fails even the lenient read with `metadata: invalid` and the reason.

## The integrity check, run first

Three parts, each reported separately, each `skipped` with a reason when its input is missing, all three skipped on a plain dataset.

- `copy`: the main-table copy equals the reduction recomputed from the native stream at the episode's frame times, `origin + k / fps` from the session record's `clock`, with the method and window declared in `info.reduction`, or `last` over one period when the rates match. Equality is exact in float32. It needs the native stream and the clock.
- `raw`: every sample row of the baseline stream equals `raw_frames()` at its `sample_index`, bit for bit: stored data plus the capture in force. A mismatch means the subtractor changed without reporting. That is a firmware that re-zeroed without telling the host, a filter that tracks slowly, an adapter bug, or a host `apply_baseline()` called on a backend whose device already stamps its own baseline, which the base class ignores silently because it subtracts only from frames that arrive unstamped. The report names the first mismatching `sample_index`. This is never counted as drift. It needs the native stream and the baseline stream.
- `files`: every episode in the session's range has its native file and its baseline file; the baseline file opens with a capture row at sample 0 when `baseline.applied` is true, and has none when it is false; `clock.episode_origins` has one entry per episode; the session's `frames.received` is at least the native rows summed over its episode files and `frames.dropped` at least the `seq` gaps inside them. The counters count every frame since `connect()`, including the baseline capture and the pauses between episodes, which the files do not hold, so the relations are inequalities. It needs the session record.

A failed part flags the check. A failure of `copy` or `raw` on a dataset nobody edited is a bug report against the adapter or the writer; on an edited dataset it says which frame was changed.

## The five health checks

| Check | Measures | Needs | On a plain dataset | Flag when |
| --- | --- | --- | --- | --- |
| `drift` | The no-contact level against the baseline in force, per channel, in native units and as a fraction of `range` | Native stream, baseline stream, the session's `baseline`, `range` for the fraction | Runs on the main table: the no-contact level and its change across the episode; the baseline part is skipped and `confidence` is `low` | `offset_pct_range` above `drift`, or the native offset above `drift_units` when the caller set one |
| `framerate` | Effective rate, dropped fraction, gap histogram, longest gap | Native `host_ts` and `seq`, the session's `frames` counters | Only the main table's `timestamp` column, which LeRobot writes as `frame_index / fps`; reported as `nominal`, it says nothing about the sensor | `dropped_pct` above `dropped`, or `max_gap_ms` above ten declared periods (or `max_gap_ms`) |
| `saturation` | Fraction of raw samples at or beyond the range limit, per channel | Native stream and `range` | The main-table copy; without `range`, the observed extremes with `confidence: low` | `saturated_pct` above `saturated` |
| `gel` | Dead pixels of an image stream and their largest cluster; constant elements of an array stream | A visuotactile feature and LeRobot's decoder, or an array stream | The same, from the main table or the video | `dead_pct` above `gel`, or `largest_cluster_px` above `cluster_px` |
| `skew` | Lag between a contact onset in the tactile stream and the same onset in a reference signal | A press visible in both; the `clock` block to put sidecar times on the dataset timeline; a reference named by `--reference` or chosen by motion energy | Both signals at nominal frame times, resolution one frame period | `abs(lag_ms)` above `skew`, only with `confidence` at or above the floor |

### drift

The baseline in force at native sample `i` is the latest capture row of the baseline stream with `sample_index <= i`. When the episode has no capture rows (`applied: false`) it is the session record's `baseline.values`, and when the session recorded no baseline there is none: the offset is then skipped and the check reports the level change alone.

The episode is cut into windows (0.5 s by default, `--window-s`), restarting at every capture row so no window straddles a re-baseline. A window is *quiet* when, after removing its own linear trend, every element's spread is within four times the stream's noise floor, the lower quartile of the spread over windows; that excludes press onsets and releases but keeps a slow ramp and a held press. The raw level of a window is the mean of the sample rows inside it, whose `values` are raw, and the report says `source: samples`; a window with no sample row uses `raw_frames()` at its native samples and the report says `source: raw_frames`. Among the quiet windows, the largest set that lies on one line, the lower one on a tie since a press adds to the rest level, is *no-contact*, and that line is the drift. A press held through a window is a quiet plateau off that line, so it is contact and is not scored. The number of window pairs tried as lines is `candidates`, at most 300.

`offset` is the line at the last no-contact window, never extrapolated past the data, per channel the element of largest magnitude; `slope_per_s` is the median element's rate; `offset_pct_range` divides the offset by `max − min` of the declared `range`. A capture row inside the episode is a `rebaseline` event with its `sample_index`, `host_ts` and the per-channel delta between the new and the old values; it is reported, not scored, and drift after it is measured against the new values. `frame_range` is the session baseline's; a record written before the key existed reads `"not recorded"`, a sensor without `seq` `"no seq"`. `first_frame_zero` is true when the stored frame 0 is exactly zero on every channel: a stream whose reference is re-zeroed at every episode start looks like that. It is an observation, never a flag.

The flag compares `offset_pct_range` with `drift` when the range is known. Without a range the fraction has no meaning, and the native offset is compared with `drift_units` only when the caller set it; with neither, drift reports its numbers and never flags. With no baseline either, the numbers are a level change in units the check cannot scale, and `confidence` is `low`.

### framerate

From the native stream: the effective rate over the episode, the histogram of gaps between consecutive `host_ts` in units of the declared period (edges at 1.5, 2.5 and 3.5 periods, the last bin open), the longest gap, and the dropped fraction from the `seq` gaps over the episode's `seq` span. The session's `frames` counters are reported beside it. From the main table, the `timestamp` column gives regularity against the dataset fps and nothing about source drops; when the column is `frame_index / fps`, which is what LeRobot's writer puts there, the report says `timestamps: nominal`.

### saturation

Per channel, the fraction of raw samples at or beyond the limit. The raw frames are reconstructed from the stored data and the captures when a baseline was subtracted, because a limit applies to what the sensor read; a few float32 ulps of slack cover the rounding. With `range` declared the limit is the sensor's and `confidence` is `declared`. Without it the observed extremes stand in and `confidence` is `low`: a clean recording then shows one sample per channel at each extreme and nothing more.

### gel

On an image stream, a sample of up to 60 frames spread over the episode is decoded, and a pixel is dead when its grayscale variance over them is below a fifth of the median pixel variance; the largest 4-connected cluster of dead pixels is reported too. A fifth, not a hundredth, because video encoding changes what a constant region looks like: LeRobot's default encoder (AV1 at CRF 30, a keyframe every two frames) leaves a dead region flickering at a few tenths of a count of variance while flattening weak sensor noise. An image stream whose live pixels vary by only a count or two cannot be checked for dead pixels after encoding, and the check then under-counts rather than inventing clusters; a sharp edge rings under the codec, so a cluster's edge pixels may be missed. Real gel images vary by tens of counts under contact and lighting and are well above that floor.

On an array stream the check reports `kind: elements` and lists the elements that are constant to within one part in a thousand of their own largest magnitude over the episode, on every channel. The rule is per element, not relative to the others: a taxel array's elements share a unit but not a level of activity, and a plain dataset's tactile vector may hold quantities of different scales.

### skew

The tactile side is an event signal: the absolute first difference of the contact magnitude, the L1 distance of each frame from the episode's median frame, placed midway between the two samples it compares. The reference is the motion energy of a camera or image stream, the mean absolute difference between consecutive frames, placed at `(k − 0.5) / fps` because it measures the change between frames `k − 1` and `k`; or another array instance's event signal. With a sidecar, both sides are on the host clock, the tactile side by `host_ts` and the video by the episode's origin from the `clock` block; on a plain dataset both are on LeRobot's nominal timeline and the resolution is one frame period. Both event trains are binned on a 5 ms grid, smoothed by a triangle as wide as the coarser series' period, and cross-correlated over ±500 ms.

`lag_ms` is positive when the tactile stream lags the reference. `confidence` is the normalized cross-correlation peak, 0 to 1: how much of the two trains' energy lines up at the best lag. One press gives an onset and a release, quantized to different video-frame phases, so a clean recording of a simulated pair scores 0.7 to 0.8, not 1.0; two unrelated trains score near 0. A lag is flagged only with `confidence` at or above the floor (0.6 by default), and below the floor the check is skipped with the value in the reason. `resolution_ms` is the coarser of the grid and the two series' spacing, so a lag against a 30 fps camera cannot be trusted below 33 ms. `onsets` counts the peaks in the tactile event train.

The reference is `--reference` when given, else the image feature with the most motion in the episode, else another array instance. A camera that watches the robot move has motion energy everywhere and no contact onset to match; on such a dataset the check skips most episodes for low confidence, and the few lags it does report are the floor being grazed. A camera framed on the contact, or a second tactile instance pressed by the same event, is the reference that measures something.

## Thresholds

The defaults are conservative: they flag what a careful operator would want to look at, not what would certainly ruin a policy, and `--fail-on` decides what fails a run.

| Name | Default | Flags | Meaning |
| --- | --- | --- | --- |
| `drift` | 0.05 | drift | Offset as a fraction of the declared range |
| `drift_units` | none | drift | Offset in native units, for an instance without a range |
| `dropped` | 0.02 | framerate | Dropped fraction of native frames |
| `max_gap_ms` | ten declared periods | framerate | Longest gap between native frames |
| `gel` | 0.05 | gel | Dead fraction of pixels or elements |
| `cluster_px` | 25 | gel | Largest dead cluster, in pixels |
| `skew` | 50 ms | skew | Lag magnitude |
| `confidence` | 0.6 | skew | Cross-correlation peak a lag needs before it can be flagged |
| `saturated` | 0.01 | saturation | Fraction of samples at or beyond the limit |
| `integrity` | on | integrity | Any failed part |

`--fail-on` takes a comma-separated list of these names; `drift=0.1` sets the value and makes a drift flag fail the run, a bare `integrity` only makes that flag fail the run. `--thresholds file.json` holds an object with any of the same keys, merged over the defaults; an unknown key is a usage error. The values used are written into the report, so two runs are distinguishable. A skipped check never fails a run.

## The report

`tactile/health.json` is one document: `doctor_version`, `generated_at`, the `thresholds` used, an `instances` block with how each instance's metadata was read and the lenient readers' unknown and missing keys, and `episodes`, keyed `episode_{index:06d}`, with one block of the six checks per instance. A check that could not run is `{"skipped": "<reason>"}` and nothing else, so an absent `flag` means not evaluated, never passed. The exact shape, with an example, is in [SCHEMA.md](../SCHEMA.md), Health annotations, and `taxel.schema.HealthReport` reads it back.

The printed table has one block per instance, one row per episode and one cell per check with the headline number, `!` after a flagged cell, `-` for a skipped one, and the skip reasons listed once per instance with their counts.

## A worked example

A simulated sensor recorded with `taxel record`: 500 Hz, 16 elements, two episodes of 2 s at 30 fps, the baseline applied with a raw sample every 50 native frames. Its injected faults are 5% source drops, a ramp of 0.8 per second on the third channel, a clamp at ±6 declared as `range`, a press of amplitude 8 held for 0.6 s early in the first episode, and a backend re-baseline 3.6 s after connecting, inside the second episode.

```shell
taxel doctor outputs/faulty --window-s 0.25 --fail-on dropped,saturated,drift,integrity
```

```
left_fingertip  observation.tactile.left_fingertip  taxel_array  metadata: info  sidecar: yes  500 Hz  range: [-6.0, 6.0]
episode  integrity          drift                          framerate                 saturation          gel                skew
000000   ok copy,raw,files  +1.6 (13.4%) !                 5.2% dropped, gap 6 ms !  7.58% (declared) !  0 dead element(s)  -
000001   ok copy,raw,files  +0.726 (6.1%), 1 rebaseline !  4.7% dropped, gap 7 ms !  0.00% (declared)    0 dead element(s)  -
  skipped skew: no reference: the dataset has no camera and no other tactile instance (2 episodes)

2 episode(s), 1 instance(s), 5 flag(s); ! marks a flag, - a skipped check
written outputs/faulty/tactile/health.json
fail: episode_000000 left_fingertip drift
fail: episode_000000 left_fingertip framerate
fail: episode_000000 left_fingertip saturation
fail: episode_000001 left_fingertip drift
fail: episode_000001 left_fingertip framerate
```

The exit code is 1. Reading the second episode's drift block from the report:

```json
"drift": {
  "no_contact_level": [-0.006, -0.0076, 2.1384],
  "level_change": [0.0144, 0.0225, 1.4862],
  "offset": [0.0111, 0.0191, 0.7262],
  "offset_pct_range": [0.0009, 0.0016, 0.0605],
  "slope_per_s": [0.0079, 0.009, 0.822],
  "baseline": "capture", "source": "samples", "baseline_age_s": 2.376, "frame_range": [8, 108],
  "windows": 8, "no_contact_windows": 8, "candidates": 6, "first_frame_zero": false,
  "rebaselines": [{"sample_index": 490, "host_ts": 3439374.93, "delta": [0.0021, 0.0019, 2.7902]}],
  "confidence": "normal", "flag": true
}
```

The slope on the third channel is 0.822 per second against an injected 0.8; the other two sit at 0.01. The backend re-baselined at native sample 490, and the new baseline is 2.79 above the old one on that channel, which is 0.8 per second over the 3.5 s since the session's capture: the ramp, absorbed into a re-zero that the stored frames no longer show. Drift after it is measured against the new values, so the offset at the last window is 0.73, which is 6.1% of the 12-unit range and over the 5% default. In the first episode the press held through a window is contact, not drift, and the clamp shows up as saturation: 4 of 16 elements pinned for 0.6 s of 2 s is 7.5% of the samples, 7.58% measured. The dropped fraction is 4.7% and 5.2% against the injected 5%. Integrity passes on all three parts: the copy reproduces, the 20 sample rows equal the reconstruction, and the files and counters agree with the session record. Skew is skipped because there is no camera and no second instance.

## From Python

```python
from taxel.doctor.loader import DoctorDataset
from taxel.doctor.report import run_doctor, write_report
from taxel.schema import Thresholds

dataset = DoctorDataset("outputs/faulty")
report = run_doctor(dataset, thresholds=Thresholds(drift=0.1), window_s=0.25)
write_report(report, dataset.root)
first = report.episodes["episode_000000"]["left_fingertip"]
print(first.drift.slope_per_s, first.framerate.dropped_pct)
```

`taxel.doctor.checks` holds the check functions as pure functions on numpy arrays, and `taxel.doctor.windows` the window and signal helpers; they are what the unit tests exercise, and a reader with its own loader can call them directly.

## What it does not do

It annotates; it never filters or edits a dataset. It does not run on a live sensor. It cannot measure drift in a stream with no baseline and no range beyond reporting the level change, cannot measure skew without an event visible in both signals, and cannot find dead pixels in an image stream quieter than the encoder's own flicker. Decoding the cameras for the skew reference dominates the running time on a dataset with video; `--reference` avoids decoding the others.
