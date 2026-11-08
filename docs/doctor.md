# Checking a recording with `taxel doctor`

A tactile sensor degrades quietly. The no-contact reading creeps (drift), a visuotactile gel wears and pixels stop responding (gel), the device or the host drops frames (framerate), the sensor's clock and the camera's clock disagree so touch and video are misaligned (skew), readings pin at the top of the range so the force is unknown (saturation), and a reader with no new frame passes the last one on again (held). None of this is an error at record time; it shows up later as a policy that trains badly. `taxel doctor` makes it visible per episode, with numbers, before anyone trains on the data. It also checks that a dataset is what it claims to be: that the stored frames can be reproduced from the sidecar, which [SCHEMA.md](../SCHEMA.md) promises under principle 5.

```shell
taxel doctor <dataset_root_or_repo_id> [--episodes 0,3-5] [--instances left_fingertip] [--guess] \
    [--checks drift,framerate] [--reference observation.images.front | all] [--clock source_timestamp] \
    [--json path] [--fail-on drift=0.05,dropped=0.02,gel=0.05,skew=50,saturated=0.01,held=0.05,integrity] \
    [--thresholds file.json] [--no-write]
```

The report goes to `tactile/health.json` inside the dataset, beside the data and never inside `info.json`; `--no-write` prints only, `--json` writes a second copy. A per-instance table is printed either way. The exit code is 0 when no `--fail-on` threshold is hit, 1 when one is, 2 on a usage error. Without `--fail-on` a flag is an annotation and nothing more (principle 6): the exit code is 0 and no frame or episode is ever dropped. `--checks` runs only the named checks; the others are reported as `skipped: not selected` and cost nothing, so a CI job can leave out the ones that decode video. A Hub repo id is downloaded into LeRobot's cache directory. Reading a dataset needs the `taxel[doctor]` extra (pyarrow and huggingface_hub); the image checks decode video and need `taxel[lerobot]`.

## What it reads

For every tactile instance and every episode: the main table's copy of the stream, and, when the dataset was recorded with `taxel record`, the native stream `tactile/<id>/episode-*.parquet`, the baseline stream `tactile/<id>/baseline/episode-*.parquet` and the session record `tactile/<id>/meta.json` ([recording.md](recording.md)). Metadata goes through the lenient readers: an unknown key in `info` or in the session record, or a key a later schema version added, is listed in the report's `instances` block, never raised. LeRobot writes keys of its own into a video feature's `info` (`video.*`, `audio.*`, `has_audio`, `is_depth_map`); they are set aside, and a block left with none of the tactile block's required keys is not an instance's.

A plain LeRobot dataset with tactile-looking features and no sidecar loads too. A feature looks tactile when its key is `observation.tactile` or `observation.tactile.<id>`. Public datasets rarely name their streams that way, so `--guess` lists the feature keys that contain `tactile`, `touch`, `wrench` or `force`, with their dtype and shape and the `--instances` that names them, and exits. `--instances` takes any instance id or feature key: a visuotactile stream stored under a camera key, a force-torque vector or a glove's zone array is checked by name. A run that finds no instance stops with a usage error that says so. Every check then degrades to what the main table allows and says what it skipped and why; the sections below say how. An instance with no `info` block is reported with `metadata: none`, one whose block fails even the lenient read with `metadata: invalid` and the reason.

Video is decoded only when a check asks for it, in one sequential pass per stream and episode from the keyframe before the episode's first frame; the pass gives the skew check its motion energy and the gel check its sampled frames. Frame `k` of the file is row `k` of the episode, and its presentation time is checked against the episode's `from_timestamp` plus the `timestamp` column to half a period, which is what LeRobot itself asks its decoder for. A dataset whose `timestamp` holds the camera's real presentation times, a 29.97 Hz video under a declared 30 fps, therefore decodes. A video file that is not on disk, as in a partial download, makes the checks that need it skip with the file's path.

LeRobot writes `frame_index / fps` into `timestamp`, and some datasets keep the recorder's real clock in a column of their own. `--clock <column>` names that column, a number per frame: the framerate check and both skew event trains of every stream read from the main table run on its times, and the framerate block names it in `clock`. Video is still decoded at `timestamp`. A column the dataset does not have, or one that is not a number per frame, is a usage error.

### Missing readings

A NaN in an array stream is a reading the dataset marks as missing: a blanked interval, a sensor that dropped out. Every array block (drift, saturation, gel, held) carries `nan_pct`, per element the fraction of frames in which it reads NaN on any channel. An element NaN in more than half the frames is left out of the check, and when every element is, the check is skipped and says so. The others run on their finite samples: a drift window counts an element-channel with fewer than two finite samples as not quiet, saturation's fractions are of the finite samples, a held pair with a NaN on either side is not compared, and the integrity check counts a NaN reproduced as NaN as equal.

## The integrity check, run first

Three parts, each reported separately, each `skipped` with a reason when its input is missing, all three skipped on a plain dataset.

- `copy`: the main-table copy equals the reduction recomputed from the native stream at the episode's frame times, `origin + k / fps` from the session record's `clock`, with the method and window declared in `info.reduction`, or `last` over one period when the rates match. Equality is exact in float32. It needs the native stream and the clock.
- `raw`: every sample row of the baseline stream equals `raw_frames()` at its `sample_index`, bit for bit: stored data plus the capture in force. A mismatch means the subtractor changed without reporting. That is a firmware that re-zeroed without telling the host, a filter that tracks slowly, an adapter bug, or a host `apply_baseline()` called on a backend whose device already stamps its own baseline, which the base class ignores silently because it subtracts only from frames that arrive unstamped. The report names the first mismatching `sample_index`. This is never counted as drift. It needs the native stream and the baseline stream.
- `files`: every episode in the session's range has its native file and its baseline file; the baseline file opens with a capture row at sample 0 when `baseline.applied` is true, and has none when it is false; `clock.episode_origins` has one entry per episode; the session's `frames.received` is at least the native rows summed over its episode files and `frames.dropped` at least the `seq` gaps inside them. The counters count every frame since `connect()`, including the baseline capture and the pauses between episodes, which the files do not hold, so the relations are inequalities. It needs the session record.

In `copy` and `raw` a NaN reproduced as NaN is equal. A failed part flags the check. A failure of `copy` or `raw` on a dataset nobody edited is a bug report against the adapter or the writer; on an edited dataset it says which frame was changed.

## The six health checks

| Check | Measures | Needs | On a plain dataset | Flag when |
| --- | --- | --- | --- | --- |
| `drift` | The no-contact level against the baseline in force, per channel, in native units and as a fraction of `range` | Native stream, baseline stream, the session's `baseline`, `range` for the fraction | Runs on the main table: the no-contact level and its change across the episode; the baseline part is skipped and `confidence` is `low` | `offset_pct_range` above `drift`, or the native offset above `drift_units` when the caller set one |
| `framerate` | Effective rate, dropped fraction, gap histogram, longest gap, stalled steps | Native `host_ts` and `seq`, the session's `frames` counters | The main table's `timestamp` column, reported as `nominal` when it is `frame_index / fps`; or the `--clock` column | `dropped_pct` above `dropped`, or `max_gap_ms` above ten declared periods (or `max_gap_ms`) |
| `saturation` | Fraction of raw samples at or beyond the limit, per channel, and at each limit apart | Native stream and `range` | The main-table copy; without `range`, the fractions at the observed extremes, never flagged | `saturated_pct` above `saturated`, against a declared `range` only |
| `gel` | Dead pixels of an image stream and their largest cluster; elements of an array stream constant in every checked episode | A visuotactile feature and LeRobot's decoder, or an array stream | The same, from the main table or the video | `dead_pct` above `gel`, or the largest cluster above `cluster_pct` of the image (`cluster_px` when set) |
| `skew` | Lag between a contact onset in the tactile stream and the same onset in a reference signal | A press visible in both; the `clock` block to put sidecar times on the dataset timeline; a reference camera or instance | Both signals on the main table's times, resolution one frame period | `abs(lag_ms)` above `skew`, only with `confidence` at or above the floor |
| `held` | Frames identical to the frame before, per group of elements when the feature's names give one | An array stream | The main-table copy | `held_pct` above `expected_pct` by more than `held`, on a stream with a declared rate |

### drift

The baseline in force at native sample `i` is the latest capture row of the baseline stream with `sample_index <= i`. When the episode has no capture rows (`applied: false`) it is the session record's `baseline.values`, and when the session recorded no baseline there is none: the offset is then skipped and the check reports the level change alone.

The episode is cut into windows (0.5 s by default, `--window-s`), restarting at every capture row so no window straddles a re-baseline. A window is *quiet* when, after removing its own linear trend, nine tenths of its element-channels have a spread within four times their noise floor, the lower quartile of the spread over windows; that excludes press onsets and releases but keeps a slow ramp and a held press. Nine tenths, not all: on a glove of hundreds of cells some cell is always moving or flickering, and a rule over every element finds no quiet window on exactly the arrays that drift. The fraction still keeps a press on a quarter of a 16-element array out of the quiet windows; a press held through most of an episode on fewer than a tenth of the elements reads as rest for them. The raw level of a window is the mean of the sample rows inside it, whose `values` are raw, and the report says `source: samples`; a window with no sample row uses `raw_frames()` at its native samples and the report says `source: raw_frames`. Among the quiet windows, the largest set that lies on one line, within six noise floors on nine tenths of the element-channels, the lower one on a tie since a press adds to the rest level, is *no-contact*, and that line is the drift. A press held through a window is a quiet plateau off that line, so it is contact and is not scored. The number of window pairs tried as lines is `candidates`, at most 300.

`offset` is the line at the last no-contact window, never extrapolated past the data, per channel the element of largest magnitude; `slope_per_s` is the median element's rate; `offset_pct_range` divides the offset by `max − min` of the declared `range`. A capture row inside the episode is a `rebaseline` event with its `sample_index`, `host_ts` and the per-channel delta between the new and the old values; it is reported, not scored, and drift after it is measured against the new values. `frame_range` is the session baseline's; a record written before the key existed reads `"not recorded"`, a sensor without `seq` `"no seq"`. `first_frame_zero` is true when the stored frame 0 is exactly zero on every channel: a stream whose reference is re-zeroed at every episode start looks like that. It is an observation, never a flag.

The flag compares `offset_pct_range` with `drift` when the range is known. Without a range the fraction has no meaning, and the native offset is compared with `drift_units` only when the caller set it; with neither, drift reports its numbers and never flags. With no baseline either, the numbers are a level change in units the check cannot scale, and `confidence` is `low`.

### framerate

From the native stream: the effective rate over the episode, the histogram of gaps between consecutive `host_ts` in units of the declared period (edges at 1.5, 2.5 and 3.5 periods, the last bin open), the longest gap, and the dropped fraction from the `seq` gaps over the episode's `seq` span. A step of zero length, a clock that did not advance between two frames, is counted in `stalled`, apart from the drops: it is a reader that delivered a frame twice or stamped two at once, not a frame the device lost. The session's `frames` counters are reported beside it. From the main table, the `timestamp` column gives regularity against the dataset fps and nothing about source drops; when the column is evenly spaced at `1 / fps` from its first value, which is what LeRobot's writer puts there, the report says `timestamps: nominal`. With `--clock` the column named is read instead, and the table shows a measured rate to two decimals with the steps over 1.5 periods and the longest: `29.68 Hz (source_timestamp), 15 gaps >50 ms, max 52 ms`.

### saturation

Per channel, the fraction of raw samples at or beyond the limit, and at the lower and the upper limit apart (`at_min_pct`, `at_max_pct`). The raw frames are reconstructed from the stored data and the captures when a baseline was subtracted, because a limit applies to what the sensor read; a few float32 ulps of slack cover the rounding. With `range` declared the limit is the sensor's, `confidence` is `declared`, and the fraction flags. Without it the observed extremes stand in, `confidence` is `low`, and the check never flags: a stream rests at an extreme as readily as it clips there. A 0/1 contact flag sits at its observed minimum most of the time, a glove at rest reads zero, an ADC rests at its floor. The observed fractions say where the data sits, and the table shows them as `min 91.5%, max 0.1% (observed)`; a declared `range` is what makes the fraction a saturation.

### gel

On an image stream, up to 60 frames spread over the episode are sampled and each pixel's grayscale variance over them is measured, in counts². A pixel is dead when its variance is at or under 2.0 counts², and the largest 4-connected cluster of dead pixels is reported in pixels and as a fraction of the image. The check flags when the dead fraction is over `gel` or the largest cluster is over `cluster_pct` of the image; `cluster_px`, when set, replaces the fraction with a pixel count.

On a quietly encoded stream the pixel check does not run, and `rest_variance` is the number that says why. Video encoding flattens weak camera noise to the same tenths of a count that a dead region flickers at, so before judging any pixel the check takes the instance's rest flicker: the median pixel variance of the quietest checked episode. When it is under 6.0 counts², no episode of the instance is checked for dead pixels, and the reason gives the number, `quiet gel: median pixel variance 0.88 counts² in the quietest of 12 checked episode(s) is under the encoder floor 6`. That is the common case for a visuotactile camera with little noise encoded with LeRobot's defaults, which reads about one count², and for a processed image such as a difference map, which is flat where nothing moves; a dead region cannot be told from a live one there, and the check says so rather than report a dead percentage. The rest flicker is the quietest episode's, not each episode's own, because contact lifts an episode's median while the regions nothing touches stay at rest and would read as dead. A checked episode reports its own `median_variance` beside `rest_variance` and `episodes_checked`.

The three numbers are defaults, set on the simulated gel (48×64, camera noise of 6 counts, a 6×6 dead block) after LeRobot's default encoding, AV1 at CRF 30 with a keyframe every two frames. The block's pixels flicker at 0.6 to 1.7 counts² and every live pixel sits at 2.3 or above, so 2.0 takes the block and no live pixel. The simulated median is 10 to 11 counts², over the 6.0 floor. The block's 36 pixels are 1.2% of the image, over the 0.5% cluster default. The margin under 2.0 is thin: another build of the same encoder reads a pixel or two of the block's edge above it.

On an array stream the check lists `constant_elements`, the elements constant to within one part in a thousand of their own largest magnitude over the episode, on every channel. The rule is per element, not relative to the others: a taxel array's elements share a unit but not a level of activity, and a plain dataset's tactile vector may hold quantities of different scales. One episode cannot tell a dead element from one nothing touched; on a glove, most cells of a short episode never see contact. So an element is dead only when it is constant in every checked episode of the instance: the block reports `dead_elements` and `episodes_checked` beside the per-episode list, and the flag is the dead fraction over `gel`. With one checked episode `dead_elements` is `null` and nothing flags. A feature whose values are only 0 and 1 is skipped, since a constant zone of a contact flag is an untouched zone.

The episodes are grouped by session where a session record says which episodes share one. Without a sidecar the session is the run's `--episodes` selection, so a cross-episode finding belongs to the run: a selection that spans two physical devices, a glove swapped between sessions, dilutes a real dead zone, because the zones dead on one device are live on the other. Check the episodes of one device together.

### skew

The tactile side is an event signal: the absolute first difference of the contact magnitude, the L1 distance of each frame from the episode's median frame, placed midway between the two samples it compares. The reference is the motion energy of a camera or image stream, the mean absolute difference between consecutive frames, placed at `(k − 0.5) / fps` because it measures the change between frames `k − 1` and `k`; or another array instance's event signal. With a sidecar, both sides are on the host clock, the tactile side by `host_ts` and the video by the episode's origin from the `clock` block; on a plain dataset both are on the main table's times, LeRobot's nominal timeline or the `--clock` column, and the resolution is one frame period. Both event trains are binned on a 5 ms grid, smoothed by a triangle as wide as the coarser series' period, and cross-correlated over ±500 ms.

`lag_ms` is positive when the tactile stream lags the reference. `confidence` is the normalized cross-correlation peak, 0 to 1: how much of the two trains' energy lines up at the best lag. One press gives an onset and a release, quantized to different video-frame phases, so a clean recording of a simulated pair scores 0.7 to 0.8, not 1.0; two unrelated trains score near 0. A lag is flagged only with `confidence` at or above the floor (0.6 by default), and below the floor the check is skipped with the value in the reason. `resolution_ms` is the coarser of the grid and the two series' spacing, so a lag against a 30 fps camera cannot be trusted below 33 ms. `onsets` counts the peaks in the tactile event train.

The reference is `--reference` when given. By default it is the first video or image feature, in feature order, that is not a checked instance and whose file is on disk: a scene camera, so one stream is decoded for all instances. Without one it is another image instance, then another array instance. `--reference all` decodes every video stream and takes the one with the most motion in the episode, which costs a decode per stream per episode. A camera that watches the robot move has motion energy everywhere and no contact onset to match; on such a dataset the check skips most episodes for low confidence, and the few lags it does report are the floor being grazed. A camera framed on the contact, or a second tactile instance pressed by the same event, is the reference that measures something.

### held

On an array stream, the frames identical to the frame before. A device or a reader with no new frame that passes the last one on again leaves an exact repeat: a glove whose clock stalls and whose frame repeats on the same steps, a blackout filled from the frame before, a run filled with one value. Sensor noise makes an exact repeat of a live frame rare, so `held_pct`, the fraction of compared frames equal to the previous one on every element, is the size of the fault; `longest_run` is the most consecutive held frames and `single_holds` the runs of one. A pair in which either frame holds a NaN is not compared, so a missing reading is never a repeat, and `frames` says how many pairs were.

The native stream is checked when there is one, the main-table copy otherwise. `expected_pct` is what a correct stream holds anyway: none on a native stream, `1 − rate / fps` on a main-table copy of a sensor slower than the dataset. The flag needs a declared rate, and is `held_pct` above `expected_pct` by more than `held`. Without a declared rate the block reports and never flags, since a stream that rests at exactly the same values, a quantized sensor with nothing touching, repeats at rest. A feature whose values are only 0 and 1 is skipped. Image streams are not checked.

When the feature's names give groups, element names that differ only in a trailing `_<index>` (`left_thumb_f6_0` to `left_thumb_f6_5`), `groups` has the same measures per group. A blackout that takes one fingertip and fills it from the frame before is a held group, not a held frame, and a whole-frame count misses it. The table shows the group that holds most beside the frame's own numbers: `0.00% held, run 0, left_thumb_f6 1.12%`. A framerate check that counts `stalled` steps and a held check that counts the same steps say the clock and the data stopped together.

## Thresholds

The defaults are conservative: they flag what a careful operator would want to look at, not what would certainly ruin a policy, and `--fail-on` decides what fails a run.

| Name | Default | Flags | Meaning |
| --- | --- | --- | --- |
| `drift` | 0.05 | drift | Offset as a fraction of the declared range |
| `drift_units` | none | drift | Offset in native units, for an instance without a range |
| `dropped` | 0.02 | framerate | Dropped fraction of native frames |
| `max_gap_ms` | ten declared periods | framerate | Longest gap between native frames |
| `gel` | 0.05 | gel | Dead fraction of pixels or elements |
| `cluster_pct` | 0.005 | gel | Largest dead cluster, as a fraction of the image |
| `cluster_px` | none | gel | Largest dead cluster in pixels; replaces `cluster_pct` when set |
| `skew` | 50 ms | skew | Lag magnitude |
| `confidence` | 0.6 | skew | Cross-correlation peak a lag needs before it can be flagged |
| `saturated` | 0.01 | saturation | Fraction of samples at or beyond a declared limit |
| `held` | 0.05 | held | Held fraction above the expected one |
| `integrity` | on | integrity | Any failed part |

`--fail-on` takes a comma-separated list of these names; `drift=0.1` sets the value and makes a drift flag fail the run, a bare `integrity` only makes that flag fail the run. A name whose check `--checks` leaves out is a usage error, since a check that does not run cannot fail. `--thresholds file.json` holds an object with any of the same keys, merged over the defaults; an unknown key is a usage error. The values used are written into the report, so two runs are distinguishable. A skipped check never fails a run.

Four numbers are constants of the checks rather than thresholds: the dead-pixel variance (2.0 counts²) and the encoder floor (6.0 counts²) in `taxel.doctor.checks`, the NaN floor (half the frames) beside them, and the quiet fraction (nine tenths) in `taxel.doctor.windows`.

## The report

`tactile/health.json` is one document: `doctor_version`, `generated_at`, the `thresholds` used, an `instances` block with how each instance's metadata was read and the lenient readers' unknown and missing keys, and `episodes`, keyed `episode_{index:06d}`, with one block per instance holding the integrity check and the six health checks. A check that could not run is `{"skipped": "<reason>"}` and nothing else, so an absent `flag` means not evaluated, never passed. The exact shape, with an example, is in [SCHEMA.md](../SCHEMA.md), Health annotations, and `taxel.schema.HealthReport` reads it back; a report written before a key was added still reads, with that key at its default and `held` absent.

The printed table has one block per instance, one row per episode and one cell per check with the headline number, `!` after a flagged cell, `-` for a skipped one, and the skip reasons listed once per instance with their counts.

## A worked example

A simulated sensor recorded with `taxel record`: 500 Hz, 16 elements, two episodes of 2 s at 30 fps, the baseline applied with a raw sample every 50 native frames. Its injected faults are 5% source drops, a ramp of 0.8 per second on the third channel, a clamp at ±6 declared as `range`, a press of amplitude 8 held for 0.6 s early in the first episode, and a backend re-baseline 3.6 s after connecting, inside the second episode.

```shell
taxel doctor outputs/faulty --window-s 0.25 --fail-on dropped,saturated,drift,integrity
```

```
left_fingertip  observation.tactile.left_fingertip  taxel_array  metadata: info  sidecar: yes  500 Hz  range: [-6.0, 6.0]
episode  integrity          drift                          framerate                  saturation          gel                            skew  held
000000   ok copy,raw,files  +1.65 (13.7%) !                5.2% dropped, gap 13 ms !  7.52% (declared) !  0 dead, 0 constant element(s)  -     0.00% held, run 0
000001   ok copy,raw,files  +0.755 (6.3%), 1 rebaseline !  4.7% dropped, gap 7 ms !   0.00% (declared)    0 dead, 0 constant element(s)  -     0.00% held, run 0
  skipped skew: no reference: the dataset has no camera and no other tactile instance (2 episodes)

2 episode(s), 1 instance(s), 5 flag(s); ! marks a flag, - a skipped check
written outputs/faulty/tactile/health.json
fail: episode_000000 left_fingertip drift
fail: episode_000000 left_fingertip framerate
fail: episode_000000 left_fingertip saturation
fail: episode_000001 left_fingertip drift
fail: episode_000001 left_fingertip framerate
```

The exit code is 1. Reading the second episode's drift block from the report, with its 16 `nan_pct` values, all zero, cut short:

```json
"drift": {
  "no_contact_level": [-0.0034, -0.0104, 2.1585],
  "level_change": [0.0247, 0.0316, 1.4837],
  "offset": [0.0195, 0.0171, 0.7554],
  "offset_pct_range": [0.0016, 0.0014, 0.0629],
  "slope_per_s": [0.0033, -0.0064, 0.8262],
  "baseline": "capture", "source": "samples", "baseline_age_s": 2.411, "frame_range": [5, 105],
  "windows": 8, "no_contact_windows": 8, "candidates": 6, "first_frame_zero": false,
  "rebaselines": [{"sample_index": 478, "host_ts": 3512564.17, "delta": [0.002, 0.0018, 2.7947]}],
  "confidence": "normal", "nan_pct": [0.0, 0.0, "..."], "flag": true
}
```

The slope on the third channel is 0.826 per second against an injected 0.8; the other two sit within 0.01 of zero. The backend re-baselined at native sample 478, and the new baseline is 2.79 above the old one on that channel, which is 0.8 per second over the 3.5 s since the session's capture: the ramp, absorbed into a re-zero that the stored frames no longer show. Drift after it is measured against the new values, so the offset at the last window is 0.76, which is 6.3% of the 12-unit range and over the 5% default. In the first episode the press held through a window is contact, not drift, and the clamp shows up as saturation: 4 of 16 elements pinned for 0.6 s of 2 s is 7.5% of the samples, 7.52% measured. The dropped fraction is 5.2% and 4.7% against the injected 5%, and no step is stalled. Integrity passes on all three parts: the copy reproduces, the 20 sample rows equal the reconstruction, and the files and counters agree with the session record. No element is constant in either episode, no frame repeats, and skew is skipped because there is no camera and no second instance.

## From Python

```python
from taxel.doctor.loader import DoctorDataset
from taxel.doctor.report import run_doctor, write_report
from taxel.schema import Thresholds

dataset = DoctorDataset("outputs/faulty")
report = run_doctor(dataset, thresholds=Thresholds(drift=0.1), window_s=0.25)
write_report(report, dataset.root)
first = report.episodes["episode_000000"]["left_fingertip"]
print(first.drift.slope_per_s, first.framerate.dropped_pct, first.held.held_pct)
```

`run_doctor` takes the command line's selections as keywords: `episodes`, `instances`, `checks`, `reference` and `clock`. `taxel.doctor.checks` holds the check functions as pure functions on numpy arrays, and `taxel.doctor.windows` the window and signal helpers; they are what the unit tests exercise, and a reader with its own loader can call them directly.

## What it does not do

It annotates; it never filters or edits a dataset. It does not run on a live sensor. It cannot measure drift in a stream with no baseline and no range beyond reporting the level change, cannot measure skew without an event visible in both signals, and cannot find dead pixels in an image stream quieter than the encoder's own flicker; it says so with the rest variance. It does not know which part of an image is gel: pixels dark in every episode, a vignetted border around the gel, count as image. It does not check image streams for held frames, and it judges dead elements across the episodes it is given, not across devices it cannot see.
