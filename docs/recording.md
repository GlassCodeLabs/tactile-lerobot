# Recording with `taxel record`

`taxel record` records a LeRobot dataset with one or more tactile sensors, beside a robot or beside cameras alone. The robot and camera side is LeRobot's own code, so a robot's features come out exactly as `lerobot-record` writes them. The tactile side is `taxel.dataset`, which gives each sensor instance the layout in [SCHEMA.md](../SCHEMA.md). This page says what gets written and where, how the per-frame copy is made from the native stream, how the timing is anchored, and how to run it with and without a robot.

## What is written

For a dataset with one robot camera and one tactile instance `left_fingertip`:

```
<root>/
├── meta/                                   # LeRobot's own files, unchanged
│   ├── info.json                           #   the tactile feature entry carries its static metadata in `info`
│   ├── stats.json  tasks.parquet  episodes/
├── data/chunk-000/file-000.parquet         # one row per dataset frame, including the tactile copy
├── videos/observation.images.webcam/...
└── tactile/
    └── left_fingertip/
        ├── meta.json                       # one session record per recording session
        ├── episode-000000.parquet          # every native frame of episode 0, with its timing
        └── episode-000001.parquet
```

| What | Where | Written by |
| --- | --- | --- |
| Static metadata of the instance: family, vendor, model, declared rate, channels, layout, mount, hub, `reduction` | `meta/info.json`, feature entry `observation.tactile.left_fingertip`, key `info` | `LeRobotDataset.create`, from `tactile_features()` |
| Dataset-rate copy, float32 `(n_elements, len(channels))` per frame | the main table, same feature key | `add_frame`, from `TactileDatasetWriter.frame()` |
| Native-rate stream with `host_ts`, `device_ts`, `seq` | `tactile/left_fingertip/episode-{index:06d}.parquet` | `end_episode()` |
| Session record: serial, adapter, calibration, baseline, clock anchor, frame counters, episode range | `tactile/left_fingertip/meta.json` | `end_episode()` and `finalize()` |

## Static metadata in the feature, everything per-session beside it

LeRobot's `merge` requires two datasets to have identical feature entries, `info` included. So the feature entry's `info` holds only what is true of the instance in every recording: the sensor's description and the `reduction` rule. Anything that can differ between two recordings of the same sensor, such as the serial after a swap, the calibration, the no-contact baseline, the gel or skin, the clock anchor and the frame counters, lives in the sidecar's session record. The writer builds the `info` block from the sensor's own `metadata()` and reads it back strictly before recording, so a per-session fact cannot end up in `info` by accident: an unknown key there is an error, not an extra field.

## The per-frame copy and the reduction rule

A tactile sensor usually runs faster than the dataset. A 500 Hz sensor delivers about seventeen frames per dataset frame at 30 fps. The native frames are what the sensor produced and go to the sidecar unchanged. The main table gets one array per dataset frame, reduced from the native frames by the method in the feature's `info.reduction`, which `taxel record` sets from `--reduction` (default `mean`) with `window_ms` equal to one dataset frame period unless `--window_ms` says otherwise.

For a dataset frame at time `T` and a window `w`:

- `mean` is the mean of the native samples with `T − w < host_ts ≤ T`.
- `last` is the newest native sample in that same window.

Both use only samples received by `T`, and the writer builds a frame at or after its time, so the window is always complete. That is what makes the stored copy reproducible: `taxel.dataset.reduce.reduce_episode` run on the sidecar file at the episode's frame times returns exactly the column in the main table, and the tests check that for every frame.

**`nearest` is refused for array streams.** `SCHEMA.md` lists it because it describes what a video encoder does with a camera frame. For an array stream it would take the native sample nearest to `T` within half a window on either side, so the nearer sample may arrive after the dataset frame has already been built. The copy would then depend on when the loop ran, and a reader recomputing it from the sidecar could get a different answer. `reduce.py` still implements it for readers; the writer and `taxel record` accept `mean` and `last` only.

**When no native sample falls in a window** the frame repeats the previous frame's value, and the first frame of an episode falls back to the newest sample at or before its time. This happens with a sensor slower than the dataset, where most windows are empty by construction, and with a sensor that stalls. The sidecar shows the truth either way: a stall is a gap in `host_ts` and `seq` there, and the per-frame copy never invents data. The rule is deterministic, so the copy stays reproducible from the sidecar.

When the sensor's declared rate equals the dataset fps, no `reduction` is declared and the copy is the newest native sample at each frame time, which is the native stream itself up to timing jitter.

## Timing: where the dataset's clock and the sensor's clock meet

Every native frame carries `host_ts`, the host's monotonic clock at receipt, from the same clock LeRobot stamps camera frames with. That clock has no epoch, and LeRobot's own `timestamp` column is nominal: `frame_index / fps`, restarting at zero in every episode. The session record's `clock` block joins the two:

```json
"clock": {
  "wall_time": "2026-11-21T14:03:05.117Z",
  "host_clock": 10234.117,
  "episode_origins": [10240.004, 10301.772]
}
```

`wall_time` and `host_clock` are read together when the session starts, right after the sensors connect, so any `host_ts` maps to the calendar as `wall_time + (host_ts − host_clock)`. `episode_origins` holds, for each episode of the session in order, the host clock when the episode's first dataset frame was built. Dataset frame `k` of that episode is at `origin + k / fps` on the host clock. That is the time the frame's tactile copy was reduced at, and the time to align the sidecar to when a loader wants the native timing. `taxel.dataset.sidecar.frame_times` and `align_nearest` do that arithmetic.

Frames are reduced at these nominal times, not at the moment the loop happened to run, so the copy is a function of the sidecar and the origin alone. The loop itself waits for each frame's nominal time if it gets there early, which keeps the tick regular.

## Counters and baseline

The session record's `frames` block is the sensor's own counters for the session: `received` by the host, `dropped` at the source (gaps in `seq`), `overflowed` on the host because the recorder did not drain the sensor in time. They are copied from the sensor, not reconstructed from the files; with `seq` present, `dropped` equals the sum of the `seq` gaps across the session's episode files.

At the start of a session the writer averages `--baseline_s` seconds of frames (default 2) into the record's `baseline` with `applied: false`. Keep hands off the sensor while `taxel record` says so. Nothing is subtracted from the recorded data; the baseline is stored so a reader can subtract it, and so drift between sessions is measurable.

## Recording by hand: cameras and tactile, no robot

In LeRobot 0.6 every camera belongs to a robot, so recording without an arm uses a robot type that is only cameras:

```shell
taxel record \
    --robot.type=cameras_only \
    --robot.cameras="{webcam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --tactile="{left_fingertip: {type: simulated, declared_rate_hz: 500, n_elements: 52}}" \
    --dataset.repo_id=<user>/<name> \
    --dataset.single_task="press the cube" \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=20
```

`--robot.cameras` is LeRobot's camera mapping, with the same options as for any robot. `--tactile` maps an instance id to a sensor config; `type` names a registered backend and the other keys are that backend's config fields (see [sensors.md](sensors.md)). The instance id becomes the feature key `observation.tactile.left_fingertip` and the sidecar directory. Leave `--robot.cameras` out to record tactile data alone. A `cameras_only` recording has no `action` or `observation.state` feature, and `robot_type` in `info.json` says `cameras_only`.

## Recording with a robot

Give the robot and teleoperator as for `lerobot-record`; the robot's own `--robot.cameras` carry the cameras:

```shell
taxel record \
    --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58760431541 --robot.id=black \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem58760431551 --teleop.id=blue \
    --tactile="{left_fingertip: {type: simulated, declared_rate_hz: 500, n_elements: 52, mount: {body: so101_follower, link: gripper_left_finger}}}" \
    --dataset.repo_id=<user>/<name> --dataset.single_task="pick up the cube" --dataset.num_episodes=50
```

The robot's observation and action features, the processor pipelines, the dataset creation and the loop are LeRobot's, so the non-tactile part of the dataset is what `lerobot-record` would have written. A teleoperator is required for a robot with actions and not allowed with `cameras_only`. Third-party robot and camera plugins are discovered the way `lerobot-record` discovers them.

## Options

Everything under `--dataset.*` is `lerobot-record`'s `DatasetRecordConfig`, with one difference: `--dataset.push_to_hub` is off by default. The rest:

| Option | Default | Meaning |
| --- | --- | --- |
| `--tactile` | required | Mapping from instance id to sensor config |
| `--reduction` | `mean` | `mean` or `last`, for sensors faster than the dataset |
| `--window_ms` | one frame period | Span of native samples each dataset frame is reduced from |
| `--baseline_s` | `2` | Seconds averaged into the session baseline; `0` records none |
| `--session_id` | UTC start time | Name of the session record |
| `--resume` | `false` | Append to an existing dataset as a new session |
| `--play_sounds` | `true` | Speak the recording events, as `lerobot-record` does |

The keys that end an episode early, re-record it or stop recording are `lerobot-record`'s. `--resume` checks the dataset's fps, robot type and features against the current setup and refuses a sensor whose metadata differs from the recorded `info`. Each session appends a record to `meta.json` with its own episode range, clock anchor, baseline and counters.

## From Python

The writer runs beside any loop that calls `add_frame`:

```python
from lerobot.datasets import LeRobotDataset

from taxel.dataset.writer import TactileDatasetWriter, tactile_features
from taxel.sensors import SimulatedTactile, SimulatedTactileConfig

sensor = SimulatedTactile(SimulatedTactileConfig(instance_id="left_fingertip", declared_rate_hz=500))
sensor.connect()
features = tactile_features([sensor], fps=30)  # plus the robot's own features
dataset = LeRobotDataset.create("user/name", fps=30, features=features, root="outputs/name")
writer = TactileDatasetWriter(dataset, [sensor])
writer.start_session()
for _ in range(2):
    writer.start_episode()
    for _ in range(60):
        dataset.add_frame({**writer.frame(), "task": "press"})
    writer.stop_episode()
    writer.end_episode()
    dataset.save_episode()
sensor.disconnect()
writer.finalize()
dataset.finalize()
```

`frame()` waits for the frame's nominal time if called early; a loop that paces itself at the dataset fps, as LeRobot's does, never waits. Between `stop_episode()` and `end_episode()` the loop may run a reset pause with `writer.idle()`, and `discard_episode()` drops the episode instead of writing it.

## What LeRobot 0.6 does with the result

- A normal load ignores `tactile/` and works as for any dataset. `delta_timestamps` works on tactile keys.
- `push_to_hub` uploads `tactile/` with the rest of the folder.
- A partial load (`episodes=[...]`) downloads only the listed files, and `lerobot-edit-dataset` copies only `meta/`, `data/` and `videos/`, so both leave the sidecar behind.
- `lerobot-edit-dataset merge` fails on any dataset with a 2-D feature, tactile or not: it rewrites the data parquet through pandas, and pyarrow cannot convert a 2-D array column from a DataFrame. Merging a tactile dataset in 0.6.x therefore needs a tool in this project, which also has to concatenate the session records and offset their episode ranges, since core does not know the sidecar. There is no such tool yet.

## Reading the sidecar without LeRobot

The episode files are plain parquet. `data` is an Arrow fixed-shape tensor column, so pyarrow returns it as one array:

```python
import json

import pyarrow.parquet as pq

table = pq.read_table("outputs/name/tactile/left_fingertip/episode-000000.parquet")
data = table.column("data").combine_chunks().to_numpy_ndarray()  # (n_samples, n_elements, n_channels)
host_ts = table.column("host_ts").to_numpy()
meta = json.load(open("outputs/name/tactile/left_fingertip/meta.json"))
```

`taxel.dataset.sidecar.read_episode` and `read_session_meta` do the same with validation, and need only numpy and pyarrow.
