# LeRobotDataset v3 on disk, and where tactile data doesn't fit

Notes from inspecting `lerobot/pusht` (LeRobot 0.6.1, dataset format v3.0) as groundwork for a tactile sensor plugin. The first half is a reference for the on-disk layout; the second half is where tactile data fits that layout, where it doesn't, and what LeRobot's open tactile PR does and doesn't address.

## Environment notes (macOS, Apple Silicon)

- Python 3.12 venv via `uv`, `uv pip install 'lerobot[pusht,training,dataset_viz]'`. PyTorch 2.11, MPS available.
- `gym-pusht` was not pulled in by the `pusht` extra in 0.6.1; `uv pip install gym-pusht` separately.
- torchcodec looks for ffmpeg at `/opt/homebrew/opt/ffmpeg`; Homebrew's `ffmpeg-full` installs elsewhere, so LeRobot falls back to pyav. Harmless for inspection and small training runs.
- `lerobot-eval` with default async envs spawns subprocesses that fail on Mac with duplicate SDL/libav library warnings; `--eval.use_async_envs=false` works.

## On-disk layout

`lerobot/pusht`: 206 episodes, 25,650 frames, 10 fps, one camera.

```
<root>/
├── meta/
│   ├── info.json            # schema: features (dtype, shape, names), fps, totals,
│   │                        #   path templates for data/ and videos/
│   ├── stats.json           # per-feature mean/std/min/max over all frames
│   ├── tasks.parquet        # task string(s) ↔ task_index
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet   # one row per episode: row range in data/,
│                                  #   from/to timestamp per video key, length,
│                                  #   per-episode per-feature stats
├── data/
│   └── chunk-000/
│       └── file-000.parquet       # all frames, one row per frame, episodes stacked
└── videos/
    └── observation.image/
        └── chunk-000/
            └── file-000.mp4       # all episodes concatenated, one file per video key
```

Path templates from `info.json`:

- `data_path`: `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
- `video_path`: `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`

PushT fits in one chunk and one file. Larger datasets roll over to `file-001`, `file-002`; the episodes table records which chunk and file each episode landed in (`data/chunk_index`, `data/file_index`, and the same per video key).

### How an episode is located

Episode boundaries are metadata, not file boundaries. The episodes table holds, per episode, `dataset_from_index` / `dataset_to_index` (row range in `data/`) and `videos/<key>/from_timestamp` / `to_timestamp` (span in that key's mp4). Frame *k* of episode *e* is row `from_index + k` and video time `from_timestamp + k / fps`.

Worked example: row 500 of `data/` is `episode_index 3`, `frame_index 80`, `timestamp 8.0`. The episodes table gives episode 3 `dataset_from_index 420`, `length 141`, and a video span of 14.1 s; 420 + 80 = 500 and 141 / 10 fps = 14.1 s.

`examples/inspect_dataset.py` reproduces the layout listing and this lookup for any dataset: `python examples/inspect_dataset.py lerobot/pusht --episode 3 --frame 80`.

### Feature keys

| Key | dtype / shape | Meaning |
| --- | --- | --- |
| `observation.image` | video (96, 96, 3) | camera frame; pixels live in the mp4, the parquet row carries only the timestamp |
| `observation.state` | float32 (2,) | agent (pusher) x, y in pixels |
| `action` | float32 (2,) | commanded x, y |
| `episode_index` | int64 | episode number |
| `frame_index` | int64 | frame number within the episode, from 0 |
| `timestamp` | float32 | seconds within the episode, = `frame_index / fps` |
| `index` | int64 | global row number |
| `task_index` | int64 | index into `meta/tasks` |
| `next.reward` | float32 | task reward after this frame |
| `next.done`, `next.success` | bool | episode terminated / succeeded |

### Normalization

`stats.json` carries mean/std/min/max per feature over the whole dataset. Policies subtract the mean and divide by std on the way in and invert it on the way out; the stats are copied into the policy checkpoint. For `lerobot/pusht`, state and action have near-identical stats (mean ≈ (229, 293), std ≈ (102, 96)), as expected for a position-following controller.

## Where tactile fits

- **A taxel array sampled at the dataset fps** is one more parquet column. Proposed key `observation.tactile.<instance_id>`, float32 `(n_taxels, n_channels)`. The parquet, the episodes table and `stats.json` already handle arbitrary extra columns.
- **A visuotactile sensor** (GelSight-family) is a video key, structurally identical to `observation.image`: its own folder under `videos/` and its own from/to timestamps in the episodes table.
- **One key per sensor instance** matches how cameras already work: `observation.image` is keyed by camera, not by a "camera" type. Keying tactile by instance (`left_fingertip`, `palm_0`) follows the existing convention rather than inventing a `TactileSensor` container.

## Where tactile doesn't fit

- **Native rate.** Every row of `data/` is one frame at the dataset's single fps, and `timestamp` is one instant per row. A 1 kHz taxel array alongside a 10–30 Hz camera has no home. Options: keep only the sample nearest each frame (loses data), store a `(samples_per_frame, n_taxels)` block per row (works but `timestamp` no longer describes the block), or a second table with its own fps and its own row ranges in the episodes table, which is what [Multi-Frequency-LeRobot](https://github.com/RoboCoin-BAAI/Multi-Frequency-LeRobot) does outside core.
- **Static per-sensor metadata.** A feature entry in `info.json` has `dtype`, `shape`, `names`. There is no slot for taxel geometry, vendor/model/serial, gel ID or calibration version. Whether extra keys under a feature entry survive `LeRobotDataset.create()` and `push_to_hub()` is untested.
- **Per-episode sensor state.** The episodes table already stores per-episode stats per feature, which is the natural place for a session baseline or a health score, but no key is defined for it.
- **Device vs host time.** `timestamp` is derived from `frame_index`. A sensor's own device clock and the host receive time have no column, so clock skew between tactile and video cannot be measured after the fact from the dataset alone.

## State of native tactile support: PR #4469

Source: [huggingface/lerobot#4469](https://github.com/huggingface/lerobot/pull/4469), open, single commit `5c65381` (17 Aug 2026), checked 28 Oct 2026.

**What it adds.** `src/lerobot/tactile/`, mirroring `src/lerobot/cameras/` almost exactly: an abstract `TactileSensor` base (`tactile.py`), a draccus `ChoiceRegistry` config (`configs.py`), `utils.py`, and per-backend folders with `tactile_<x>.py` / `configuration_<x>.py`. The base class exposes `read()`, `async_read()` and `read_latest()`.

**Data types.** `TactileDataType` has four values, all float64 with shape `(N, data_dim)`:

| Type | Columns | Shape |
| --- | --- | --- |
| `displacement` | dx, dy, dz | (N, 3) |
| `force` | Fx, Fy, Fz | (N, 3) |
| `full` | dx, dy, dz, Fx, Fy, Fz | (N, 6), carried over from closed PR #3175 |
| `wrench` | Fx, Fy, Fz, Mx, My, Mz | (N, 6), new in this PR |

`read()` returns only the array. There are no taxel positions or geometry; point order is implicit row order. Frames carry no timestamp. Each sensor config has a single `fps` value (PaXini default 90); there is no per-stream native-rate storage.

**Backends.** `simulated` (Gaussian-noise frames for CI, not a physics sim) and `paxini` (pyserial with a hand-written wire codec, no vendor SDK). The PaXini backend reads resultant mode only: one `[Fx, Fy, Fz]` point per fingertip, up to five fingertips. Fz is decoded from an unsigned byte at 0.1 N per LSB, so it caps at 25.5 N. Distributed (per-taxel) mode is not implemented. Only module slots 2, 6, 10, 14 and 18 are recognised, so sensors at other hub slots are missed.

**Tests.** 27, all hardware-free: codec functions, config validation, simulated backend.

**Scope.** Changes nothing outside the tactile module and one `pyproject` extra. No dataset features, no `lerobot-record` path, no policy feature types. Nothing in LeRobot consumes a `TactileSensor` yet.

**Unresolved in the PR thread** as of the check date:

- how `observation.tactile.*` should land in `LeRobotDataset` features
- whether the 6D `[dx, dy, dz, Fx, Fy, Fz]` layout should be the canonical on-disk form, or sibling `(N, 3)` features instead
- naming and registry conventions for third-party tactile plugins (today they register as `lerobot_camera_*`; `lerobot_tactile_<name>` was proposed in the thread)
- how a ~90 Hz tactile stream maps onto a ~30 fps dataset (decimate, hold or resample)

No Hugging Face maintainer has commented on the PR.

## Open questions

1. Does LeRobot preserve unknown keys inside a feature's `info.json` entry, or strip them on write? Decides whether sensor metadata can ride in the feature entry or needs a top-level `tactile_sensors` block.
2. Where should per-sensor metadata (taxel positions, serial, calibration version) and native-rate timestamps live: inside the feature entry, a top-level block, or a sidecar? PR #4469 defines neither; see the section above.
3. Does [TLabel](https://github.com/liesliy/tlabel)'s 14-dimension schema map onto one-key-per-instance, or does it collapse instances into one feature?
4. For a visuotactile sensor stored as a video key, where do gel ID and calibration version attach?
5. What does `LeRobotDataset.create()` → `add_frame()` → `save_episode()` → `finalize()` require of a new feature?
