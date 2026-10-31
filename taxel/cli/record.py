"""``taxel record``: record a LeRobot dataset with tactile sensors, beside a robot or beside cameras alone.

The robot and camera side is LeRobot's own: the robot comes from
``make_robot_from_config``, its features from the processor pipelines through
``hw_to_dataset_features`` and its frames from ``build_dataset_frame``, so a
robot's data records exactly as ``lerobot-record`` records it. The tactile side
is :mod:`taxel.dataset`: each instance gets the ``SCHEMA.md`` layout, the
per-frame copy in the main table and the native stream and session record in
``tactile/<instance_id>/``.

By hand, with a webcam and no robot::

    taxel record \\
        --robot.type=cameras_only \\
        --robot.cameras="{webcam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --tactile="{left_fingertip: {type: simulated, declared_rate_hz: 500, n_elements: 52}}" \\
        --dataset.repo_id=<user>/<name> --dataset.single_task="press the cube" \\
        --dataset.num_episodes=5 --dataset.episode_time_s=20

With a robot, give ``--robot.type`` and ``--teleop.type`` as for ``lerobot-record``;
the robot's own ``--robot.cameras`` carry the cameras. ``--tactile`` maps each
instance id to its sensor config, where ``type`` names a registered backend. The
dataset options are ``lerobot-record``'s under ``--dataset.*``, with the Hub push
off by default.
"""

# No `from __future__ import annotations`: LeRobot's parser.wrap reads the config class from the
# function's annotation object, and draccus builds the dataclasses from their real type hints.

import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from taxel.dataset import missing_dependency
from taxel.dataset.writer import TactileDatasetWriter, tactile_features
from taxel.schema import SchemaError
from taxel.sensors import TactileSensor, TactileSensorConfig, make_tactile_sensor

try:
    import lerobot.scripts.lerobot_record  # noqa: F401  registers every robot, teleoperator and camera type
    from draccus.parsers.decoding import decode, decode_dataclass
    from draccus.utils import ParsingError
    from lerobot.cameras import CameraConfig, make_cameras_from_configs
    from lerobot.configs import parser
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.datasets import (
        LeRobotDataset,
        VideoEncodingManager,
        aggregate_pipeline_dataset_features,
        create_initial_features,
        safe_stop_image_writer,
    )
    from lerobot.processor import make_default_processors
    from lerobot.robots import Robot, RobotConfig, make_robot_from_config
    from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
    from lerobot.utils.constants import ACTION, DEFAULT_FEATURES, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.keyboard_input import init_keyboard_listener
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging, log_say
except ImportError as e:
    raise missing_dependency("LeRobot", e) from None

__all__ = ["CamerasOnly", "CamerasOnlyConfig", "RecordConfig", "TactileDatasetRecordConfig", "main", "record"]

logger = logging.getLogger(__name__)


class UsageError(ValueError):
    """A problem with the options that is reported without a traceback."""


# --- Configuration ---------------------------------------------------------------------------------


@dataclass
class TactileDatasetRecordConfig(DatasetRecordConfig):
    """``lerobot-record``'s dataset options, with the Hub push off by default."""

    push_to_hub: bool = False


@RobotConfig.register_subclass("cameras_only")
@dataclass
class CamerasOnlyConfig(RobotConfig):
    """A robot that is only cameras, for recording by hand: no motors, no actions, no teleoperator."""

    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: TactileDatasetRecordConfig
    # Tactile sensors by instance id; each value's `type` names a registered backend.
    tactile: dict[str, TactileSensorConfig] = field(default_factory=dict)
    # Required for a robot with actions, as in lerobot-record; not allowed with cameras_only.
    teleop: TeleoperatorConfig | None = None
    # How the per-frame copy is reduced from a faster native stream: mean or last.
    reduction: str = "mean"
    # Span of native samples each dataset frame is reduced from; default one frame period.
    window_ms: float | None = None
    # Seconds of no-contact frames averaged into the session baseline, hands off; 0 skips it.
    baseline_s: float = 2.0
    # Session record id; default the UTC start time.
    session_id: str | None = None
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Append to an existing dataset as a new session.
    resume: bool = False

    def __post_init__(self) -> None:
        if not self.tactile:
            raise UsageError('at least one tactile sensor is required: --tactile="{<id>: {type: ..., ...}}"')
        for instance_id, config in self.tactile.items():
            if config.instance_id != instance_id:
                raise UsageError(f"--tactile key {instance_id!r} is not instance_id {config.instance_id!r}")
        if self.reduction not in ("mean", "last"):
            raise UsageError(f"--reduction must be mean or last, got {self.reduction!r}")
        if not self.dataset.repo_id:
            raise UsageError("--dataset.repo_id is required")


@decode.register(TactileSensorConfig, include_subclasses=True)
def _decode_tactile_config(cls: type, raw: Any, path: Sequence[str] = ()) -> TactileSensorConfig:
    """draccus decoder for ``--tactile``: the mapping key is the instance id, ``type`` picks the backend."""
    if not isinstance(raw, dict):
        raise ParsingError(f"expected a mapping for a tactile sensor config, got {raw!r}")
    raw = dict(raw)
    if path:
        raw.setdefault("instance_id", path[-1])
    type_name = raw.pop("type", None)
    if type_name is not None:
        subcls = TactileSensorConfig.get_choice_class(type_name)
    elif cls is not TactileSensorConfig:
        subcls = cls
    else:
        known = sorted(TactileSensorConfig.get_known_choices())
        raise ParsingError(f"tactile sensor config needs a 'type', one of {known}")
    return decode_dataclass(subcls, raw, path)


# --- A camera-only robot -----------------------------------------------------------------------------


class CamerasOnly(Robot):
    """Cameras and nothing else. ``get_observation`` returns one frame per camera; actions are ignored."""

    config_class = CamerasOnlyConfig
    name = "cameras_only"

    def __init__(self, config: CamerasOnlyConfig) -> None:
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._connected = False

    @property
    def observation_features(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @property
    def action_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        for cam in self.cameras.values():
            cam.connect()
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected")
        return {name: cam.async_read() for name, cam in self.cameras.items()}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return action

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        self._connected = False


def make_robot(config: RobotConfig) -> Robot:
    return CamerasOnly(config) if isinstance(config, CamerasOnlyConfig) else make_robot_from_config(config)


# --- The loop ----------------------------------------------------------------------------------------


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    writer: TactileDatasetWriter,
    events: dict,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
) -> None:
    """One dataset frame per iteration, as ``lerobot-record``'s loop, with the tactile frame added.

    With ``dataset`` the loop records; without it the loop only runs the robot and
    drains the sensors, which is how the reset pause between episodes is spent.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
    control_interval = 1 / fps
    timestamp = 0.0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        # The tactile frame is built right after the cameras are read, so both see the same instant.
        tactile_frame = writer.frame() if dataset is not None else None
        if dataset is None:
            writer.idle()
        obs_processed = robot_observation_processor(obs)

        action_values = None
        if teleop is not None:
            act = teleop.get_action()
            action_values = teleop_action_processor((act, obs))
            robot.send_action(robot_action_processor((action_values, obs)))

        if dataset is not None:
            frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            if action_values is not None:
                frame.update(build_dataset_frame(dataset.features, action_values, prefix=ACTION))
            frame.update(tactile_frame)
            frame["task"] = single_task
            dataset.add_frame(frame)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). "
                "Dataset frames might be dropped and robot control might be unstable."
            )
        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


def check_resume_compatibility(dataset: LeRobotDataset, robot: Robot, fps: int, features: dict) -> None:
    """The dataset being resumed must have been recorded at this fps, with this robot type and these features.

    Feature ``info`` blocks are not compared here, as in ``lerobot-record``; the writer
    checks the tactile ones strictly.
    """
    if dataset.fps != fps:
        raise UsageError(f"the dataset was recorded at {dataset.fps} fps, not {fps}")
    if dataset.meta.robot_type != robot.name:
        recorded = dataset.meta.robot_type
        raise UsageError(f"the dataset was recorded with robot {recorded!r}, not {robot.name!r}")
    expected = {**features, **DEFAULT_FEATURES}
    for key, ft in expected.items():
        actual = dataset.meta.features.get(key)
        if actual is None:
            raise UsageError(f"the dataset has no feature {key!r}")
        if actual["dtype"] != ft["dtype"] or tuple(actual["shape"]) != tuple(ft["shape"]):
            raise UsageError(f"feature {key!r} is {actual['dtype']} {tuple(actual['shape'])} in the dataset")
    extra = sorted(set(dataset.meta.features) - set(expected))
    if extra:
        raise UsageError(f"the dataset has features this setup does not produce: {extra}")


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    if robot.action_features and teleop is None:
        raise UsageError(f"robot {robot.name!r} takes actions, so a teleoperator is required: --teleop.type")
    if not robot.action_features and teleop is not None:
        raise UsageError(f"robot {robot.name!r} takes no actions; drop --teleop")
    sensors: list[TactileSensor] = [make_tactile_sensor(config) for config in cfg.tactile.values()]

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    robot_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    num_cameras = len(getattr(robot, "cameras", {}))

    dataset = None
    writer = None
    listener = None
    try:
        # Sensors first: a connected sensor's first frame gives the per-frame shape of its feature.
        for sensor in sensors:
            sensor.connect()
        tactile = tactile_features(sensors, cfg.dataset.fps, cfg.reduction, cfg.window_ms)
        features = {**robot_features, **tactile}

        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
            )
            check_resume_compatibility(dataset, robot, cfg.dataset.fps, features)
        else:
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )
        writer = TactileDatasetWriter(dataset, sensors, baseline_s=cfg.baseline_s)

        if teleop is not None:
            teleop.connect()
        robot.connect()
        listener, events = init_keyboard_listener()

        if cfg.baseline_s > 0:
            log_say("Capturing the tactile baseline, keep the sensors untouched", cfg.play_sounds)
        writer.start_session(cfg.session_id)

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                writer.start_episode()
                record_loop(
                    robot=robot,
                    writer=writer,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                )
                writer.stop_episode()

                # A pause to reset the environment, skipped after the last episode.
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        writer=writer,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    writer.discard_episode()
                    continue

                writer.end_episode()
                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        for sensor in sensors:
            sensor.disconnect()
        if writer is not None:
            writer.finalize()
        if dataset is not None:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset is not None and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved, skipping the push to the Hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main(argv: Sequence[str]) -> int:
    """Run ``taxel record`` with the options after the subcommand; return the exit code."""
    register_third_party_plugins()
    saved_argv = sys.argv
    sys.argv = [f"{Path(saved_argv[0]).name} record", *argv]
    try:
        record()
    except ParsingError as e:
        # draccus wraps an error raised while building the config; the cause is the message that matters.
        cause = e.__cause__
        message = str(cause) if isinstance(cause, UsageError) else f"{e}: {cause}" if cause else str(e)
        print(f"error: {message}", file=sys.stderr)
        return 2
    except (UsageError, SchemaError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        sys.argv = saved_argv
    return 0
