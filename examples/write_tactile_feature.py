"""Write a LeRobot v3 dataset with a per-instance tactile feature and read it back."""

import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("outputs/tactile_feature_demo")
TACTILE_KEY = "observation.tactile.left_fingertip"
N_TAXELS, N_CHANNELS = 52, 3

features = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"joint_{i}" for i in range(6)]},
    "action": {"dtype": "float32", "shape": (6,), "names": [f"joint_{i}" for i in range(6)]},
    TACTILE_KEY: {
        "dtype": "float32",
        "shape": (N_TAXELS, N_CHANNELS),
        "names": None,
        # Static, per-instance metadata rides along in the feature's `info` dict.
        "info": {
            "family": "taxel_array",
            "vendor": "example",
            "model": "example-52",
            "declared_rate_hz": 30,
            "channels": ["fx", "fy", "fz"],
            "units": {"force": "N"},
        },
    },
}


def main() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)
    rng = np.random.default_rng(0)

    dataset = LeRobotDataset.create(
        "example/tactile_feature_demo", fps=30, features=features, root=ROOT, use_videos=False
    )
    for _ in range(3):
        for _ in range(50):
            dataset.add_frame(
                {
                    "observation.state": rng.standard_normal(6).astype(np.float32),
                    "action": rng.standard_normal(6).astype(np.float32),
                    TACTILE_KEY: rng.standard_normal((N_TAXELS, N_CHANNELS)).astype(np.float32),
                    "task": "press",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    reloaded = LeRobotDataset("example/tactile_feature_demo", root=ROOT)
    frame = reloaded[0]
    print(f"{reloaded.num_episodes} episodes, {reloaded.num_frames} frames")
    print(f"{TACTILE_KEY}: dtype={frame[TACTILE_KEY].dtype}, shape={tuple(frame[TACTILE_KEY].shape)}")
    print("sensor metadata:", reloaded.meta.features[TACTILE_KEY]["info"])
    # Dataset stats for an N-D feature are reduced over every axis but the last,
    # so this is per channel, not per taxel.
    print("stats mean shape:", np.asarray(reloaded.meta.stats[TACTILE_KEY]["mean"]).shape)


if __name__ == "__main__":
    main()
