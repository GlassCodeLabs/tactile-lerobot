"""Print the on-disk layout, feature keys and one worked episode lookup for a LeRobotDataset."""

import argparse
import json

import pandas as pd
from lerobot.datasets import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Hub dataset id, e.g. lerobot/pusht")
    parser.add_argument("--episode", type=int, default=0, help="episode to use for the worked lookup")
    parser.add_argument("--frame", type=int, default=0, help="frame within that episode")
    args = parser.parse_args()

    ds = LeRobotDataset(args.repo_id)
    root = ds.root

    print("episodes:", ds.num_episodes)
    print("frames:  ", ds.num_frames)
    print("fps:     ", ds.fps)
    print("root:    ", root)

    print("\nfeatures:")
    for key, spec in ds.features.items():
        print(f"  {key:32s} dtype={spec['dtype']:8s} shape={spec['shape']}")

    info = json.loads((root / "meta" / "info.json").read_text())
    print("\nmeta/info.json keys:", list(info.keys()))
    print("data_path: ", info["data_path"])
    print("video_path:", info.get("video_path"))

    print("\nfiles:")
    for sub in ("meta", "data", "videos"):
        for p in sorted((root / sub).rglob("*")):
            if p.is_file():
                print("  ", p.relative_to(root))

    # Worked lookup: episodes table -> row in data/ -> span in videos/
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    episodes = pd.concat(pd.read_parquet(p) for p in ep_files).set_index("episode_index")
    ep = episodes.loc[args.episode]
    row = int(ep["dataset_from_index"]) + args.frame

    print(
        f"\nepisode {args.episode}: rows {int(ep['dataset_from_index'])}–{int(ep['dataset_to_index'])}, "
        f"length {int(ep['length'])}"
    )
    for col in episodes.columns:
        if col.startswith("videos/") and col.endswith("timestamp"):
            print(f"  {col}: {ep[col]}")

    data_files = sorted((root / "data").rglob("*.parquet"))
    data = pd.concat(pd.read_parquet(p) for p in data_files)
    print(f"\nframe {args.frame} of episode {args.episode} -> row {row}:")
    print(data.iloc[row].to_string())


if __name__ == "__main__":
    main()
