import json
import pandas as pd
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset("lerobot/pusht")

print("episodes:", ds.num_episodes)
print("frames:  ", ds.num_frames)
print("fps:     ", ds.fps)
print("root:    ", ds.root)
print("\nfeatures:")
for k, v in ds.features.items():
    print(f"  {k:30s} dtype={v['dtype']:8s} shape={v['shape']}")

root = ds.root
print("\n--- meta/info.json (top-level keys) ---")
info = json.load(open(root / "meta" / "info.json"))
print(list(info.keys()))

print("\n--- first parquet shard ---")
pq = sorted((root / "data").rglob("*.parquet"))[0]
df = pd.read_parquet(pq)
print(pq.relative_to(root), df.shape)
print(df.head(3))

print("\n--- videos ---")
for p in sorted((root / "videos").rglob("*.mp4"))[:5]:
    print(" ", p.relative_to(root))
