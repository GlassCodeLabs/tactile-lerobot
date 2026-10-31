"""Assertions shared by the writer and record tests: a recorded dataset in the SCHEMA.md layout."""

from collections.abc import Callable

import numpy as np

from taxel.dataset.features import read_feature_entry
from taxel.dataset.reduce import reduce_episode
from taxel.dataset.sidecar import episode_path, frame_times, read_episode, read_session_meta
from taxel.schema import feature_key


def check_recording(
    root,
    repo_id: str,
    instance_id: str,
    *,
    is_dropped: Callable[[int], bool],
    session: int = -1,
    device_offset_s: float | None = None,
):
    """Reload a dataset and check one session of one instance against SCHEMA.md.

    Returns ``(dataset, info, record)``. ``is_dropped(seq)`` says whether the sensor
    dropped frame ``seq`` at the source, so the native stream can be checked for
    completeness. Every episode of the session is checked: the file's index and
    shape, monotonic ``host_ts``, the device clock offset when given, no missing
    ``seq`` beyond the source drops, and the stored per-frame copy equal to the
    reduction recomputed from the sidecar at the episode's frame times.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root)
    key = feature_key(instance_id)
    entry = dataset.meta.features[key]
    info = read_feature_entry(entry)
    assert info.reduction is not None
    meta = read_session_meta(root, instance_id)
    record = meta.sessions[session]
    first, last = record.episodes
    assert len(record.clock.episode_origins) == last - first + 1
    assert record.frames is not None and record.frames.received > 0

    for e in range(first, last + 1):
        stream = read_episode(episode_path(root, instance_id, e))
        assert stream.episode_index == e and stream.shape == tuple(entry["shape"])
        assert np.all(np.diff(stream.host_ts) > 0)
        if device_offset_s is not None:
            np.testing.assert_allclose(stream.device_ts - stream.host_ts, device_offset_s, atol=1e-6)
        lo, hi = int(stream.seq[0]), int(stream.seq[-1])
        assert stream.seq.tolist() == [s for s in range(lo, hi + 1) if not is_dropped(s)]

        row = dataset.meta.episodes[e]
        n_frames, start = int(row["length"]), int(row["dataset_from_index"])
        origin = record.clock.episode_origins[e - first]
        targets = frame_times(origin, n_frames, dataset.fps)
        assert stream.host_ts[0] <= targets[0]
        expected = reduce_episode(
            info.reduction.method, stream.host_ts, stream.data, targets, info.reduction.window_ms
        )
        stored = np.stack([dataset[start + k][key].numpy() for k in range(n_frames)])
        np.testing.assert_array_equal(stored, expected)
    return dataset, info, record
