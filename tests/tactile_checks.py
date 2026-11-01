"""Assertions shared by the writer and record tests: a recorded dataset in the SCHEMA.md layout."""

from collections.abc import Callable

import numpy as np

from taxel.dataset.features import read_feature_entry
from taxel.dataset.reduce import reduce_episode
from taxel.dataset.sidecar import (
    baseline_path,
    episode_path,
    frame_times,
    raw_frames,
    read_baseline_episode,
    read_episode,
    read_session_meta,
)
from taxel.schema import feature_key


def check_recording(
    root,
    repo_id: str,
    instance_id: str,
    *,
    is_dropped: Callable[[int], bool],
    session: int = -1,
    device_offset_s: float | None = None,
    raw_of_seq: Callable[[int], np.ndarray] | None = None,
):
    """Reload a dataset and check one session of one instance against SCHEMA.md.

    Returns ``(dataset, info, record)``. ``is_dropped(seq)`` says whether the sensor
    dropped frame ``seq`` at the source, so the native stream can be checked for
    completeness. Every episode of the session is checked: the file's index and
    shape, monotonic ``host_ts``, the device clock offset when given, no missing
    ``seq`` beyond the source drops, the stored per-frame copy equal to the
    reduction recomputed from the sidecar at the episode's frame times, and the
    baseline stream: a raw sample at every multiple of ``baseline_stream.sample_every``
    equal to the reconstruction from the native stream plus the captures, a capture
    row at sample 0 exactly when the session baseline is applied, and, with
    ``raw_of_seq`` (the sensor's true raw frame for a ``seq``), the reconstruction
    equal to the truth to within float32 rounding.
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

        rows = read_baseline_episode(baseline_path(root, instance_id, e))
        assert rows.episode_index == e and rows.shape == stream.shape
        assert record.baseline_stream is not None
        period = record.baseline_stream.sample_every
        sampled = rows.sample_index[rows.samples]
        assert sampled.tolist() == list(range(0, len(stream), period))
        np.testing.assert_array_equal(rows.seq[rows.samples], stream.seq[sampled])
        raw = raw_frames(stream, rows)
        np.testing.assert_array_equal(rows.values[rows.samples], raw[sampled])
        applied = record.baseline is not None and record.baseline.applied
        assert bool(rows.captures.any()) == applied
        if applied:
            assert rows.captures[0] and rows.sample_index[0] == 0
            assert rows.frame_range[0] == record.baseline.frame_range
        else:
            np.testing.assert_array_equal(raw, stream.data)
        if raw_of_seq is not None:
            truth = np.stack([raw_of_seq(int(s)) for s in stream.seq])
            np.testing.assert_allclose(raw, truth, rtol=0, atol=1e-6)
            if applied:
                assert np.any(stream.data != truth)
    return dataset, info, record
