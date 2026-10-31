"""The three reduction methods against hand-computed expectations, and the hold rule over an episode."""

import numpy as np
import pytest

from taxel.dataset.reduce import reduce_episode, reduce_frame

# Five samples 10 ms apart; sample i reads i everywhere, so means are easy to compute by hand.
HOST_TS = np.array([1.00, 1.01, 1.02, 1.03, 1.04])
DATA = np.stack([np.full((2, 3), float(i), dtype=np.float32) for i in range(5)])


def value(frame) -> float:
    assert frame is not None and frame.dtype == np.float32 and frame.shape == (2, 3)
    assert np.all(frame == frame.flat[0])
    return float(frame.flat[0])


class TestMean:
    def test_window_is_open_below_and_closed_above(self) -> None:
        # (1.005, 1.025] holds samples 1 and 2
        assert value(reduce_frame("mean", HOST_TS, DATA, 1.025, 20.0)) == 1.5
        # (1.00, 1.02] holds 1 and 2: the sample exactly at T - w is excluded, the one at T included
        assert value(reduce_frame("mean", HOST_TS, DATA, 1.02, 20.0)) == 1.5
        # (1.01, 1.03] holds 2 and 3
        assert value(reduce_frame("mean", HOST_TS, DATA, 1.03, 20.0)) == 2.5

    def test_mean_is_computed_in_float64(self) -> None:
        data = np.array([[[16777216.0]], [[1.0]]], dtype=np.float32)  # 2**24 and 1: float32 would lose the 1
        frame = reduce_frame("mean", [1.0, 1.001], data, 1.001, 10.0)
        assert frame is not None and frame.dtype == np.float32
        assert float(frame[0, 0]) == np.float32((16777216.0 + 1.0) / 2)

    def test_empty_window(self) -> None:
        assert reduce_frame("mean", HOST_TS, DATA, 2.0, 20.0) is None
        assert reduce_frame("mean", HOST_TS, DATA, 0.99, 20.0) is None


class TestLast:
    def test_newest_sample_at_or_before_target(self) -> None:
        assert value(reduce_frame("last", HOST_TS, DATA, 1.025, 100.0)) == 2.0
        assert value(reduce_frame("last", HOST_TS, DATA, 1.03, 100.0)) == 3.0
        assert value(reduce_frame("last", HOST_TS, DATA, 5.0, 100000.0)) == 4.0

    def test_window_bounds_how_stale_the_sample_may_be(self) -> None:
        assert reduce_frame("last", HOST_TS, DATA, 1.06, 15.0) is None
        assert value(reduce_frame("last", HOST_TS, DATA, 1.06, 25.0)) == 4.0


class TestNearest:
    def test_nearest_either_side_with_ties_to_the_earlier(self) -> None:
        assert value(reduce_frame("nearest", HOST_TS, DATA, 1.014, 20.0)) == 1.0
        assert value(reduce_frame("nearest", HOST_TS, DATA, 1.016, 20.0)) == 2.0
        assert value(reduce_frame("nearest", HOST_TS, DATA, 1.015, 20.0)) == 1.0
        assert value(reduce_frame("nearest", HOST_TS, DATA, 0.995, 20.0)) == 0.0  # looks ahead

    def test_window_is_centred(self) -> None:
        assert reduce_frame("nearest", HOST_TS, DATA, 1.06, 30.0) is None  # [1.045, 1.075] is empty
        assert value(reduce_frame("nearest", HOST_TS, DATA, 1.06, 40.0)) == 4.0


class TestValidation:
    def test_arguments(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            reduce_frame("median", HOST_TS, DATA, 1.0, 10.0)
        with pytest.raises(ValueError, match="window_ms must be a positive number"):
            reduce_frame("mean", HOST_TS, DATA, 1.0, 0.0)
        with pytest.raises(ValueError, match="one row per sample"):
            reduce_frame("mean", HOST_TS, DATA[:3], 1.0, 10.0)


class TestEpisode:
    def test_hold_rule_repeats_the_previous_frame(self) -> None:
        # Frames at 1.005, 1.055, 1.105: the second and third windows (10 ms) hold nothing.
        out = reduce_episode("mean", HOST_TS, DATA, [1.005, 1.055, 1.105], 10.0)
        assert out.shape == (3, 2, 3) and out.dtype == np.float32
        assert [value(f) for f in out] == [0.0, 0.0, 0.0]
        out = reduce_episode("last", HOST_TS, DATA, [1.035, 1.045, 1.055], 10.0)
        assert [value(f) for f in out] == [3.0, 4.0, 4.0]

    def test_first_frame_falls_back_to_the_newest_sample_before_it(self) -> None:
        out = reduce_episode("mean", HOST_TS, DATA, [1.5, 1.51], 10.0)
        assert [value(f) for f in out] == [4.0, 4.0]
        with pytest.raises(ValueError, match="no native sample at or before the first frame"):
            reduce_episode("mean", HOST_TS, DATA, [0.5], 10.0)

    def test_matches_reduce_frame_when_every_window_has_samples(self) -> None:
        targets = [1.011, 1.021, 1.031, 1.041]
        out = reduce_episode("mean", HOST_TS, DATA, targets, 20.0)
        for k, target in enumerate(targets):
            np.testing.assert_array_equal(out[k], reduce_frame("mean", HOST_TS, DATA, target, 20.0))
