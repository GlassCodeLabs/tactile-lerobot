"""Window statistics and no-contact detection on synthetic streams with known answers."""

import numpy as np
import pytest

from taxel.doctor.windows import no_contact, window_edges, window_stats

RATE = 100.0


def ramp_stream(n: int = 200, slope: float = 0.5, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """``(n, 2, 1)``: a rest pattern plus ``slope`` per second on both elements, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    rest = np.array([[0.1], [-0.2]])
    return rest[None] + slope * t[:, None, None] + rng.normal(0.0, noise, (n, 2, 1))


class TestWindowEdges:
    def test_consecutive_windows_drop_a_short_tail(self) -> None:
        starts, ends = window_edges(100, 30)
        assert starts.tolist() == [0, 30, 60] and ends.tolist() == [30, 60, 90]

    def test_a_tail_of_half_a_window_is_kept(self) -> None:
        starts, ends = window_edges(105, 30)
        assert ends.tolist() == [30, 60, 90, 105]
        assert window_edges(16, 30)[1].tolist() == [16]
        assert window_edges(14, 30)[0].size == 0

    def test_windows_restart_at_breaks(self) -> None:
        starts, ends = window_edges(100, 30, breaks=[45])
        assert starts.tolist() == [0, 30, 45, 75] and ends.tolist() == [30, 45, 75, 100]
        assert window_edges(100, 30, breaks=[0, 100, 150])[0].tolist() == [0, 30, 60]

    def test_window_must_hold_two_samples(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            window_edges(10, 1)


class TestWindowStats:
    def test_detrended_spread_ignores_a_ramp(self) -> None:
        data = ramp_stream()
        stats = window_stats(data, *window_edges(200, 50))
        assert len(stats) == 4 and stats.quiet.all()
        np.testing.assert_allclose(stats.spread, 0.01, atol=0.004)
        np.testing.assert_allclose(stats.noise, 0.01, atol=0.004)
        np.testing.assert_allclose(stats.centres(RATE), [0.25, 0.75, 1.25, 1.75])
        np.testing.assert_allclose(
            stats.level[:, :, 0], [[0.1 + 0.5 * c, -0.2 + 0.5 * c] for c in stats.centres(RATE)], atol=0.01
        )

    def test_an_onset_window_is_not_quiet_but_the_plateau_is(self) -> None:
        data = ramp_stream()
        data[110:, 0] += 2.0  # a press on element 0 held to the end
        stats = window_stats(data, *window_edges(200, 50))
        assert stats.quiet.tolist() == [True, True, False, True]
        assert stats.level[3, 0, 0] > stats.level[1, 0, 0] + 1.5

    def test_noise_free_data_is_quiet(self) -> None:
        stats = window_stats(ramp_stream(noise=0.0).astype(np.float32), *window_edges(200, 50))
        assert stats.quiet.all() and float(stats.noise.max()) < 1e-6


class TestNoContact:
    def test_plateau_is_contact_and_the_line_is_the_ramp(self) -> None:
        t = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        y = 0.5 * t[:, None, None] + np.array([[0.1], [-0.2]])[None]
        y[4, 0, 0] += 2.0  # a held press on element 0 in the fifth window
        noise = np.full((2, 1), 0.01)
        mask, slope, intercept = no_contact(y, t, np.ones(6, bool), noise)
        assert mask.tolist() == [True, True, True, True, False, True]
        np.testing.assert_allclose(slope, 0.5, atol=1e-9)
        np.testing.assert_allclose(intercept[:, 0], [0.1, -0.2], atol=1e-9)

    def test_non_quiet_windows_stay_out_and_do_not_bias_the_line(self) -> None:
        t = np.arange(8) * 0.5 + 0.25
        y = 0.5 * t[:, None, None] + np.zeros((1, 1))[None]
        y[2] = 50.0  # an onset window: not quiet, whatever its level
        quiet = np.ones(8, bool)
        quiet[2] = False
        mask, slope, _ = no_contact(y, t, quiet, np.full((1, 1), 0.01))
        assert mask.tolist() == [True, True, False, True, True, True, True, True]
        np.testing.assert_allclose(slope, 0.5, atol=1e-9)

    def test_fewer_than_two_quiet_windows(self) -> None:
        y = np.arange(3, dtype=float)[:, None, None]
        t = np.arange(3, dtype=float)
        mask, slope, intercept = no_contact(y, t, np.zeros(3, bool), np.zeros((1, 1)))
        assert not mask.any() and slope.tolist() == [[0.0]] and np.isnan(intercept).all()
        quiet = np.array([False, True, False])
        mask, slope, intercept = no_contact(y, t, quiet, np.zeros((1, 1)))
        assert (
            mask.tolist() == [False, True, False]
            and slope.tolist() == [[0.0]]
            and intercept.tolist() == [[1.0]]
        )

    def test_many_windows_subsample_pairs_deterministically(self) -> None:
        t = np.arange(400) * 0.5
        y = 0.5 * t[:, None, None] + np.zeros((1, 1))[None]
        first = no_contact(y, t, np.ones(400, bool), np.full((1, 1), 0.01))
        second = no_contact(y, t, np.ones(400, bool), np.full((1, 1), 0.01))
        assert first[0].all() and np.array_equal(first[1], second[1])
        np.testing.assert_allclose(first[1], 0.5, atol=1e-9)

    def test_half_the_windows_pressed_still_picks_the_rest_level(self) -> None:
        t = np.arange(6) * 0.5 + 0.25
        y = 0.5 * t[:, None, None] + np.zeros((2, 1))[None]
        y[3:, 0] += 2.0  # a press held through the second half of the episode
        mask, slope, intercept = no_contact(y, t, np.ones(6, bool), np.full((2, 1), 0.01))
        assert mask.tolist() == [True, True, True, False, False, False]
        np.testing.assert_allclose(slope, 0.5, atol=1e-9)
        np.testing.assert_allclose(intercept, 0.0, atol=1e-9)
        y[:3, 0] += 4.0  # the other way round: the lower level is the rest, whichever half holds it
        mask, _, _ = no_contact(y, t, np.ones(6, bool), np.full((2, 1), 0.01))
        assert mask.tolist() == [False, False, False, True, True, True]
