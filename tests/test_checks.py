"""The integrity parts and the array checks on synthetic streams with hand-computed answers."""

import numpy as np
import pytest

from taxel.dataset.reduce import reduce_episode
from taxel.dataset.sidecar import BaselineRows, EpisodeStream, frame_times
from taxel.doctor.checks import (
    ENCODER_FLOOR,
    NAN_FLOOR,
    BaselineInputs,
    drift_check,
    framerate_main_table,
    framerate_sidecar,
    gel_check_elements,
    gel_check_pixels,
    gel_dead_across,
    held_check,
    integrity_copy,
    integrity_files,
    integrity_raw,
    max_gap_threshold_ms,
    name_groups,
    nan_fraction,
    pixel_stats,
    saturation_check,
    skew_check,
)
from taxel.schema import (
    Baseline,
    Clock,
    DriftReport,
    FrameCounts,
    SessionRecord,
    Skipped,
    Thresholds,
)

RATE = 100.0
REST = np.array([[0.1], [-0.2]], dtype=np.float32)  # two elements, one channel


def stream(data: np.ndarray, host_ts=None, seq=None, episode_index: int = 0) -> EpisodeStream:
    n = len(data)
    host_ts = np.arange(n) / RATE + 1000.0 if host_ts is None else np.asarray(host_ts, dtype=np.float64)
    seq = np.arange(n, dtype=np.int64) if seq is None else np.asarray(seq, dtype=np.int64)
    return EpisodeStream(episode_index, host_ts, host_ts.copy(), seq, np.asarray(data, dtype=np.float32))


def baseline_rows(
    captures: list[tuple[int, np.ndarray]], samples: list[tuple[int, np.ndarray]], episode_index=0
):
    rows = sorted(
        [("capture", i, v) for i, v in captures] + [("sample", i, v) for i, v in samples],
        key=lambda r: (r[1], r[0]),
    )
    return BaselineRows(
        episode_index,
        np.array([r[0] for r in rows], dtype=str),
        np.array([r[1] for r in rows], dtype=np.int64),
        np.array([r[1] for r in rows], dtype=np.int64),
        np.array([1000.0 + r[1] / RATE for r in rows]),
        [None] * len(rows),
        np.stack([r[2] for r in rows]).astype(np.float32),
    )


def drifting_raw(
    n: int = 1000, slope: float = 0.5, noise: float = 0.01, seed: int = 0, rate: float = RATE
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    return (REST[None] + slope * t[:, None, None] + rng.normal(0.0, noise, (n, 2, 1))).astype(np.float32)


class TestFramerate:
    def test_sidecar_gaps_drops_and_histogram(self) -> None:
        periods = np.full(99, 0.01)
        periods[40], periods[70] = 0.03, 0.02  # two frames lost, then one
        host_ts = np.concatenate([[1000.0], 1000.0 + np.cumsum(periods)])
        seq = np.concatenate([np.arange(41), np.arange(43, 73), np.arange(74, 103)])
        counters = FrameCounts(received=100, dropped=3, overflowed=0)
        report = framerate_sidecar(host_ts, seq, RATE, counters, Thresholds())
        assert report.source == "sidecar" and report.timestamps == "measured" and report.frames == counters
        assert report.declared_hz == 100.0 and abs(report.effective_hz - 99 / 1.02) < 1e-6
        assert report.gap_histogram.edges_periods == [1.5, 2.5, 3.5]
        assert report.gap_histogram.counts == [97, 1, 1, 0]
        assert abs(report.max_gap_ms - 30.0) < 1e-6 and report.max_gap_threshold_ms == 100.0
        assert abs(report.dropped_pct - 3 / 103) < 1e-9 and report.flag
        assert not framerate_sidecar(host_ts, seq, RATE, None, Thresholds(dropped=0.05)).flag
        assert framerate_sidecar(host_ts, seq, RATE, None, Thresholds(dropped=0.05, max_gap_ms=25.0)).flag
        assert framerate_sidecar(host_ts, None, RATE, None, Thresholds()).dropped_pct is None

    def test_main_table_nominal_timestamps_are_named(self) -> None:
        nominal = (np.arange(30) / 30).astype(np.float32).astype(np.float64)
        report = framerate_main_table(nominal, 30.0, Thresholds())
        assert report.source == "main_table" and report.timestamps == "nominal"
        assert report.dropped_pct is None and report.frames is None and not report.flag
        assert abs(report.effective_hz - 30.0) < 1e-3 and report.max_gap_threshold_ms == pytest.approx(
            1000 / 3
        )
        measured = nominal + np.random.default_rng(1).normal(0.0, 0.002, 30)
        measured.sort()
        assert framerate_main_table(measured, 30.0, Thresholds()).timestamps == "measured"
        assert framerate_main_table(np.array([0.0, 0.5]), 30.0, Thresholds()).flag  # a 500 ms gap at 30 fps

    def test_zero_length_steps_are_stalled_not_dropped(self) -> None:
        host_ts = 1000.0 + np.arange(100) / RATE
        host_ts[[20, 40, 41]] = host_ts[[19, 39, 39]]  # the clock did not advance on three steps
        host_ts[42:] += 0.0  # and caught up after them
        report = framerate_sidecar(np.sort(host_ts), np.arange(100), RATE, None, Thresholds())
        assert report.stalled == 3 and report.dropped_pct == 0.0
        assert framerate_sidecar(1000.0 + np.arange(100) / RATE, None, RATE, None, Thresholds()).stalled == 0
        table = framerate_main_table(np.repeat(np.arange(15) / 15.0, 2), 30.0, Thresholds())
        assert table.stalled == 15 and table.timestamps == "measured"

    def test_a_clock_column_is_measured_and_named(self) -> None:
        rng = np.random.default_rng(0)
        steps = np.full(299, 1 / 30) + rng.normal(0.0, 0.002, 299)
        steps[[50, 120]] = 0.052  # two steps over 1.5 periods
        clock = 1.7e9 + np.concatenate([[0.0], np.cumsum(steps)])  # epoch seconds, as a recorder writes them
        report = framerate_main_table(clock, 30.0, Thresholds(), clock="source_timestamp")
        assert report.clock == "source_timestamp" and report.timestamps == "measured"
        assert report.gap_histogram.counts[1] == 2 and report.max_gap_ms == pytest.approx(52.0)
        assert report.effective_hz == pytest.approx(299 / float(clock[-1] - clock[0]))
        # Evenly spaced at the dataset period from its first value is nominal, wherever it starts.
        offset = 5.0 + np.arange(300) / 30.0
        assert framerate_main_table(offset, 30.0, Thresholds(), clock="t").timestamps == "nominal"
        assert framerate_main_table(np.arange(300) / 30.0, 30.0, Thresholds()).clock is None

    def test_max_gap_threshold(self) -> None:
        assert max_gap_threshold_ms(Thresholds(), 500.0) == 20.0
        assert max_gap_threshold_ms(Thresholds(max_gap_ms=7.5), 500.0) == 7.5


class TestSaturation:
    def test_declared_limits_count_samples_at_or_beyond(self) -> None:
        data = np.random.default_rng(0).normal(0.0, 0.5, (100, 4, 2)).astype(np.float32)
        data[:10, 0, 0] = 3.0  # 10 of 400 samples of channel 0 pinned at the limit
        data[:4, 1, 1] = -3.5  # 4 of 400 beyond the lower limit on channel 1
        lo, hi = np.array([-3.0, -3.0]), np.array([3.0, 3.0])
        report = saturation_check(data, (lo, hi), "sidecar", Thresholds())
        assert report.limit == "declared" and report.confidence == "declared" and report.source == "sidecar"
        assert report.min == [-3.0, -3.0] and report.max == [3.0, 3.0]
        assert report.saturated_pct == [0.025, 0.01] and report.flag
        assert report.at_min_pct == [0.0, 0.01] and report.at_max_pct == [0.025, 0.0]
        assert not saturation_check(data, (lo, hi), "sidecar", Thresholds(saturated=0.03)).flag
        pinned = np.full((50, 4, 1), 3.0, dtype=np.float32)  # every sample at a declared ceiling
        pinned[:25] = 0.0
        report = saturation_check(pinned, (np.array([-3.0]), np.array([3.0])), "sidecar", Thresholds())
        assert report.saturated_pct == [0.5] and report.at_max_pct == [0.5] and report.flag

    def test_observed_extremes_are_reported_and_never_flagged(self) -> None:
        data = np.random.default_rng(0).normal(0.0, 0.5, (100, 4, 2)).astype(np.float32)
        report = saturation_check(data, None, "main_table", Thresholds())
        assert report.limit == "observed" and report.confidence == "low" and report.source == "main_table"
        assert report.min == [float(data[:, :, c].min()) for c in range(2)]
        assert report.max == [float(data[:, :, c].max()) for c in range(2)]
        assert report.saturated_pct == [2 / 400, 2 / 400] and not report.flag
        assert report.at_min_pct == [1 / 400, 1 / 400] and report.at_max_pct == [1 / 400, 1 / 400]
        # A 0/1 contact flag, or a stream that rests at zero: at an observed extreme in nearly every
        # sample, which says where the data rests, not that the sensor clips.
        contact = np.random.default_rng(1).integers(0, 2, (100, 17, 1)).astype(np.float32)
        contact[:90] = 0.0
        report = saturation_check(contact, None, "main_table", Thresholds())
        assert report.saturated_pct == [1.0] and report.at_min_pct[0] > 0.9 and not report.flag
        # The same values against a declared range still flag.
        assert saturation_check(contact, (np.array([0.0]), np.array([1.0])), "main_table", Thresholds()).flag


class TestIntegrity:
    def test_raw_samples_against_the_reconstruction(self) -> None:
        raw = drifting_raw(300)
        stored = raw - REST[None]
        # Sample rows hold reading.raw, which the writer computes as stored plus the baseline in force.
        rows = baseline_rows([(0, REST)], [(i, stored[i] + REST) for i in (0, 100, 200)])
        report = integrity_raw(stream(stored), rows)
        assert report.samples == 3 and report.mismatches == 0 and report.first_mismatch is None
        assert report.max_abs_diff == 0.0 and not report.flag
        tampered = stored.copy()
        tampered[100, 0, 0] += 1.0
        report = integrity_raw(stream(tampered), rows)
        assert report.mismatches == 1 and report.first_mismatch == 100 and report.flag
        assert abs(report.max_abs_diff - 1.0) < 1e-6
        assert integrity_raw(stream(stored), baseline_rows([(0, REST)], [])).samples == 0

    def test_copy_against_the_recomputed_reduction(self) -> None:
        native = stream(drifting_raw(200))
        origin = float(native.host_ts[0]) + 0.1
        targets = frame_times(origin, 30, 30.0)
        frames = reduce_episode("mean", native.host_ts, native.data, targets, 33.333)
        from taxel.schema import Reduction

        report = integrity_copy(frames, native, origin, 30.0, Reduction("mean", 33.333))
        assert report.frames == 30 and report.mismatches == 0 and not report.flag
        tampered = frames.copy()
        tampered[7, 1, 0] += 0.5
        report = integrity_copy(tampered, native, origin, 30.0, Reduction("mean", 33.333))
        assert report.mismatches == 1 and report.first_mismatch == 7 and report.flag
        last = reduce_episode("last", native.host_ts, native.data, targets, 1000 / 30)
        assert not integrity_copy(last, native, origin, 30.0, None).flag
        assert integrity_copy(frames, native, origin, 30.0, None).flag  # mean copy checked as last
        early = integrity_copy(frames, native, origin - 10.0, 30.0, Reduction("mean", 33.333))
        assert early.flag and early.mismatches == 30  # no native sample before the first frame

    def test_files_consistency(self) -> None:
        applied = Baseline("2026-11-05T10:00:00Z", 0.2, True, (0, 99), REST.tolist())
        session = SessionRecord(
            "s",
            (0, 1),
            None,
            None,
            applied,
            clock=Clock("2026-11-05T10:00:00Z", 1000.0, [1001.0, 1003.0]),
            frames=FrameCounts(received=700, dropped=3, overflowed=0),
        )
        rows = baseline_rows([(0, REST)], [(0, REST)])
        ok = integrity_files(True, rows, session, 0, 600, 2)
        assert ok.to_dict() == {
            "native": True,
            "baseline": True,
            "capture_at_zero": True,
            "origins": True,
            "counters": True,
            "flag": False,
        }
        assert integrity_files(False, rows, session, 0, None, None).flag
        missing = integrity_files(True, None, session, 0, 600, 2)
        assert not missing.baseline and missing.capture_at_zero is False and missing.flag
        assert integrity_files(True, rows, session, 0, 800, 2).counters is False
        assert integrity_files(True, rows, session, 0, 600, 5).counters is False
        no_counts = SessionRecord("s", (0, 1), None, None, applied, clock=session.clock, frames=None)
        assert integrity_files(True, rows, no_counts, 0, 600, 2).counters is None
        assert integrity_files(True, rows, session, 0, None, None).counters is None
        one_origin = SessionRecord(
            "s",
            (0, 1),
            None,
            None,
            applied,
            clock=Clock("2026-11-05T10:00:00Z", 1000.0, [1001.0, 1003.0]),
            frames=session.frames,
        )
        assert integrity_files(True, rows, one_origin, 1, 600, 2).origins
        not_applied = SessionRecord(
            "s", (0, 1), None, None, Baseline("2026-11-05T10:00:00Z", 0.2, False, (0, 99), REST.tolist())
        )
        assert (
            integrity_files(True, baseline_rows([], [(0, REST)]), not_applied, 0, 600, 2).capture_at_zero
            is None
        )
        assert integrity_files(True, rows, not_applied, 0, 600, 2).capture_at_zero is False
        unknown = SessionRecord("s", (0, 1), None, None, None)
        assert integrity_files(True, baseline_rows([], [(0, REST)]), unknown, 0, 600, 2).origins is False


LIMITS = (np.array([-5.0]), np.array([5.0]))


def applied_inputs(raw: np.ndarray, every: int = 50, captures=None) -> tuple[np.ndarray, BaselineInputs]:
    """Stored frames and baseline inputs for ``raw`` with the baseline ``REST`` applied at sample 0."""
    captures = captures or [(0, REST)]
    index = np.array([i for i, _ in captures], dtype=np.int64)
    values = np.stack([v for _, v in captures]).astype(np.float32)
    stored = raw.copy()
    order = np.searchsorted(index, np.arange(len(raw)), side="right") - 1
    stored -= values[order]
    sampled = np.arange(0, len(raw), every, dtype=np.int64)
    samples = (sampled, (stored[sampled] + values[order[sampled]]).astype(np.float32))  # reading.raw
    host_ts = 1000.0 + index / RATE
    return stored, BaselineInputs((index, host_ts, values), REST, samples, [0, 99], 999.0)


class TestDrift:
    def test_ramp_against_the_capture_from_samples(self) -> None:
        raw = drifting_raw()
        stored, inputs = applied_inputs(raw)
        report = drift_check(stored, RATE, inputs, LIMITS, "sidecar", 1000.5, Thresholds())
        assert isinstance(report, DriftReport)
        assert report.windows == 20 and report.no_contact_windows == 20
        assert report.baseline == "capture" and report.source == "samples" and report.frame_range == [0, 99]
        assert report.baseline_age_s == pytest.approx(1.5)
        assert report.slope_per_s[0] == pytest.approx(0.5, abs=0.02)
        assert report.offset[0] == pytest.approx(
            0.5 * 9.5, abs=0.05
        )  # the line at the last no-contact window
        assert report.offset_pct_range[0] == pytest.approx(0.5 * 9.5 / 10, abs=0.005)
        assert report.no_contact_level[0] == pytest.approx(float(REST.mean()), abs=0.02)  # sample 0
        assert report.level_change[0] == pytest.approx(0.5 * 9.5, abs=0.05)  # sample 950 less sample 0
        assert report.rebaselines == [] and not report.first_frame_zero and report.flag
        assert report.confidence == "normal"
        assert not drift_check(stored, RATE, inputs, LIMITS, "sidecar", None, Thresholds(drift=0.6)).flag
        # Without a range the fraction has no meaning: only a caller's native-unit threshold flags.
        native = drift_check(stored, RATE, inputs, None, "sidecar", None, Thresholds())
        assert native.offset_pct_range is None and native.confidence == "normal" and not native.flag
        assert drift_check(stored, RATE, inputs, None, "sidecar", None, Thresholds(drift_units=1.0)).flag
        assert not drift_check(stored, RATE, inputs, None, "sidecar", None, Thresholds(drift_units=10.0)).flag

    def test_windows_without_a_sample_row_fall_back_to_raw_frames(self) -> None:
        stored, inputs = applied_inputs(drifting_raw(), every=100)
        report = drift_check(stored, RATE, inputs, LIMITS, "sidecar", None, Thresholds())
        assert report.source == "raw_frames" and report.baseline_age_s is None
        assert report.slope_per_s[0] == pytest.approx(0.5, abs=0.02)

    def test_a_held_press_is_contact_not_drift(self) -> None:
        raw = drifting_raw()
        raw[300:420, 0] += 2.0  # element 0 pressed for 1.2 s, spanning two whole windows
        stored, inputs = applied_inputs(raw)
        report = drift_check(stored, RATE, inputs, LIMITS, "sidecar", None, Thresholds())
        assert report.windows == 20 and report.no_contact_windows == 17
        assert report.slope_per_s[0] == pytest.approx(0.5, abs=0.02)
        assert report.offset[0] == pytest.approx(0.5 * 9.5, abs=0.05)

    def test_a_rebaseline_is_an_event_and_drift_restarts_from_it(self) -> None:
        raw = drifting_raw()
        rezeroed = (REST + np.array([[2.5], [2.5]], dtype=np.float32)).astype(np.float32)
        stored, inputs = applied_inputs(raw, captures=[(0, REST), (500, rezeroed)])
        report = drift_check(stored, RATE, inputs, LIMITS, "sidecar", None, Thresholds())
        assert np.all(np.abs(stored[500]) < 0.1) and report.windows == 20 and report.no_contact_windows == 20
        assert len(report.rebaselines) == 1
        event = report.rebaselines[0]
        assert event.sample_index == 500 and event.host_ts == pytest.approx(1005.0)
        assert event.delta == [pytest.approx(2.5, abs=1e-6)]
        assert report.slope_per_s[0] == pytest.approx(0.5, abs=0.02)
        assert report.offset[0] == pytest.approx(0.5 * 9.5 - 2.5, abs=0.05)
        assert report.level_change[0] == pytest.approx(0.5 * 9.5, abs=0.05)  # the raw level keeps climbing

    def test_session_baseline_without_subtraction(self) -> None:
        raw = drifting_raw()
        inputs = BaselineInputs(None, REST, (np.arange(0, 1000, 50), raw[::50]), None, None)
        report = drift_check(raw, RATE, inputs, LIMITS, "sidecar", None, Thresholds())
        assert report.baseline == "session" and report.frame_range is None and report.source == "samples"
        assert report.offset[0] == pytest.approx(0.5 * 9.5, abs=0.05)

    def test_main_table_without_a_baseline(self) -> None:
        raw = drifting_raw(300, rate=30.0)  # 10 s at 30 fps, as a main-table copy
        report = drift_check(raw, 30.0, BaselineInputs(), None, "main_table", None, Thresholds())
        assert report.baseline == "none" and report.source == "main_table"
        assert isinstance(report.offset, Skipped) and report.offset.skipped == "no session baseline"
        assert report.offset_pct_range is None and report.frame_range is None
        assert report.level_change[0] == pytest.approx(0.5 * 9.5, abs=0.05)
        assert report.confidence == "low" and not report.flag  # no range, no baseline: numbers only
        assert drift_check(
            raw, 30.0, BaselineInputs(), None, "main_table", None, Thresholds(drift_units=1.0)
        ).flag
        with_range = drift_check(raw, 30.0, BaselineInputs(), LIMITS, "main_table", None, Thresholds())
        assert with_range.confidence == "normal" and with_range.flag  # level change over 5% of the span
        raw[0] = 0.0
        zero = drift_check(raw, 30.0, BaselineInputs(), None, "main_table", None, Thresholds())
        assert zero.first_frame_zero and not zero.flag

    def test_frame_range_words_pass_through(self) -> None:
        stored, inputs = applied_inputs(drifting_raw())
        for word in ("not recorded", "no seq"):
            report = drift_check(
                stored,
                RATE,
                BaselineInputs(inputs.captures, REST, inputs.samples, word, None),
                LIMITS,
                "sidecar",
                None,
                Thresholds(),
            )
            assert report.frame_range == word

    def test_skipped_when_too_short(self) -> None:
        stored, inputs = applied_inputs(drifting_raw(60))
        short = drift_check(stored, RATE, inputs, LIMITS, "sidecar", None, Thresholds())
        assert isinstance(short, Skipped) and short.skipped.startswith(
            "episode shorter than two drift windows"
        )


def noisy_frames(k: int = 20, h: int = 16, w: int = 20, seed: int = 0, noise: int = 12) -> np.ndarray:
    """A fixed random image plus uniform integer noise in ``[-noise, noise]`` per channel and frame. The
    channel mean then varies by ``((2 noise + 1)² - 1) / 36`` counts²: about 17 at the default, 0.7 at 2."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8).astype(np.int16)
    frames = base[None] + rng.integers(-noise, noise + 1, size=(k, h, w, 3), dtype=np.int16)
    return np.clip(frames, 0, 255).astype(np.uint8)


class TestGel:
    def test_dead_pixels_and_the_largest_cluster(self) -> None:
        frames = noisy_frames(h=60, w=80)  # 4,800 pixels
        frames[:, 2:5, 5:9, :] = 0  # a 3x4 dead block
        frames[:, 12:14, 1:3, :] = 255  # a 2x2 dead block elsewhere
        stats = pixel_stats(frames)
        assert stats.frames_used == 20 and stats.area == 4800 and 14 < stats.median_variance < 21
        assert stats.dead_pct == pytest.approx(16 / 4800) and stats.largest_cluster_px == 12
        report = gel_check_pixels(stats, Thresholds())
        assert report.kind == "pixels" and report.frames_used == 20 and report.dead_elements is None
        assert report.dead_pct == pytest.approx(16 / 4800) and report.largest_cluster_px == 12
        assert report.largest_cluster_pct == pytest.approx(12 / 4800)
        assert report.median_variance == report.rest_variance == stats.median_variance
        assert report.episodes_checked == 1 and not report.flag  # 0.33% dead, a 0.25% cluster
        assert gel_check_pixels(stats, Thresholds(cluster_pct=0.002)).flag
        assert gel_check_pixels(stats, Thresholds(cluster_px=10)).flag  # pixels replace the fraction
        assert not gel_check_pixels(stats, Thresholds(cluster_px=12, cluster_pct=0.001)).flag
        assert gel_check_pixels(stats, Thresholds(gel=0.003)).flag
        assert gel_check_pixels(pixel_stats(noisy_frames(h=60, w=80)), Thresholds()).dead_pct == 0.0

    def test_quiet_gel_is_skipped_on_the_instance_rest_level(self) -> None:
        quiet = pixel_stats(noisy_frames(noise=2))  # the channel mean varies by about 0.7 counts²
        assert quiet.median_variance < ENCODER_FLOOR and quiet.dead_pct > 0.9
        skipped = gel_check_pixels(quiet, Thresholds())
        assert isinstance(skipped, Skipped)
        assert skipped.skipped.startswith(
            f"quiet gel: median pixel variance {quiet.median_variance:.2f} counts² in the quietest of 1"
        )
        # The instance's rest level decides, not the episode's own median: a loud episode of a quiet
        # instance is skipped too, and a quiet episode of a loud instance runs.
        loud = pixel_stats(noisy_frames(h=60, w=80))
        across = gel_check_pixels(loud, Thresholds(), rest_variance=0.9, episodes_checked=12)
        assert isinstance(across, Skipped) and "0.90 counts² in the quietest of 12" in across.skipped
        ran = gel_check_pixels(quiet, Thresholds(), rest_variance=10.0, episodes_checked=3)
        assert ran.rest_variance == 10.0 and ran.episodes_checked == 3
        assert ran.median_variance == quiet.median_variance and ran.dead_pct == quiet.dead_pct
        one = gel_check_pixels(pixel_stats(noisy_frames(1)), Thresholds())
        assert one == Skipped("1 frame(s) decoded: variance needs at least two")
        still = pixel_stats(np.repeat(noisy_frames(1), 5, axis=0))
        assert (
            still.median_variance == 0.0 and "0.00 counts²" in gel_check_pixels(still, Thresholds()).skipped
        )

    def test_constant_elements_are_listed_and_dead_needs_every_episode(self) -> None:
        data = np.random.default_rng(0).normal(0.0, 0.01, (100, 8, 2)).astype(np.float32)
        data[:, 3, :] = 0.0
        data[:, 6, :] = -1.0
        first = gel_check_elements(data, Thresholds())
        assert first.kind == "elements" and first.constant_elements == [3, 6]
        # One episode cannot tell a dead element from an untouched one: dead stays undecided.
        assert first.dead_elements is None and first.dead_pct == 0.0 and not first.flag
        assert first.largest_cluster_px is None and first.frames_used == 100 and first.episodes_checked == 1
        second = np.random.default_rng(1).normal(0.0, 0.01, (80, 8, 2)).astype(np.float32)
        second[:, 3, :] = 0.0  # element 3 again; element 6 moves this time
        second[:, 1, :] = 2.0  # element 1 idle in this episode only
        blocks = gel_dead_across([first, gel_check_elements(second, Thresholds())], 8, Thresholds())
        assert [b.dead_elements for b in blocks] == [[3], [3]] and [b.episodes_checked for b in blocks] == [
            2,
            2,
        ]
        assert blocks[0].constant_elements == [3, 6] and blocks[1].constant_elements == [1, 3]
        assert all(b.dead_pct == 0.125 and b.flag for b in blocks)  # 1 of 8 is over the 5% default
        alone = gel_dead_across([first], 8, Thresholds())[0]
        assert alone.dead_elements is None and alone.dead_pct == 0.0 and not alone.flag  # one episode
        assert alone.constant_elements == [3, 6] and alone.episodes_checked == 1
        assert gel_dead_across([], 8, Thresholds()) == []
        clean = gel_check_elements(np.random.default_rng(1).normal(size=(100, 8, 2)), Thresholds())
        assert clean.constant_elements == [] and not clean.flag
        # A 0/1 feature is skipped: a constant zone of a contact flag is an untouched zone.
        contact = np.random.default_rng(2).integers(0, 2, (100, 17, 1)).astype(np.float32)
        assert gel_check_elements(contact, Thresholds()).skipped.startswith("binary feature")
        assert isinstance(gel_check_elements(np.zeros((100, 8, 2)), Thresholds()), Skipped)
        assert "does not vary" in gel_check_elements(np.full((100, 8, 2), 0.5), Thresholds()).skipped
        # Elements of different scales are all alive: the rule is each element against itself.
        mixed = np.random.default_rng(2).normal(size=(100, 3, 1)) * np.array([[100.0], [0.01], [1e-4]])[None]
        assert gel_check_elements(mixed, Thresholds()).constant_elements == []
        # An element that is almost constant, within one part in a thousand of its level, is constant.
        mixed[:, 2, 0] = 5.0 + np.random.default_rng(3).normal(0.0, 1e-4, 100)
        assert gel_check_elements(mixed, Thresholds()).constant_elements == [2]


class TestNaN:
    """NaN marks a missing reading: left out of every statistic, and an element missing more than
    half the frames is left out of the check, which says so when nothing is left."""

    @staticmethod
    def stream(n: int = 300, seed: int = 0) -> np.ndarray:
        """Three elements on one channel: noise around 1.0, a slow ramp, a NaN interval on element 1
        (20% of the frames) and element 2 NaN in 80% of them."""
        rng = np.random.default_rng(seed)
        t = np.arange(n) / RATE
        data = (1.0 + 0.5 * t[:, None, None] + rng.normal(0.0, 0.01, (n, 3, 1))).astype(np.float32)
        data[100:160, 1] = np.nan
        data[: int(0.8 * n), 2] = np.nan
        return data

    def test_nan_fraction_is_per_element(self) -> None:
        np.testing.assert_allclose(nan_fraction(self.stream()), [0.0, 0.2, 0.8])
        two_channels = np.zeros((10, 2, 2))
        two_channels[:3, 0, 1] = np.nan  # one channel is enough to make the frame missing
        np.testing.assert_allclose(nan_fraction(two_channels), [0.3, 0.0])

    def test_saturation_counts_finite_samples_of_the_kept_elements(self) -> None:
        data = self.stream()
        data[160:190, 1] = 3.0  # 30 of element 1's 240 finite frames pinned at a declared ceiling
        report = saturation_check(data, (np.array([-3.0]), np.array([3.0])), "main_table", Thresholds())
        assert report.nan_pct == pytest.approx([0.0, 0.2, 0.8])
        assert report.saturated_pct == [pytest.approx(30 / 540)] and report.flag  # elements 0 and 1
        observed = saturation_check(self.stream(), None, "main_table", Thresholds())
        assert np.isfinite(observed.min[0]) and np.isfinite(observed.max[0]) and not observed.flag
        all_out = np.full((10, 2, 1), np.nan)
        all_out[:2] = 1.0
        skipped = saturation_check(all_out, None, "main_table", Thresholds())
        assert skipped == Skipped(
            f"NaN in more than {NAN_FLOOR:.0%} of the frames on every element (the least, 80.0%): "
            "nothing left to check"
        )

    def test_gel_leaves_a_missing_element_out_and_keeps_a_constant_one(self) -> None:
        data = self.stream()
        data[:, 0] = 0.25  # constant where it is finite
        data[200:220, 0] = np.nan
        report = gel_check_elements(data, Thresholds())
        assert report.constant_elements == [0] and report.nan_pct == pytest.approx(
            [0.0667, 0.2, 0.8], abs=1e-3
        )

    def test_drift_finds_the_ramp_on_the_finite_windows(self) -> None:
        report = drift_check(self.stream(), RATE, BaselineInputs(), None, "main_table", None, Thresholds())
        assert isinstance(report, DriftReport) and report.nan_pct == pytest.approx([0.0, 0.2, 0.8])
        # Element 1 is missing through one window, which is therefore not quiet.
        assert report.windows == 6 and report.no_contact_windows == 5
        assert report.slope_per_s[0] == pytest.approx(0.5, abs=0.05)
        assert report.level_change[0] == pytest.approx(0.5 * (2.75 - 0.25), abs=0.05)  # first to last window
        dark = self.stream()
        dark[:, 1] = np.nan
        dark[::2, 1] = 1.0  # element 1 half missing, every other frame: no window has two finite samples
        dark[25::50, 0] = np.nan  # and one NaN per window on element 0
        dark[25::50, 1] = np.nan
        report = drift_check(dark, RATE, BaselineInputs(), None, "main_table", None, Thresholds())
        assert isinstance(report, DriftReport)  # every window still has element 0 on its finite samples

    def test_drift_says_why_when_nan_breaks_every_window(self) -> None:
        data = np.random.default_rng(0).normal(1.0, 0.01, (300, 2, 1)).astype(np.float32)
        data[:150, 0] = np.nan  # element 0 missing for the first half, element 1 for the second:
        data[150:, 1] = np.nan  # each is at the floor and kept, and no window has both
        report = drift_check(data, RATE, BaselineInputs(), None, "main_table", None, Thresholds())
        assert report == Skipped(
            "no no-contact window in the episode: contact throughout, or NaN on more than 10% of the "
            "element-channels of every window"
        )

    def test_integrity_matches_a_nan_reproduced_as_nan(self) -> None:
        raw = drifting_raw(300)
        raw[100:110, 1] = np.nan
        stored = raw - REST[None]
        rows = baseline_rows([(0, REST)], [(i, stored[i] + REST) for i in (0, 100, 200)])
        report = integrity_raw(stream(stored), rows)
        assert report.mismatches == 0 and not report.flag and report.max_abs_diff == 0.0
        tampered = stored.copy()
        tampered[100, 1] = 0.0  # a number where the sample row holds NaN
        assert integrity_raw(stream(tampered), rows).first_mismatch == 100
        native = stream(raw)
        origin = float(native.host_ts[0]) + 0.1
        frames = reduce_episode("mean", native.host_ts, native.data, frame_times(origin, 30, 30.0), 33.333)
        from taxel.schema import Reduction

        copy = integrity_copy(frames, native, origin, 30.0, Reduction("mean", 33.333))
        assert np.isnan(frames).any() and copy.mismatches == 0 and not copy.flag


def tips(n: int = 200, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Two fingertips of six channels each, as T-Rex names them, with noise on every frame."""
    names = [f"left_{tip}_f6_{i}" for tip in ("thumb", "index") for i in range(6)]
    data = np.random.default_rng(seed).normal(0.0, 1.0, (n, 12, 1)).astype(np.float32)
    return data, names


class TestHeld:
    def test_holds_runs_and_single_holds(self) -> None:
        data, _ = tips()
        data[10] = data[9]  # a single-frame hold
        data[20:23] = data[19]  # a run of three
        data[50] = data[49]
        report = held_check(data, "sidecar", 0.0, True, Thresholds())
        assert report.frames == 199 and report.held_pct == pytest.approx(5 / 199)
        assert report.longest_run == 3 and report.single_holds == 2 and report.groups is None
        assert report.source == "sidecar" and report.expected_pct == 0.0 and not report.flag  # 2.5% < 5%
        data[100:120] = data[99]
        assert held_check(data, "sidecar", 0.0, True, Thresholds()).flag  # 12.6% held
        assert not held_check(data, "sidecar", 0.0, False, Thresholds()).flag  # no declared rate: report only
        assert not held_check(data, "main_table", 0.5, True, Thresholds()).flag  # a slower sensor's copy

    def test_a_missing_frame_is_never_a_repeat(self) -> None:
        data, _ = tips()
        data[60:63] = np.nan  # three NaN frames: equal to each other, and still not held
        data[61, 0] = 0.0
        report = held_check(data, "main_table", 0.0, True, Thresholds())
        assert report.frames == 199 - 4 and report.held_pct == 0.0 and report.longest_run == 0
        assert report.nan_pct == pytest.approx([2 / 200] + [3 / 200] * 11)

    def test_groups_from_names_catch_a_hold_of_one_fingertip(self) -> None:
        data, names = tips()
        data[30:32, :6] = data[29, :6]  # the thumb repeats for two frames, the index does not
        groups = name_groups(names)
        assert groups == {"left_thumb_f6": list(range(6)), "left_index_f6": list(range(6, 12))}
        report = held_check(data, "main_table", 0.0, False, Thresholds(), groups)
        assert report.held_pct == 0.0 and set(report.groups) == {"left_thumb_f6", "left_index_f6"}
        thumb = report.groups["left_thumb_f6"]
        assert thumb.held_pct == pytest.approx(2 / 199) and thumb.longest_run == 2 and thumb.single_holds == 0
        assert report.groups["left_index_f6"].held_pct == 0.0

    def test_name_groups_need_two_groups_of_two(self) -> None:
        assert name_groups(["Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm"]) is None
        assert name_groups([f"taxel_{i}" for i in range(16)]) is None  # one group is the whole frame
        assert name_groups(None) is None and name_groups([]) is None
        assert name_groups(["a_0", "a_1", "b_0", "b_1", "c"]) == {"a": [0, 1], "b": [2, 3]}

    def test_skips(self) -> None:
        data, _ = tips()
        assert held_check(data[:1], "sidecar", 0.0, True, Thresholds()).skipped.startswith("1 frame(s)")
        contact = np.random.default_rng(0).integers(0, 2, (100, 17, 1)).astype(np.float32)
        assert held_check(contact, "main_table", 0.0, True, Thresholds()).skipped.startswith("binary feature")
        gone = np.full((10, 2, 1), np.nan)
        assert "nothing left to check" in held_check(gone, "main_table", 0.0, True, Thresholds()).skipped


def step_signal(times: np.ndarray, onsets: list[tuple[float, float]], level: float = 1.0) -> np.ndarray:
    signal = np.zeros(len(times))
    for start, duration in onsets:
        signal[(times >= start) & (times < start + duration)] = level
    return signal


class TestSkew:
    def test_lag_is_recovered_to_a_frame(self) -> None:
        tactile_times = np.arange(0.0, 4.0, 1 / 500)  # 500 Hz
        tactile_events = np.abs(np.diff(step_signal(tactile_times, [(1.0, 0.4), (2.5, 0.3)])))
        ref_times = np.arange(0.0, 4.0, 1 / 30)  # a camera at 30 fps, seeing the presses 40 ms later
        motion = np.abs(np.diff(step_signal(ref_times, [(1.04, 0.4), (2.54, 0.3)]), prepend=0.0))
        report = skew_check(
            tactile_times[1:], tactile_events, ref_times, motion, "observation.images.cam", Thresholds()
        )
        assert report.reference == "observation.images.cam" and report.onsets == 4  # two onsets, two releases
        assert report.lag_ms == pytest.approx(
            -1000 / 15, abs=5.0
        )  # the camera's first frame past 1.04 s is 1.067 s
        assert report.lag_ms == pytest.approx(-40.0, abs=1000 / 30)  # within one frame of the truth
        assert report.confidence > 0.5 and report.resolution_ms == pytest.approx(1000 / 30)
        assert report.flag  # 67 ms is over the 50 ms default
        assert not skew_check(
            tactile_times[1:], tactile_events, ref_times, motion, "cam", Thresholds(skew=80.0)
        ).flag

    def test_unrelated_events_are_skipped_for_low_confidence(self) -> None:
        tactile_times = np.arange(0.0, 4.0, 1 / 500)
        tactile_events = np.abs(np.diff(step_signal(tactile_times, [(1.0, 0.4)])))
        ref_times = np.arange(0.0, 4.0, 1 / 30)
        motion = np.abs(np.diff(step_signal(ref_times, [(3.0, 0.4)]), prepend=0.0))  # two seconds away
        skipped = skew_check(tactile_times[1:], tactile_events, ref_times, motion, "cam", Thresholds())
        assert isinstance(skipped, Skipped) and skipped.skipped.startswith("no onset matched")
        flat = skew_check(
            tactile_times[1:], np.zeros(len(tactile_events)), ref_times, motion, "cam", Thresholds()
        )
        assert isinstance(flat, Skipped) and "no contact onset" in flat.skipped
        assert isinstance(
            skew_check(tactile_times[1:2], tactile_events[:1], ref_times, motion, "cam", Thresholds()),
            Skipped,
        )
