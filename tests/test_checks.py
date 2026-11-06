"""The integrity parts and the array checks on synthetic streams with hand-computed answers."""

import numpy as np
import pytest

from taxel.dataset.reduce import reduce_episode
from taxel.dataset.sidecar import BaselineRows, EpisodeStream, frame_times
from taxel.doctor.checks import (
    BaselineInputs,
    drift_check,
    framerate_main_table,
    framerate_sidecar,
    gel_check_elements,
    gel_check_pixels,
    integrity_copy,
    integrity_files,
    integrity_raw,
    max_gap_threshold_ms,
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
        assert not saturation_check(data, (lo, hi), "sidecar", Thresholds(saturated=0.03)).flag

    def test_observed_extremes_with_low_confidence(self) -> None:
        data = np.random.default_rng(0).normal(0.0, 0.5, (100, 4, 2)).astype(np.float32)
        report = saturation_check(data, None, "main_table", Thresholds())
        assert report.limit == "observed" and report.confidence == "low" and report.source == "main_table"
        assert report.min == [float(data[:, :, c].min()) for c in range(2)]
        assert report.max == [float(data[:, :, c].max()) for c in range(2)]
        assert report.saturated_pct == [2 / 400, 2 / 400] and not report.flag


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
        assert not drift_check(stored, RATE, inputs, LIMITS, "sidecar", None, Thresholds(drift=0.6)).flag

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
        assert report.level_change[0] == pytest.approx(0.5 * 9.5, abs=0.05) and report.flag
        assert not drift_check(
            raw, 30.0, BaselineInputs(), None, "main_table", None, Thresholds(drift=5.0)
        ).flag
        raw[0] = 0.0
        assert drift_check(
            raw, 30.0, BaselineInputs(), None, "main_table", None, Thresholds()
        ).first_frame_zero

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


def noisy_frames(k: int = 20, h: int = 16, w: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8).astype(np.int16)
    frames = base[None] + rng.integers(-4, 5, size=(k, h, w, 3), dtype=np.int16)
    return np.clip(frames, 0, 255).astype(np.uint8)


class TestGel:
    def test_dead_pixels_and_the_largest_cluster(self) -> None:
        frames = noisy_frames()
        frames[:, 2:5, 5:9, :] = 0  # a 3x4 dead block
        frames[:, 12:14, 1:3, :] = 255  # a 2x2 dead block elsewhere
        report = gel_check_pixels(frames, Thresholds())
        assert report.kind == "pixels" and report.frames_used == 20
        assert report.dead_pct == pytest.approx(16 / 320) and report.largest_cluster_px == 12
        assert (
            report.dead_elements is None and not report.flag
        )  # 5% dead and 12 pixels are under the defaults
        assert gel_check_pixels(frames, Thresholds(cluster_px=10)).flag
        assert gel_check_pixels(frames, Thresholds(gel=0.04)).flag
        assert gel_check_pixels(noisy_frames(), Thresholds()).dead_pct == 0.0

    def test_pixels_skipped_without_variation(self) -> None:
        frames = np.repeat(noisy_frames(1), 5, axis=0)
        skipped = gel_check_pixels(frames, Thresholds())
        assert isinstance(skipped, Skipped) and "do not vary" in skipped.skipped
        assert isinstance(gel_check_pixels(noisy_frames(1), Thresholds()), Skipped)

    def test_dead_elements_are_listed(self) -> None:
        data = np.random.default_rng(0).normal(0.0, 0.01, (100, 8, 2)).astype(np.float32)
        data[:, 3, :] = 0.0
        data[:, 6, :] = -1.0
        report = gel_check_elements(data, Thresholds())
        assert report.kind == "elements" and report.dead_elements == [3, 6]
        assert report.dead_pct == 0.25 and report.largest_cluster_px is None and report.frames_used == 100
        assert report.flag
        clean = gel_check_elements(np.random.default_rng(1).normal(size=(100, 8, 2)), Thresholds())
        assert clean.dead_elements == [] and clean.dead_pct == 0.0 and not clean.flag
        assert isinstance(gel_check_elements(np.zeros((100, 8, 2)), Thresholds()), Skipped)


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
