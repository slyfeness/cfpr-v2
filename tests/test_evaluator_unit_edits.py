import numpy as np
import pytest

from scripts.evaluator_unit_tests.make_edits import (
    align_emotion_scores,
    apply_flicker_ev,
    assert_changed_frames,
    hard_cut_window,
    jump_frame,
    occlude_right_hand,
    replace_window,
    shift_audio_window,
    skip_forward_window,
    verify_audio_shift_xcorr,
    freeze_window,
)


def frames(n=24):
    return np.stack([np.full((3, 4, 3), i, np.uint8) for i in range(n)])


def test_replace_window_only_changes_designated_frames_and_preserves_frame_numbers():
    clean, donor = frames(), frames() + np.uint8(100)
    out, meta = replace_window(clean, donor, 4, 9)
    assert_changed_frames(clean, out, range(4, 9))
    np.testing.assert_array_equal(out[4:9], donor[4:9])
    assert meta["source_frame_map"] == {str(i): i for i in range(4, 9)}


def test_freeze_window_repeats_anchor_for_exactly_twelve_frames():
    clean = frames(30)
    out, meta = freeze_window(clean, 6, length=12)
    assert_changed_frames(clean, out, range(7, 18))
    np.testing.assert_array_equal(out[6:18], np.repeat(clean[6:7], 12, axis=0))
    assert meta["anchor"] == 6


def test_skip_forward_window_uses_t_plus_six_without_touching_exterior():
    clean = frames(30)
    out, meta = skip_forward_window(clean, 4, 12, shift=6)
    assert_changed_frames(clean, out, range(4, 12))
    np.testing.assert_array_equal(out[4:12], clean[10:18])
    assert meta["source_frame_map"] == {str(i): i + 6 for i in range(4, 12)}


def test_jump_frame_changes_only_a_to_a_plus_thirty():
    clean = frames(50)
    out, meta = jump_frame(clean, 8, jump=30)
    assert_changed_frames(clean, out, [8])
    np.testing.assert_array_equal(out[8], clean[38])
    assert meta["source_frame_map"] == {"8": 38}


def test_flicker_applies_linear_rgb_ratio_and_preserves_exterior():
    clean = np.full((12, 2, 2, 3), 128, dtype=np.uint8)
    out, meta = apply_flicker_ev(clean, 2, 10, ev=0.2)
    assert_changed_frames(clean, out, range(2, 10))
    linear = lambda x: ((x / 255.0 + 0.055) / 1.055) ** 2.4
    means = np.array([linear(out[i].astype(float)).mean() for i in range(2, 10)])
    assert means[0] / means[1] == pytest.approx(2**0.2, rel=0.05)
    assert meta["clipped_fraction"] >= 0.0


def test_hard_cut_uses_future_source_and_only_changes_window():
    clean = frames(80)
    out, meta = hard_cut_window(clean, 10, 18, offset=48)
    assert_changed_frames(clean, out, range(10, 18))
    np.testing.assert_array_equal(out[10:18], clean[58:66])
    assert meta["source_frame_map"] == {str(i): i + 48 for i in range(10, 18)}


def test_right_hand_occlusion_covers_all_valid_points_and_skips_missing_frames():
    clean = np.full((4, 100, 100, 3), 255, dtype=np.uint8)
    hands = np.full((4, 2, 21, 2), np.nan, dtype=np.float32)
    scores = np.zeros((4, 2, 21), dtype=np.float32)
    hands[1, 1] = np.array([[.40, .50], [.55, .60]] + [[.48, .55]] * 19)
    scores[1, 1] = .9
    hands[2, 1] = np.array([[.10, .10]] * 21)
    scores[2, 1] = .2
    out, meta = occlude_right_hand(clean, hands, scores, 0, 4, right_hand_index=1, margin=0.2)
    assert meta["applied_frames"] == [1]
    assert meta["skipped_frames"] == [0, 2, 3]
    x0, y0, x1, y1 = meta["boxes"]["1"]
    assert x0 <= 40 and y0 <= 50 and x1 >= 55 and y1 >= 60
    assert np.all(out[1, y0:y1 + 1, x0:x1 + 1] == 128)
    assert_changed_frames(clean, out, [1])


def test_window_audio_shift_has_expected_signed_lag_and_unchanged_exterior():
    audio = np.arange(1000, dtype=np.int16)
    out, meta = shift_audio_window(audio, 200, 800, shift_samples=41)
    np.testing.assert_array_equal(out[:200], audio[:200])
    np.testing.assert_array_equal(out[800:], audio[800:])
    np.testing.assert_array_equal(out[241:800], audio[200:759])
    assert meta["shift_samples"] == 41


@pytest.mark.parametrize("shift", [-41, 41])
def test_audio_shift_cross_correlation_recovers_the_requested_signed_lag(shift):
    rng = np.random.default_rng(20261108)
    audio = rng.integers(-12000, 12000, size=4096, dtype=np.int16)
    out, _ = shift_audio_window(audio, 512, 3584, shift_samples=shift)
    result = verify_audio_shift_xcorr(audio, out, 512, 3584, shift)
    assert result["passed"] is True
    assert result["recovered_lag_samples"] == shift


def test_emotion_alignment_uses_start_seconds_and_30_to_24_fps_linear_interpolation():
    scores = np.stack([np.arange(400, dtype=float), np.ones(400)], axis=1)
    out, indices = align_emotion_scores(scores, start_seconds=2.0, n_frames=3, src_fps=30, dst_fps=24)
    np.testing.assert_allclose(indices, [60.0, 61.25, 62.5])
    np.testing.assert_allclose(out[:, 0], [60.0, 61.25, 62.5])
    np.testing.assert_allclose(out[:, 1], 1.0)
