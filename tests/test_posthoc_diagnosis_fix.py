import numpy as np

from scripts.evaluator_unit_tests.posthoc_diagnosis_fix import (
    classify_d1_person,
    classify_d1_window,
    best_audio_lag,
    calibrate_audio_lag,
    d1_measurement,
    lipsync_json,
)


def test_d1_window_statuses_follow_required_priority():
    assert classify_d1_window([
        {"valid_segments": 2, "abs_error": 0.5},
        {"valid_segments": 3, "abs_error": 1.0},
    ]) == "accurate"
    assert classify_d1_window([
        {"valid_segments": 1, "abs_error": float("nan")},
        {"valid_segments": 2, "abs_error": 1.5},
    ]) == "inaccurate"
    assert classify_d1_window([
        {"valid_segments": 1, "abs_error": float("nan")},
        {"valid_segments": 3, "abs_error": 0.5},
    ]) == "undetermined"


def test_d1_person_with_accurate_and_undetermined_window_is_undetermined():
    assert classify_d1_person(["accurate", "undetermined"]) == "undetermined"


def test_audio_delay_calibration_reports_positive_lag_for_delayed_audio():
    visual = np.random.default_rng(7).normal(size=64)
    delayed_audio = np.pad(visual, (5, 0))[: len(visual)]
    lag, correlation, _ = best_audio_lag(visual, delayed_audio, max_lag=10)
    assert lag == 5
    assert correlation > 0.99


def test_lipsync_json_reads_a_lipsync_directory(tmp_path):
    directory = tmp_path / "lipsync"
    directory.mkdir()
    (directory / "gen_lipsync.json").write_text('{"av_offset_frames": 0}')
    assert lipsync_json(directory)["av_offset_frames"] == 0


def test_d1_measurement_can_be_classified_by_d1_window_rule():
    clean = {"seg_offsets": [0, 0], "seg_conf": [3, 3]}
    edited = {"seg_offsets": [-4, -4]}
    measurement = d1_measurement(clean, edited, np.asarray([True, True]), 3, 4)
    assert classify_d1_window([measurement]) == "accurate"


def test_audio_lag_calibration_uses_the_audio_envelope_itself():
    envelope = np.random.default_rng(11).normal(size=80)
    calibration = calibrate_audio_lag(envelope)
    assert calibration["base_lag_24fps"] == 0
    assert calibration["detected_change_24fps"] == 5
