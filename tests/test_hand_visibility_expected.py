import numpy as np

from src.hand_expected_visibility import associate_expected_hands, xyxy_iou


def test_expected_hand_association_requires_same_side_and_iou_and_marks_missing_zero():
    left = np.array([10, 10, 30, 30], dtype=float)
    right = np.array([60, 10, 80, 30], dtype=float)
    regions = [
        {"box": [11, 11, 31, 31], "canonical_is_right": False, "visibility": [0.2, 0.6]},
        {"box": [60, 10, 80, 30], "canonical_is_right": True, "visibility": [0.9, 0.7]},
        {"box": [60, 10, 80, 30], "canonical_is_right": False, "visibility": [0.0, 0.0]},
    ]
    got = associate_expected_hands(regions, left, right, min_iou=0.3)
    assert got["missing_left"] == 0
    assert got["missing_right"] == 0
    assert got["vis_left"] == 0.4
    assert got["vis_right"] == 0.8

    unmatched = associate_expected_hands([], left, np.full(4, np.nan), min_iou=0.3)
    assert unmatched["vis_left"] == 0.0
    assert unmatched["missing_left"] == 1
    assert np.isnan(unmatched["vis_right"])
    assert np.isnan(unmatched["missing_right"])


def test_xyxy_iou_is_zero_for_nonoverlap_and_one_for_same_box():
    assert xyxy_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert xyxy_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
