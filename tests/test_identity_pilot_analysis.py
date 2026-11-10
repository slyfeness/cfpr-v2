import math

from scripts.chain_tests.analyze_identity_pilot import is_candidate


def test_candidate_requires_all_four_pre_registered_conditions_for_both_people():
    metrics = {
        "CRB100_001": {"donor_rise": .15, "own_drop": .30, "sharpness_ratio": .80, "outside_own_delta": -.02},
        "CRB100_025": {"donor_rise": .16, "own_drop": .31, "sharpness_ratio": .90, "outside_own_delta": .01},
    }
    assert is_candidate(metrics) is True


def test_candidate_rejects_nan_and_one_person_failure_without_relaxing_any_threshold():
    metrics = {
        "CRB100_001": {"donor_rise": .20, "own_drop": .31, "sharpness_ratio": .80, "outside_own_delta": .02},
        "CRB100_025": {"donor_rise": .20, "own_drop": .31, "sharpness_ratio": math.nan, "outside_own_delta": .01},
    }
    assert is_candidate(metrics) is False
