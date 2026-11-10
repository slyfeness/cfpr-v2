import numpy as np

from scripts.chain_tests.analyze_identity_drift import analyze_window, dose_monotonic


def test_analyze_window_uses_registered_inside_and_exterior_ranges():
    clean_own = np.full(240, .8)
    changed_own = clean_own.copy()
    clean_donor = np.full(240, .2)
    changed_donor = clean_donor.copy()
    changed_own[96:144] = .65
    changed_donor[96:144] = .35
    result = analyze_window(clean_own, changed_own, clean_donor, changed_donor, (96, 144))
    assert result["valid_inside_frames"] == 48
    assert np.isclose(result["inside_own_delta"], -.15)
    assert np.isclose(result["inside_donor_delta"], .15)
    assert np.isclose(result["outside_own_delta"], 0.0)


def test_dose_monotonic_allows_registered_small_reverse_only():
    assert dose_monotonic([.04, .03, .14, .26])
    assert not dose_monotonic([.04, .01, .14, .26])
