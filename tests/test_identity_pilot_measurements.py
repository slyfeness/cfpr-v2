import numpy as np

from scripts.chain_tests.identity_pilot_measurements import candidate_metrics


def test_candidate_metrics_uses_fixed_window_and_24_frame_exclusion_band():
    clean_own = np.ones(240)
    pilot_own = np.ones(240)
    clean_donor = np.zeros(240)
    pilot_donor = np.zeros(240)
    pilot_own[96:144] = .6
    pilot_donor[96:144] = .2
    pilot_own[:72] = .99
    pilot_own[168:] = .99
    metrics = candidate_metrics(clean_own, pilot_own, clean_donor, pilot_donor, 2., 1.)
    assert np.isclose(metrics["donor_rise"], .2)
    assert np.isclose(metrics["own_drop"], .4)
    assert np.isclose(metrics["sharpness_ratio"], .5)
    assert np.isclose(metrics["outside_own_delta"], -.01)
