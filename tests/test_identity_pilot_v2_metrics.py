import numpy as np
from scripts.chain_tests.identity_pilot_v2_metrics import laplacian256, v2_ratios

def test_laplacian256_is_finite_for_non_square_crop():
    image=np.tile(np.arange(40,dtype=np.uint8),(20,1))
    assert np.isfinite(laplacian256(image))

def test_v2_ratios_use_donor_clean_and_own_clean_separately():
    values=v2_ratios(np.array([2.,2.]),np.array([1.,1.]),np.array([4.,4.]))
    assert values=={"donor_clean_ratio":2.,"own_clean_ratio":.5}
