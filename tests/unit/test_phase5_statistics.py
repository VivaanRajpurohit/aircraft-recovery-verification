import math
from aircraft_recovery.analysis.statistics import paired_bootstrap_difference, wilson_interval

def test_wilson_interval_is_bounded_and_contains_half():
    low, high = wilson_interval(5, 10)
    assert 0 <= low < .5 < high <= 1

def test_paired_bootstrap_is_reproducible():
    a=[2,4,6,8]; b=[1,2,3,4]
    first=paired_bootstrap_difference(a,b,7,500)
    second=paired_bootstrap_difference(a,b,7,500)
    assert first == second
    assert first["n_pairs"] == 4 and math.isclose(first["mean_difference"],2.5)
