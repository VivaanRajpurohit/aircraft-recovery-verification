from __future__ import annotations

from aircraft_recovery.analysis.v2_1b_final import paired_binary_effect


def test_paired_binary_effect_preserves_discordant_counts_and_seed() -> None:
    result = paired_binary_effect([1, 1, 0, 0], [1, 0, 1, 0], seed=19001, resamples=1000)
    assert result["pair_count"] == 4
    assert result["both_recovered"] == 1
    assert result["monitored_only_recovered"] == 1
    assert result["unmonitored_only_recovered"] == 1
    assert result["neither_recovered"] == 1
    assert result["discordant_pair_count"] == 2
    assert result["absolute_rate_difference"] == 0.0
    assert result["bootstrap_seed"] == 19001
    assert result["hypothesis_test"] == "none_not_preregistered"
    assert result["p_value"] is None


def test_paired_binary_effect_is_deterministic() -> None:
    a = paired_binary_effect([1, 1, 1, 0], [1, 0, 0, 0], seed=19001, resamples=250)
    b = paired_binary_effect([1, 1, 1, 0], [1, 0, 0, 0], seed=19001, resamples=250)
    assert a == b
