"""Prospective precision analysis for the pre-generation flight counts."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

from aircraft_recovery.analysis.statistics import wilson_interval
from aircraft_recovery.data_quality.common import object_hash, write_json


def main() -> None:
    rows = []
    for n in (12, 24, 72, 144, 288):
        for probability in (0.5, 0.8, 0.9):
            successes = round(n * probability)
            low, high = wilson_interval(successes, n)
            rows.append({
                "n": n, "assumed_rate": probability, "integer_successes": successes,
                "wilson_95_ci": [low, high], "approximate_half_width": (high - low) / 2.0,
            })
    z_sum = 1.959963984540054 + 0.8416212335729143
    paired_binary = [
        {"n_pairs": n, "assumed_discordant_fraction": discordance,
         "approximate_80_percent_mde_absolute_rate_difference": z_sum * sqrt(discordance / n)}
        for n in (72, 144, 288) for discordance in (0.3, 0.5)
    ]
    paired_continuous = [
        {"n_pairs": n, "approximate_80_percent_mde_standardized_mean_difference": z_sum / sqrt(n)}
        for n in (24, 72, 144, 288)
    ]
    result = {
        "analysis_type": "prospective_approximate_precision_not_formal_power_guarantee",
        "confidence_level": 0.95, "target_power_for_mde_approximation": 0.80,
        "proportion_precision": rows, "paired_binary_mde": paired_binary,
        "paired_continuous_mde": paired_continuous,
        "decision": "SAMPLE_SIZE_REVISION_REQUIRED_FOR_720; REVISED_1152_ACCEPTED",
        "rationale": "Increase each evaluation split from 72 to 144 and each tier stratum from 12 to 24. Family-by-split cells of six remain exploratory.",
        "sealed_test_opened": False,
    }
    result["analysis_hash"] = object_hash(result)
    output = Path("artifacts/research_stage/final_sample_size_analysis.json")
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
