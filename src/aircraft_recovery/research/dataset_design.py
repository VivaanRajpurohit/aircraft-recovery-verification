"""Static audit for the design-only final dataset plan."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.common import file_hash, object_hash


SPLITS = ("train", "development", "challenge", "ood_stress", "final_sealed_test")


def audit_dataset_design(path: str | Path) -> dict[str, Any]:
    design = load_yaml(path)
    errors: list[str] = []
    matrix = design.get("scenario_matrix", [])
    total = sum(int(row["count"]) for row in matrix)
    if total != int(design["total_flights"]):
        errors.append("scenario_matrix_total_mismatch")
    split_totals = Counter()
    tier_splits: dict[int, Counter] = defaultdict(Counter)
    family_names: set[str] = set()
    for row in matrix:
        allocation = list(map(int, row["split_allocation"]))
        if len(allocation) != len(SPLITS) or sum(allocation) != int(row["count"]):
            errors.append(f"invalid_family_allocation:{row['family']}")
            continue
        if row["family"] in family_names:
            errors.append(f"duplicate_family:{row['family']}")
        family_names.add(row["family"])
        for split, count in zip(SPLITS, allocation, strict=True):
            split_totals[split] += count
            tier_splits[int(row["tier"])][split] += count
    if dict(split_totals) != {name: int(design["split_counts"][name]) for name in SPLITS}:
        errors.append("scenario_matrix_split_total_mismatch")
    for tier in range(6):
        expected = {name: int(design["tier_split_counts"][f"tier_{tier}"][name]) for name in SPLITS}
        if dict(tier_splits[tier]) != expected or sum(expected.values()) != 192:
            errors.append(f"tier_allocation_mismatch:{tier}")
    seed_sets: dict[str, set[int]] = {}
    for split in SPLITS:
        low, high = map(int, design["seed_ranges"][split])
        seed_sets[split] = set(range(low, high + 1))
        if len(seed_sets[split]) != int(design["split_counts"][split]):
            errors.append(f"seed_range_count_mismatch:{split}")
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if seed_sets[left] & seed_sets[right]:
                errors.append(f"seed_overlap:{left}:{right}")
    holdouts = design["holdout_rules"]
    severity_sets = [set(map(float, holdouts[name])) for name in (
        "challenge_severity_values", "ood_severity_values", "final_sealed_severity_values",
    )]
    if any(severity_sets[left] & severity_sets[right] for left in range(3) for right in range(left + 1, 3)):
        errors.append("heldout_severity_overlap")
    if not holdouts.get("final_sealed_conditions_may_not_be_used_for_tuning"):
        errors.append("sealed_tuning_guard_missing")
    if any(bool(value) for value in design.get("guards", {}).values()):
        errors.append("design_only_guard_enabled")
    if design.get("vertical_speed_plan", {}).get("direct_initial_override_supported") is not False:
        errors.append("unsupported_initial_vertical_speed_claim")
    factor_counts = {key: int(value) for key, value in design.get("factor_flight_counts", {}).items()}
    if factor_counts.get("left_engine_loss") != factor_counts.get("right_engine_loss"):
        errors.append("engine_side_imbalance")
    severity_counts = {key: int(value) for key, value in design.get("severity_band_counts", {}).items()}
    if set(severity_counts.values()) != {192} or sum(severity_counts.values()) != total:
        errors.append("severity_band_imbalance")
    result = {
        "design_id": design["design_id"], "status": "PASS" if not errors else "FAIL",
        "errors": errors, "design_file_sha256": file_hash(path),
        "total_flights": total, "split_totals": dict(split_totals),
        "tier_totals": {str(tier): sum(tier_splits[tier].values()) for tier in range(6)},
        "family_count": len(family_names), "seed_overlap": False if not any("seed_overlap" in item for item in errors) else True,
        "factor_flight_counts": factor_counts, "severity_band_counts": severity_counts,
        "engine_side_balanced": factor_counts.get("left_engine_loss") == factor_counts.get("right_engine_loss"),
        "large_dataset_generated": False, "sealed_test_opened": False,
    }
    result["audit_hash"] = object_hash(result)
    return result
