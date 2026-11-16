"""Cross-split exposure and semantic holdout audit without trajectory access."""

from __future__ import annotations

import copy
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.common import file_hash, object_hash
from aircraft_recovery.research.pre_generation import expand_seed_registry


SPLITS = ("train", "development", "challenge", "ood_stress", "final_sealed_test")
EARLIER = SPLITS[:-1]


def resolved_v2_1_schedule(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = load_yaml(root / "configs/research/final_heldout_factor_matrix_v2_1.yaml")
    schedule = copy.deepcopy(load_yaml(root / overlay["base_schedule"]))
    for tier, replacements in overlay["sealed_replacements"].items():
        for factor, values in replacements.items():
            schedule["tiers"][int(tier)][factor]["final_sealed_test"] = values
    schedule["contract_id"] = overlay["contract_id"]
    schedule["factor_semantics"] = overlay["factor_semantics"]
    schedule["canonical_scenario_signature"] = overlay["canonical_scenario_signature"]
    return schedule, overlay


def _values(schedule: dict[str, Any], factor: str) -> dict[str, set[float]]:
    result = {split: set() for split in SPLITS}
    for tier in range(6):
        row = schedule["tiers"][tier]
        factor_row = row.get(factor)
        if isinstance(factor_row, dict) and all(split in factor_row for split in SPLITS):
            for split in SPLITS:
                result[split].update(map(float, factor_row[split]))
    return result


def _sensor_values(schedule: dict[str, Any]) -> dict[str, dict[str, set[float]]]:
    output: dict[str, dict[str, set[float]]] = {}
    for name in ("airspeed_kts", "altitude_ft", "heading_deg"):
        output[name] = {split: set() for split in SPLITS}
        for tier in range(6):
            factor = schedule["tiers"][tier].get("sensor_bias_maximums", {}).get(name)
            if factor:
                for split in SPLITS:
                    output[name][split].update(map(float, factor[split]))
    return output


def _strict_collision(values: dict[str, set[float]]) -> list[float]:
    earlier = set().union(*(values[split] for split in EARLIER))
    return sorted(values["final_sealed_test"] & earlier)


def _model_selection_collision(values: dict[str, set[float]], split: str) -> list[float]:
    return sorted(values[split] & (values["train"] | values["development"]))


def _pairwise(values: dict[str, set[Any]]) -> dict[str, list[Any]]:
    return {
        f"{left}__{right}": sorted(values[left] & values[right])
        for left, right in combinations(SPLITS, 2)
        if values[left] & values[right]
    }


def _tier_values(schedule: dict[str, Any], tier: int, factor: str) -> dict[str, set[float]]:
    row = schedule["tiers"][tier].get(factor, {})
    return {split: set(map(float, row.get(split, []))) for split in SPLITS}


def _signature_support(root: Path, schedule: dict[str, Any]) -> dict[str, set[tuple[Any, ...]]]:
    design = load_yaml(root / "configs/training/final_policy_dataset_design_v1.yaml")
    semantics = schedule["family_parameter_semantics"]
    support = {split: set() for split in SPLITS}
    grid = schedule["initial_condition_grid"]
    for matrix_row in design["scenario_matrix"]:
        family = matrix_row["family"]
        tier = int(matrix_row["tier"])
        parameters = semantics[family]["parameters"]
        tier_row = schedule["tiers"][tier]
        for split, count in zip(SPLITS, matrix_row["split_allocation"], strict=True):
            if not count:
                continue
            component = tier_row.get("component_severity", {}).get(split, [None])
            activation = tier_row.get("activation_seconds", {}).get(split, [None])
            delay = tier_row.get("actuator_delay_seconds", {}).get(split, [None])
            turbulence = tier_row.get("turbulence_intensity", {}).get(split, [None])
            crosswind = tier_row.get("crosswind_maximum_kts", {}).get(split, [None])
            bias_group = tier_row.get("sensor_bias_maximums", {})
            bias = [None]
            if "sensor_bias_maximums" in parameters:
                bias = [value for name in sorted(bias_group) for value in bias_group[name][split]]
            for index in range(int(count)):
                support[split].add((
                    tier, family,
                    component[index % len(component)] if "component_severity" in parameters else None,
                    activation[index % len(activation)] if "activation_seconds" in parameters else None,
                    delay[index % len(delay)] if "actuator_delay_seconds" in parameters else None,
                    bias[index % len(bias)] if "sensor_bias_maximums" in parameters else None,
                    turbulence[index % len(turbulence)] if "turbulence_intensity" in parameters else None,
                    crosswind[index % len(crosswind)] if "crosswind_maximum_kts" in parameters else None,
                    grid[split][index % len(grid[split])],
                    "right" if family == "right_engine_rudder_crosswind" else "balanced_or_not_applicable",
                ))
    return support


def build_exposure_matrix(root: Path, *, use_v2_1: bool = True) -> dict[str, Any]:
    if use_v2_1:
        schedule, overlay = resolved_v2_1_schedule(root)
    else:
        schedule = load_yaml(root / "configs/research/final_heldout_factor_matrix_v2.yaml")
        overlay = {"factor_semantics": {}}
    tiers = load_yaml(root / "configs/research/final_tiers_v1.yaml")
    registry = json.loads((root / "configs/research/final_seed_registry_v1.json").read_text(encoding="utf-8"))
    seeds = {split: set(values) for split, values in expand_seed_registry(registry).items()}
    rows: list[dict[str, Any]] = []
    for tier in range(6):
        tier_row = schedule["tiers"][tier]
        factors = [name for name in ("component_severity", "activation_seconds", "actuator_delay_seconds", "turbulence_intensity", "crosswind_maximum_kts") if name in tier_row]
        for factor in factors:
            values = _tier_values(schedule, tier, factor)
            rows.append({"tier": tier, "factor": factor, "exposures": {split: sorted(values[split]) for split in SPLITS}, "pairwise_collisions": _pairwise(values)})
        for name, factor_row in tier_row.get("sensor_bias_maximums", {}).items():
            values = {split: set(map(float, factor_row[split])) for split in SPLITS}
            rows.append({"tier": tier, "factor": f"sensor_bias_{name}", "exposures": {split: sorted(values[split]) for split in SPLITS}, "pairwise_collisions": _pairwise(values)})
        severity = _tier_values(schedule, tier, "component_severity")
        if any(severity.values()):
            for derived in ("actuator_effectiveness", "partial_thrust_fraction"):
                values = {split: {round(1.0 - value, 10) for value in severity[split]} for split in SPLITS}
                rows.append({"tier": tier, "factor": derived, "exposures": {split: sorted(values[split]) for split in SPLITS}, "pairwise_collisions": _pairwise(values), "derived": "1-component_severity"})
        common = tiers["common"]
        for factor in ("initial_airspeed_kts", "initial_bank_deg", "initial_pitch_deg", "altitude_ft", "terrain_clearance_ft"):
            low, high = map(float, tiers["tiers"][tier].get(factor, common[factor]))
            values = {split: {round(low + float(fraction) * (high - low), 8) for fraction in schedule["initial_condition_grid"][split]} for split in SPLITS}
            rows.append({"tier": tier, "factor": factor, "allowed_range": [low, high], "exposures": {split: sorted(values[split]) for split in SPLITS}, "pairwise_collisions": _pairwise(values)})
    global_factors = {
        "component_severity": _values(schedule, "component_severity"),
        "failure_activation_time": _values(schedule, "activation_seconds"),
        "derived_actuator_effectiveness": {split: {round(1.0 - value, 10) for value in _values(schedule, "component_severity")[split]} for split in SPLITS},
        "derived_partial_thrust_fraction": {split: {round(1.0 - value, 10) for value in _values(schedule, "component_severity")[split]} for split in SPLITS},
        "crosswind_maximum_kts": _values(schedule, "crosswind_maximum_kts"),
        "actuator_delay": _values(schedule, "actuator_delay_seconds"),
        "turbulence": _values(schedule, "turbulence_intensity"),
    }
    for name, values in _sensor_values(schedule).items():
        global_factors[f"sensor_bias_{name}"] = values
    strict_names = ("component_severity", "failure_activation_time", "derived_actuator_effectiveness", "derived_partial_thrust_fraction", "crosswind_maximum_kts")
    strict_collisions = {name: _strict_collision(global_factors[name]) for name in strict_names}
    model_selection = {"actuator_delay": _model_selection_collision(global_factors["actuator_delay"], "final_sealed_test")}
    challenge_tier_collisions: dict[str, list[float]] = {}
    ood_tier_collisions: dict[str, list[float]] = {}
    for tier in range(1, 6):
        for factor in ("component_severity", "activation_seconds"):
            values = _tier_values(schedule, tier, factor)
            challenge_tier_collisions[f"tier_{tier}:{factor}"] = _model_selection_collision(values, "challenge")
            ood_tier_collisions[f"tier_{tier}:{factor}"] = _model_selection_collision(values, "ood_stress")
    signatures = _signature_support(root, schedule)
    signature_duplicates = {
        split: len(signatures["final_sealed_test"] & signatures[split]) for split in EARLIER
    }
    family_sets = {split: {signature[1] for signature in values} for split, values in signatures.items()}
    errors = []
    for factor, collision in strict_collisions.items():
        if collision:
            errors.append(f"strict_value_collision:{factor}:{collision}")
    for factor, collision in model_selection.items():
        if collision:
            errors.append(f"model_selection_collision:{factor}:{collision}")
    for factor, collision in challenge_tier_collisions.items():
        if collision:
            errors.append(f"challenge_model_selection_collision:{factor}:{collision}")
    for factor, collision in ood_tier_collisions.items():
        if collision:
            errors.append(f"ood_model_selection_collision:{factor}:{collision}")
    if any(signature_duplicates.values()):
        errors.append(f"sealed_signature_duplicates:{signature_duplicates}")
    seed_collisions = _pairwise(seeds)
    if seed_collisions:
        errors.append(f"seed_collisions:{seed_collisions}")
    result: dict[str, Any] = {
        "audit_id": "final_holdout_separation_v2_1" if use_v2_1 else "final_holdout_separation_v2_original",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "schedule_id": schedule["contract_id"],
        "factor_semantics": overlay.get("factor_semantics", {}),
        "exposure_rows": rows,
        "global_factor_exposures": {name: {split: sorted(values[split]) for split in SPLITS} for name, values in global_factors.items()},
        "global_factor_pairwise_collisions": {name: _pairwise(values) for name, values in global_factors.items()},
        "strict_sealed_collisions": strict_collisions,
        "model_selection_collisions": model_selection,
        "challenge_tier_model_selection_collisions": challenge_tier_collisions,
        "ood_tier_model_selection_collisions": ood_tier_collisions,
        "seed_exposures": {split: sorted(values) for split, values in seeds.items()},
        "seed_collisions": seed_collisions,
        "sealed_signature_duplicate_counts": signature_duplicates,
        "scenario_families_seen_in_all_splits": sorted(set.intersection(*(family_sets[split] for split in SPLITS))),
        "same_family_different_numeric_signature_count": len(family_sets["final_sealed_test"] & set.union(*(family_sets[split] for split in EARLIER))),
        "sealed_signature_count": len(signatures["final_sealed_test"]),
        "sealed_outcomes_accessed": False,
        "files": {
            "base_v2": file_hash(root / "configs/research/final_heldout_factor_matrix_v2.yaml"),
            "overlay_v2_1": file_hash(root / "configs/research/final_heldout_factor_matrix_v2_1.yaml"),
        },
    }
    result["audit_hash"] = object_hash(result)
    return result
