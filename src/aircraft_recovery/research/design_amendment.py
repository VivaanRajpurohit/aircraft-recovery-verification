"""Static audit for the proposed per-tier final-generation amendment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.common import file_hash, object_hash
from aircraft_recovery.research.dataset_design import audit_dataset_design
from aircraft_recovery.research.pre_generation import expand_seed_registry


SPLITS = ("train", "development", "challenge", "ood_stress", "final_sealed_test")


def _flatten_values(row: dict[str, Any]) -> list[float]:
    return [float(value) for split in SPLITS for value in row.get(split, [])]


def _check_schedule(
    errors: list[str], tier: int, factor: str, row: dict[str, Any], bounds: tuple[float, float],
) -> None:
    low, high = bounds
    for split in SPLITS:
        values = [float(value) for value in row.get(split, [])]
        for value in values:
            if not low <= value <= high:
                errors.append(f"out_of_tier_range:tier_{tier}:{factor}:{split}:{value}")
    train = set(map(float, row.get("train", [])))
    for split in ("challenge", "ood_stress", "final_sealed_test"):
        collision = train & set(map(float, row.get(split, [])))
        if collision:
            errors.append(f"required_value_holdout_collision:tier_{tier}:{factor}:{split}:{sorted(collision)}")
    sealed = set(map(float, row.get("final_sealed_test", [])))
    for split in ("challenge", "ood_stress"):
        collision = sealed & set(map(float, row.get(split, [])))
        if collision:
            errors.append(f"sealed_value_collision:tier_{tier}:{factor}:{split}:{sorted(collision)}")


def v1_conflict_table(root: Path) -> list[dict[str, Any]]:
    """Describe every split-value ambiguity or incompatibility in v1."""
    tiers = load_yaml(root / "configs/research/final_tiers_v1.yaml")["tiers"]
    heldout = load_yaml(root / "configs/research/final_heldout_factor_matrix_v1.yaml")["splits"]
    rows: list[dict[str, Any]] = []
    bounds_by_factor = {
        0: {"turbulence_intensity": tiers[0]["turbulence_intensity"]},
        1: {"severity": tiers[1]["surface_failure_severity"], "failure_activation_seconds": tiers[1]["failure_activation_seconds"], "turbulence_intensity": tiers[1]["turbulence_intensity"]},
        2: {"severity": tiers[2]["surface_failure_severity"], "failure_activation_seconds": tiers[2]["failure_activation_seconds"], "actuator_delay_seconds": tiers[2]["actuator_delay_seconds"], "turbulence_intensity": tiers[2]["turbulence_intensity"], **{f"sensor_bias_{name}": value for name, value in tiers[2]["sensor_bias_maximums"].items()}},
        3: {"severity": tiers[3]["component_severity"], "failure_activation_seconds": tiers[3]["failure_activation_seconds"], "actuator_delay_seconds": tiers[3]["actuator_delay_seconds"], "turbulence_intensity": tiers[3]["turbulence_intensity"], **{f"sensor_bias_{name}": value for name, value in tiers[3]["sensor_bias_maximums"].items()}},
        4: {"severity": tiers[4]["component_severity"], "failure_activation_seconds": tiers[4]["failure_activation_seconds"], "actuator_delay_seconds": tiers[4]["actuator_delay_seconds"], "turbulence_intensity": tiers[4]["turbulence_intensity"], "crosswind_maximum_kts": "UNDEFINED_IN_V1", **{f"sensor_bias_{name}": value for name, value in tiers[4]["sensor_bias_maximums"].items()}},
        5: {"severity": tiers[5]["component_severity"], "failure_activation_seconds": tiers[5]["failure_activation_seconds"], "actuator_delay_seconds": tiers[5]["actuator_delay_seconds"], "turbulence_intensity": tiers[5]["turbulence_intensity"], **{f"sensor_bias_{name}": value for name, value in tiers[5]["sensor_bias_maximums"].items()}},
    }
    for tier, factors in bounds_by_factor.items():
        if tier == 0:
            rows.append({"tier": 0, "factor": "failure_parameters", "allowed_range": "NOT_APPLICABLE", "train": [], "development": [], "challenge": [], "ood_stress": [], "final_sealed_test": [], "status": "COMPATIBLE_NOMINAL"})
        for factor, bounds in factors.items():
            values: dict[str, Any] = {"train": "UNSPECIFIED", "development": "UNSPECIFIED"}
            if factor == "severity":
                for split in ("challenge", "ood_stress", "final_sealed_test"):
                    values[split] = heldout[split]["severity_values"]
            elif factor == "failure_activation_seconds":
                for split in ("challenge", "ood_stress", "final_sealed_test"):
                    values[split] = heldout[split]["activation_seconds"]
            else:
                values.update({split: "UNSPECIFIED" for split in ("challenge", "ood_stress", "final_sealed_test")})
            if bounds == "UNDEFINED_IN_V1":
                status = "INCOMPATIBLE_UNDEFINED_BOUND"
            elif factor in {"severity", "failure_activation_seconds"}:
                low, high = map(float, bounds)
                bad = {split: [float(v) for v in values[split] if not low <= float(v) <= high] for split in ("challenge", "ood_stress", "final_sealed_test")}
                status = "INCOMPATIBLE" if any(bad.values()) else "COMPATIBLE_BUT_TRAIN_DEV_UNSPECIFIED"
            else:
                status = "INCOMPATIBLE_UNSPECIFIED_SPLIT_VALUES"
            rows.append({"tier": tier, "factor": factor, "allowed_range": bounds, **values, "status": status})
        initial_keys = ("initial_airspeed_kts", "initial_bank_deg", "initial_pitch_deg", "altitude_ft", "terrain_clearance_ft")
        common = load_yaml(root / "configs/research/final_tiers_v1.yaml")["common"]
        for factor in initial_keys:
            bounds = tiers[tier].get(factor, common.get(factor))
            rows.append({"tier": tier, "factor": factor, "allowed_range": bounds, "train": "UNSPECIFIED_GRID", "development": "UNSPECIFIED_GRID", "challenge": "UNSPECIFIED_GRID", "ood_stress": "UNSPECIFIED_GRID", "final_sealed_test": "UNSPECIFIED_GRID", "status": "INCOMPATIBLE_UNSPECIFIED_SPLIT_VALUES"})
    return rows


def audit_v2_amendment(root: Path) -> dict[str, Any]:
    tiers_v1 = load_yaml(root / "configs/research/final_tiers_v1.yaml")["tiers"]
    tiers_v2 = load_yaml(root / "configs/research/final_tiers_v2.yaml")
    schedule = load_yaml(root / "configs/research/final_heldout_factor_matrix_v2.yaml")
    design_v2 = load_yaml(root / "configs/training/final_policy_dataset_design_v2.yaml")
    design_v1 = load_yaml(root / design_v2["base_design"])
    errors: list[str] = []
    for tier in (0, 1):
        _check_schedule(errors, tier, "turbulence_intensity", schedule["tiers"][tier]["turbulence_intensity"], tuple(map(float, tiers_v1[tier]["turbulence_intensity"])))
    grid = schedule["initial_condition_grid"]
    for split in SPLITS:
        fractions = [float(value) for value in grid[split]]
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            errors.append(f"initial_condition_fraction_out_of_range:{split}")
    train_fractions = set(map(float, grid["train"]))
    for split in ("challenge", "ood_stress", "final_sealed_test"):
        if train_fractions & set(map(float, grid[split])):
            errors.append(f"initial_condition_holdout_collision:{split}")
    for tier in range(1, 6):
        row = schedule["tiers"][tier]
        severity_key = "surface_failure_severity" if tier in (1, 2) else "component_severity"
        _check_schedule(errors, tier, "component_severity", row["component_severity"], tuple(map(float, tiers_v1[tier][severity_key])))
        _check_schedule(errors, tier, "activation_seconds", row["activation_seconds"], tuple(map(float, tiers_v1[tier]["failure_activation_seconds"])))
        if "actuator_delay_seconds" in row:
            _check_schedule(errors, tier, "actuator_delay_seconds", row["actuator_delay_seconds"], tuple(map(float, tiers_v1[tier]["actuator_delay_seconds"])))
        if "turbulence_intensity" in row:
            _check_schedule(errors, tier, "turbulence_intensity", row["turbulence_intensity"], tuple(map(float, tiers_v1[tier]["turbulence_intensity"])))
        for name, factor_schedule in row.get("sensor_bias_maximums", {}).items():
            _check_schedule(errors, tier, f"sensor_bias_{name}", factor_schedule, tuple(map(float, tiers_v1[tier]["sensor_bias_maximums"][name])))
    crosswind_bounds = tuple(map(float, tiers_v2["additional_explicit_bounds"]["tier_4_crosswind_maximum_kts"]))
    _check_schedule(errors, 4, "crosswind_maximum_kts", schedule["tiers"][4]["crosswind_maximum_kts"], crosswind_bounds)
    base_audit = audit_dataset_design(root / design_v2["base_design"])
    if base_audit["status"] != "PASS" or int(design_v2["total_flights"]) != 1152:
        errors.append("base_design_arithmetic_failed")
    if design_v2["split_counts"] != design_v1["split_counts"]:
        errors.append("split_counts_changed")
    matrix_families = {row["family"] for row in design_v1["scenario_matrix"]}
    semantic_families = set(schedule["family_parameter_semantics"])
    missing_semantics = sorted(matrix_families - semantic_families)
    extra_semantics = sorted(semantic_families - matrix_families)
    if missing_semantics:
        errors.append("missing_family_semantics:" + ",".join(missing_semantics))
    if extra_semantics:
        errors.append("extra_family_semantics:" + ",".join(extra_semantics))
    registry = load_yaml(root / "configs/research/final_seed_registry_v1.json")
    try:
        expanded = expand_seed_registry(registry)
    except ValueError as exc:
        errors.append(str(exc))
        expanded = {}
    if design_v1["factor_flight_counts"]["left_engine_loss"] != design_v1["factor_flight_counts"]["right_engine_loss"]:
        errors.append("engine_side_imbalance")
    files = [
        "configs/training/final_policy_dataset_design_v2.yaml",
        "configs/research/final_tiers_v2.yaml",
        "configs/research/final_heldout_factor_matrix_v2.yaml",
        "configs/research/final_evaluation_design_v2.yaml",
    ]
    result: dict[str, Any] = {
        "audit_id": "final_generation_design_amendment_v2_static_audit",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "v1_conflict_table": v1_conflict_table(root),
        "v2_file_hashes": {path: file_hash(root / path) for path in files},
        "total_flights": design_v1["total_flights"],
        "split_counts": design_v1["split_counts"],
        "tier_counts": base_audit["tier_totals"],
        "scenario_family_count": len(matrix_families),
        "missing_family_semantics": missing_semantics,
        "seed_counts": {name: len(values) for name, values in expanded.items()},
        "engine_side_balanced": True,
        "generation_authorized": False,
        "sealed_outcomes_accessed": False,
    }
    result["audit_hash"] = object_hash(result)
    return result


def build_v2_freeze_manifest(root: Path) -> dict[str, Any]:
    """Create the deterministic proposed-v2 freeze without authorizing generation."""
    audit = audit_v2_amendment(root)
    if audit["status"] != "PASS":
        raise ValueError("v2 amendment audit must pass before a freeze can be proposed")
    paths = [
        "configs/training/final_policy_dataset_design_v1.yaml",
        "configs/training/final_policy_dataset_design_v2.yaml",
        "configs/research/final_tiers_v1.yaml",
        "configs/research/final_tiers_v2.yaml",
        "configs/research/final_heldout_factor_matrix_v2.yaml",
        "configs/research/final_evaluation_design_v2.yaml",
        "configs/research/final_seed_registry_v1.json",
        "configs/research/final_metrics_v1.yaml",
        "configs/research/final_monitor_v1.yaml",
        "configs/research/final_fallback_v1.yaml",
        "artifacts/research_stage/expert_controller_manifest.json",
        "src/aircraft_recovery/data_quality/candidate_generation.py",
        "src/aircraft_recovery/research/design_amendment.py",
    ]
    manifest: dict[str, Any] = {
        "manifest_id": "final_generation_freeze_v2",
        "status": "PROPOSED_V2_FREEZE_PENDING_REVIEW",
        "dataset_id": "final_policy_candidate_v2",
        "schema_version": 2,
        "supersedes_v1_before_generation": True,
        "superseded_v1_freeze_hash": "fcfde639acb99d85b17477dee7da5743da305b46a32cb0b4293fd9c9423a1cd4",
        "amendment_audit_hash": audit["audit_hash"],
        "expected_flights": 1152,
        "expected_split_counts": audit["split_counts"],
        "expected_tier_counts": audit["tier_counts"],
        "files": {path: file_hash(root / path) for path in paths},
        "generation_authorized": False,
        "training_authorized": False,
        "final_sealed_test_outcomes_authorized": False,
        "dataset_records_generated": False,
    }
    manifest["freeze_hash"] = object_hash(manifest)
    return manifest


def audit_v2_1_amendment(root: Path) -> dict[str, Any]:
    """Validate the minimal semantic overlay without changing v2 artifacts."""
    from aircraft_recovery.research.holdout_audit import build_exposure_matrix, resolved_v2_1_schedule

    base = audit_v2_amendment(root)
    schedule, overlay = resolved_v2_1_schedule(root)
    tiers = load_yaml(root / "configs/research/final_tiers_v1.yaml")["tiers"]
    errors = list(base["errors"])
    for tier, replacements in overlay["sealed_replacements"].items():
        tier = int(tier)
        for factor, values in replacements.items():
            key = "surface_failure_severity" if tier == 2 and factor == "component_severity" else (
                "component_severity" if factor == "component_severity" else "failure_activation_seconds"
            )
            low, high = map(float, tiers[tier][key])
            if any(not low <= float(value) <= high for value in values):
                errors.append(f"v2_1_replacement_out_of_range:tier_{tier}:{factor}")
    exposure = build_exposure_matrix(root, use_v2_1=True)
    if exposure["status"] != "PASS":
        errors.extend(exposure["errors"])
    result: dict[str, Any] = {
        "audit_id": "final_generation_design_amendment_v2_1_static_audit",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "base_v2_audit_hash": base["audit_hash"],
        "holdout_exposure_audit_hash": exposure["audit_hash"],
        "changed_values": overlay["sealed_replacements"],
        "total_flights": base["total_flights"],
        "split_counts": base["split_counts"],
        "tier_counts": base["tier_counts"],
        "generation_authorized": False,
        "sealed_outcomes_accessed": False,
    }
    result["audit_hash"] = object_hash(result)
    return result


def build_v2_1_freeze_manifest(root: Path) -> dict[str, Any]:
    """Build the non-active proposed v2.1 freeze."""
    audit = audit_v2_1_amendment(root)
    if audit["status"] != "PASS":
        raise ValueError("v2.1 amendment audit must pass before freeze proposal")
    previous = load_yaml(root / "artifacts/research_stage/final_generation_manifest_v2.json")
    paths = [
        "configs/training/final_policy_dataset_design_v1.yaml",
        "configs/training/final_policy_dataset_design_v2.yaml",
        "configs/training/final_policy_dataset_design_v2_1.yaml",
        "configs/research/final_tiers_v1.yaml",
        "configs/research/final_tiers_v2.yaml",
        "configs/research/final_heldout_factor_matrix_v2.yaml",
        "configs/research/final_heldout_factor_matrix_v2_1.yaml",
        "configs/research/final_evaluation_design_v2_1.yaml",
        "configs/research/final_seed_registry_v1.json",
        "configs/research/final_metrics_v1.yaml",
        "configs/research/final_monitor_v1.yaml",
        "configs/research/final_fallback_v1.yaml",
        "artifacts/research_stage/expert_controller_manifest.json",
        "src/aircraft_recovery/data_quality/candidate_generation.py",
        "src/aircraft_recovery/research/design_amendment.py",
        "src/aircraft_recovery/research/holdout_audit.py",
    ]
    manifest: dict[str, Any] = {
        "manifest_id": "final_generation_freeze_v2_1",
        "status": "PROPOSED_V2_1_FREEZE_PENDING_ACTIVATION_REVIEW",
        "dataset_id": "final_policy_candidate_v2_1",
        "schema_version": 2,
        "historical_v1_freeze_hash": "fcfde639acb99d85b17477dee7da5743da305b46a32cb0b4293fd9c9423a1cd4",
        "superseded_v2_proposal_hash": previous["freeze_hash"],
        "amendment_audit_hash": audit["audit_hash"],
        "holdout_exposure_audit_hash": audit["holdout_exposure_audit_hash"],
        "expected_flights": 1152,
        "expected_split_counts": audit["split_counts"],
        "expected_tier_counts": audit["tier_counts"],
        "files": {path: file_hash(root / path) for path in paths},
        "generation_authorized": False,
        "training_authorized": False,
        "final_sealed_test_outcomes_authorized": False,
        "dataset_records_generated": False,
    }
    manifest["freeze_hash"] = object_hash(manifest)
    return manifest
