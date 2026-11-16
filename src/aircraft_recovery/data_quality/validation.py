"""Stage 2: independent validation of serialized candidate records.

This module intentionally has no simulator, controller, neural-policy, or
formal-verification imports.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data_quality.common import (
    CONTROL_NAMES,
    SCHEMA_VERSION,
    assert_not_sealed_path,
    file_hash,
    load_jsonl,
    object_hash,
    record_hash,
    write_json,
)
from aircraft_recovery.data_quality.contracts import flight_contract_checks


STATUS_ORDER = {"PASS": 0, "REVIEW_REQUIRED": 1, "FAIL": 2}
VALIDATOR_VERSION = "candidate-validator-v3-independent-contracts"
PLAUSIBLE_LIMITS = {
    "airspeed_kts": (0.0, 600.0),
    "altitude_ft": (-1000.0, 60000.0),
    "pitch_deg": (-90.0, 90.0),
    "roll_deg": (-180.0, 180.0),
    "heading_deg": (0.0, 360.0),
    "vertical_speed_fpm": (-15000.0, 15000.0),
}
STATE_FEATURES = {
    "altitude_ft": "altitude_ft", "airspeed_kts": "airspeed_kts",
    "pitch_deg": "pitch_deg", "roll_deg": "roll_deg", "yaw_deg": "yaw_deg",
    "vertical_speed_fpm": "vertical_speed_fpm", "heading_deg": "heading_deg",
    "latitude_deg": "latitude_deg", "longitude_deg": "longitude_deg",
}


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return False


def _check(name: str, status: str, reason: str | None = None) -> dict[str, str | None]:
    return {"check": name, "status": status, "reason": reason}


def _result(record_id: str, checks: list[dict[str, str | None]]) -> dict[str, Any]:
    status = max((str(item["status"]) for item in checks), key=STATUS_ORDER.get)
    return {
        "record_id": record_id,
        "status": status,
        "checks": checks,
        "reasons": [str(item["reason"]) for item in checks if item["reason"]],
    }


def validate_record(
    record: dict[str, Any],
    previous: dict[str, Any] | None = None,
    global_integrity_errors: list[str] | None = None,
    contract_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one serialized mapping using independent cross-field checks."""
    checks: list[dict[str, str | None]] = []
    record_id = str(record.get("record_id", "<missing>"))
    for error in global_integrity_errors or []:
        checks.append(_check("file_and_provenance_integrity", "FAIL", error))
    required = {
        "schema_version", "dataset_id", "record_id", "flight_id", "scenario_id",
        "random_seed", "split", "timestep_index", "simulation_timestamp_seconds",
        "state_after_timestamp_seconds", "physics_timestep_seconds", "aircraft_state_before",
        "ordered_observation_42", "observation_feature_names", "expert_proposed_controls",
        "executed_controls", "actuator_state", "engine_state", "active_failures",
        "disturbance_values", "aircraft_state_after", "termination_status",
        "source_configuration_hash", "git_commit_hash", "record_hash",
    }
    if record.get("schema_version") == 2:
        required |= {
            "scenario_instance_id", "difficulty_tier", "scenario_family", "severity_band",
            "monitor_modified_controls", "controller_known_failures", "ground_truth_failures",
            "ground_truth_failures_after",
            "failure_schedule", "actuator_effectiveness", "actuator_delay_seconds",
            "sensor_error_failures", "effective_scenario_configuration_hash",
            "simulator_configuration_hash", "expert_configuration_hash", "generator_version",
            "simulator_version",
        }
    missing = sorted(required - set(record))
    checks.append(_check("required_fields", "FAIL" if missing else "PASS", f"missing:{','.join(missing)}" if missing else None))
    if missing:
        return _result(record_id, checks)
    checks.append(_check(
        "schema_version", "PASS" if record["schema_version"] in {1, SCHEMA_VERSION} else "FAIL",
        None if record["schema_version"] in {1, SCHEMA_VERSION} else "unsupported_schema_version",
    ))
    try:
        hash_matches = record.get("record_hash") == record_hash(record)
    except (TypeError, ValueError):
        hash_matches = False
    checks.append(_check(
        "record_hash", "PASS" if hash_matches else "FAIL",
        None if hash_matches else "record_hash_mismatch",
    ))
    provenance_fields = ["source_configuration_hash", "git_commit_hash"]
    if record["schema_version"] == 2:
        provenance_fields += ["effective_scenario_configuration_hash", "simulator_configuration_hash",
                              "expert_configuration_hash", "generator_version", "simulator_version"]
    provenance_present = all(bool(str(record.get(name, "")).strip()) for name in provenance_fields)
    checks.append(_check(
        "provenance_hashes_present", "PASS" if provenance_present else "FAIL",
        None if provenance_present else "missing_configuration_or_git_hash",
    ))
    checks.append(_check("finite_values", "PASS" if _finite(record) else "FAIL", None if _finite(record) else "nan_or_infinity_or_invalid_numeric_type"))
    observations = record["ordered_observation_42"]
    feature_names = record["observation_feature_names"]
    proposed = record["expert_proposed_controls"]
    executed = record["executed_controls"]
    numeric_types_valid = (
        isinstance(observations, list) and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in observations)
        and isinstance(proposed, list) and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in proposed)
        and isinstance(executed, list) and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in executed)
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for state in (record["aircraft_state_before"], record["aircraft_state_after"])
            for value in state.values()
        )
    )
    checks.append(_check(
        "numeric_types", "PASS" if numeric_types_valid else "FAIL",
        None if numeric_types_valid else "invalid_numerical_type",
    ))
    checks.append(_check("observation_dimension", "PASS" if len(observations) == 42 else "FAIL", None if len(observations) == 42 else "observation_dimension_not_42"))
    checks.append(_check("feature_dimension", "PASS" if len(feature_names) == 42 else "FAIL", None if len(feature_names) == 42 else "feature_name_dimension_not_42"))
    correct_order = feature_names == list(OBSERVATION_FEATURES)
    checks.append(_check("canonical_feature_order", "PASS" if correct_order else "FAIL", None if correct_order else "wrong_feature_order"))
    checks.append(_check("proposed_action_dimension", "PASS" if len(proposed) == 5 else "FAIL", None if len(proposed) == 5 else "proposed_action_dimension_not_5"))
    checks.append(_check("executed_action_dimension", "PASS" if len(executed) == 5 else "FAIL", None if len(executed) == 5 else "executed_action_dimension_not_5"))
    if len(proposed) == 5 and len(executed) == 5:
        action_valid = all(-1.0 <= float(value) <= 1.0 for value in proposed[:3] + executed[:3]) and all(
            0.0 <= float(value) <= 1.0 for value in proposed[3:] + executed[3:]
        )
        checks.append(_check("action_bounds", "PASS" if action_valid else "FAIL", None if action_valid else "control_out_of_bounds"))
    dt = float(record["physics_timestep_seconds"])
    timestamp_delta = float(record["state_after_timestamp_seconds"]) - float(record["simulation_timestamp_seconds"])
    dt_valid = dt > 0 and abs(timestamp_delta - dt) <= 1e-8
    checks.append(_check("timestamp_step", "PASS" if dt_valid else "FAIL", None if dt_valid else "state_timestamp_delta_disagrees_with_dt"))
    state_before, state_after = record["aircraft_state_before"], record["aircraft_state_after"]
    plausible = True
    for field, (low, high) in PLAUSIBLE_LIMITS.items():
        for state in (state_before, state_after):
            value = state.get(field)
            plausible &= isinstance(value, (int, float)) and low <= float(value) <= high
    if "heading_deg" in state_before:
        plausible &= 0.0 <= float(state_before["heading_deg"]) < 360.0
    checks.append(_check("simulator_plausibility_ranges", "PASS" if plausible else "FAIL", None if plausible else "state_outside_declared_simulator_data_quality_limits"))
    if dt > 0:
        pitch_rate = abs(float(state_after["pitch_deg"]) - float(state_before["pitch_deg"])) / dt
        roll_rate = abs(float(state_after["roll_deg"]) - float(state_before["roll_deg"])) / dt
        angular_valid = pitch_rate <= 50.0 and roll_rate <= 100.0
        checks.append(_check("angular_rate_limits", "PASS" if angular_valid else "FAIL", None if angular_valid else "implausible_angular_rate"))
    actuator = record["actuator_state"]
    actuator_valid = all(0.0 <= float(actuator.get(name, -1)) <= 1.0 for name in (
        "elevator_effectiveness", "aileron_effectiveness", "rudder_effectiveness"
    )) and float(actuator.get("actuator_delay_ms", -1)) >= 0.0
    checks.append(_check("actuator_ranges", "PASS" if actuator_valid else "FAIL", None if actuator_valid else "invalid_actuator_state"))
    engine = record["engine_state"]
    engine_valid = all(0.0 <= float(engine.get(name, -1)) <= 1.0 for name in (
        "left_engine_thrust_available", "right_engine_thrust_available"
    ))
    checks.append(_check("engine_availability", "PASS" if engine_valid else "FAIL", None if engine_valid else "invalid_engine_availability"))
    disturbance = record["disturbance_values"]
    disturbance_valid = (
        0.0 <= float(disturbance.get("wind_speed_kts", -1)) <= 250.0
        and 0.0 <= float(disturbance.get("wind_direction_deg", -1)) < 360.0
        and 0.0 <= float(disturbance.get("turbulence_intensity", -1)) <= 1.0
        and float(disturbance.get("terrain_clearance_ft", -1001)) >= -1000.0
    )
    checks.append(_check("disturbance_ranges", "PASS" if disturbance_valid else "FAIL", None if disturbance_valid else "invalid_disturbance_or_unit"))
    if correct_order and len(observations) == 42 and contract_checks is None:
        matches = True
        for feature, state_field in STATE_FEATURES.items():
            expected = float(state_before[state_field])
            actual = float(observations[OBSERVATION_FEATURES.index(feature)])
            matches &= abs(actual - expected) <= max(1e-5, abs(expected) * 1e-6)
        sensor_fault_active = any(
            "sensor" in str(failure.get("affected_subsystem", ""))
            or any(token in str(failure.get("failure_type", "")) for token in ("bias", "frozen", "missing"))
            for failure in record["active_failures"]
        )
        observation_status = "PASS" if matches else "REVIEW_REQUIRED" if sensor_fault_active else "FAIL"
        observation_reason = None if matches else (
            "observed_true_state_difference_under_declared_sensor_fault"
            if sensor_fault_active else "observation_state_mismatch"
        )
        checks.append(_check("observation_matches_state_before", observation_status, observation_reason))
    if previous is not None:
        contiguous = int(record["timestep_index"]) == int(previous["timestep_index"]) + 1
        monotonic = abs(float(record["simulation_timestamp_seconds"]) - float(previous["state_after_timestamp_seconds"])) <= 1e-8
        continuity = all(
            abs(float(state_before[field]) - float(previous["aircraft_state_after"][field])) <= 1e-6
            for field in state_before
        )
        checks.extend([
            _check("contiguous_timestep", "PASS" if contiguous else "FAIL", None if contiguous else "noncontiguous_timestep_index"),
            _check("monotonic_timestamp", "PASS" if monotonic else "FAIL", None if monotonic else "nonmonotonic_or_gapped_timestamp"),
            _check("state_transition_continuity", "PASS" if continuity else "FAIL", None if continuity else "state_after_before_discontinuity"),
        ])
    failures_consistent = all(
        float(failure.get("activation_time_seconds", 0.0)) <= float(record["simulation_timestamp_seconds"]) + 1e-8
        for failure in record["active_failures"]
    )
    checks.append(_check("failure_timing", "PASS" if failures_consistent else "FAIL", None if failures_consistent else "active_failure_precedes_activation"))
    if record["schema_version"] == 2:
        truth_matches = record["ground_truth_failures"] == record["active_failures"]
        known_ids = set(map(str, record["controller_known_failures"]))
        expected_known = {
            str(item.get("failure_id")) for item in record["ground_truth_failures"]
            if item.get("observable", False) and item.get("detected", False)
        }
        checks.append(_check("ground_truth_failure_separation", "PASS" if truth_matches else "FAIL",
                             None if truth_matches else "ground_truth_alias_mismatch"))
        checks.append(_check("controller_known_failure_state", "PASS" if known_ids == expected_known else "FAIL",
                             None if known_ids == expected_known else "known_failure_set_mismatch"))
        monitor_separation = record["monitor_modified_controls"] is None and record.get("monitor_status") is None
        checks.append(_check("proposed_monitor_executed_separation", "PASS" if monitor_separation else "REVIEW_REQUIRED",
                             None if monitor_separation else "monitor_path_requires_declared_runtime_assurance_semantics"))
    uncertain_actuation = (
        float(actuator.get("actuator_delay_ms", 0.0)) > 0.0
        or any(float(actuator.get(name, 1.0)) < 1.0 for name in (
            "elevator_effectiveness", "aileron_effectiveness", "rudder_effectiveness"
        ))
        or any(bool(actuator.get(name, False)) for name in ("elevator_stuck", "aileron_stuck", "rudder_stuck"))
    )
    if contract_checks is not None:
        checks.extend(contract_checks)
    elif uncertain_actuation:
        checks.append(_check("executed_action_actuator_consistency", "REVIEW_REQUIRED", "actuator_degradation_requires_human_review_of_serialized_history"))
    elif len(proposed) == 5 and len(executed) == 5:
        same = np.allclose(np.asarray(proposed), np.asarray(executed), atol=1e-7, rtol=0.0)
        checks.append(_check("executed_action_actuator_consistency", "PASS" if same else "FAIL", None if same else "executed_action_differs_without_declared_actuator_effect"))
    if record.get("label_semantics") != "expert_proposed_action":
        checks.append(_check("label_semantics", "REVIEW_REQUIRED", "nonstandard_runtime_assurance_label_semantics"))
    else:
        checks.append(_check("label_semantics", "PASS"))
    return _result(record_id, checks)


def validate_candidate_dataset(
    dataset_directory: str | Path,
    *,
    suppress_outcomes_for_splits: set[str] | None = None,
    embed_record_results: bool = True,
) -> dict[str, Any]:
    directory = assert_not_sealed_path(dataset_directory)
    required_paths = {
        "transitions": directory / "raw" / "transitions.jsonl",
        "flights": directory / "raw" / "flights.jsonl",
        "manifest": directory / "manifests" / "generation_manifest.json",
        "schema": directory / "manifests" / "schema.json",
        "checksums": directory / "manifests" / "file_checksums.json",
    }
    missing_files = [name for name, path in required_paths.items() if not path.is_file()]
    if missing_files:
        raise ValueError(f"Candidate dataset is incomplete: {missing_files}")
    manifest = json.loads(required_paths["manifest"].read_text(encoding="utf-8"))
    schema = json.loads(required_paths["schema"].read_text(encoding="utf-8"))
    checksums = json.loads(required_paths["checksums"].read_text(encoding="utf-8"))
    records = load_jsonl(required_paths["transitions"])
    flights = load_jsonl(required_paths["flights"])
    integrity_errors: list[str] = []
    if manifest.get("schema_version") not in {1, SCHEMA_VERSION} or schema.get("schema_version") not in {1, SCHEMA_VERSION}:
        integrity_errors.append("unsupported_manifest_or_schema_version")
    if len(records) != manifest.get("record_count"):
        integrity_errors.append("record_count_mismatch")
    if len(flights) != manifest.get("flight_count"):
        integrity_errors.append("flight_count_mismatch")
    flight_ids = [str(row.get("flight_id")) for row in flights]
    if len(flight_ids) != len(set(flight_ids)):
        integrity_errors.append("duplicate_flight_ids")
    for relative, expected in checksums.items():
        path = directory / relative
        if not path.is_file() or file_hash(path) != expected:
            integrity_errors.append(f"file_checksum_mismatch:{relative}")
    ids = [str(record.get("record_id")) for record in records]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        integrity_errors.append("duplicate_record_ids")
    content_hashes = [str(record.get("record_hash")) for record in records]
    duplicate_content_hashes = sorted(
        key for key, count in Counter(content_hashes).items() if count > 1
    )
    if duplicate_content_hashes:
        integrity_errors.append("duplicate_record_hashes")

    by_flight: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_flight[str(record.get("flight_id"))].append(record)
    results: list[dict[str, Any]] = []
    for flight_id in sorted(by_flight):
        ordered = sorted(by_flight[flight_id], key=lambda row: int(row.get("timestep_index", -1)))
        contracts = flight_contract_checks(ordered) if ordered and ordered[0].get("schema_version") == 2 else {}
        previous = None
        active_ids: set[str] = set()
        for record in ordered:
            result = validate_record(
                record, previous, integrity_errors, contracts.get(record.get("record_id")),
            )
            current_ids = {str(item.get("failure_id")) for item in record.get("active_failures", [])}
            disappeared = active_ids - current_ids
            if disappeared:
                result["checks"].append(_check("failure_temporal_consistency", "FAIL", f"permanent_failure_disappeared:{','.join(sorted(disappeared))}"))
                result = _result(result["record_id"], result["checks"])
            active_ids |= current_ids
            results.append(result)
            previous = record

    splits = manifest.get("splits", {})
    split_names = tuple(name for name in (
        "train", "development", "challenge", "ood_stress", "final_sealed_test",
        "validation", "test",
    ) if name in splits)
    split_sets = {name: set(splits.get(name, [])) for name in split_names}
    leakage_errors: list[str] = []
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            if split_sets[left_name] & split_sets[right_name]:
                leakage_errors.append(f"flight_overlap:{left_name}:{right_name}")
    flight_meta = {row["flight_id"]: row for row in flights}
    pairs: dict[tuple[str, int], str] = {}
    for split_name, flight_ids in split_sets.items():
        for flight_id in flight_ids:
            meta = flight_meta.get(flight_id)
            if meta is None:
                leakage_errors.append(f"unknown_split_flight:{flight_id}")
                continue
            pair = (meta["scenario_id"], int(meta["random_seed"]))
            if pair in pairs and pairs[pair] != split_name:
                leakage_errors.append(f"scenario_seed_overlap:{pair[0]}:{pair[1]}")
            pairs[pair] = split_name
    if any("policy_smoke_v1" in canonical for canonical in (directory.as_posix(), json.dumps(manifest))):
        leakage_errors.append("sealed_held_out_dataset_reference")
    trajectory_signatures: dict[str, str] = {}
    near_trajectory_signatures: dict[str, str] = {}
    duplicate_trajectory_pairs: list[list[str]] = []
    near_duplicate_trajectory_pairs: list[list[str]] = []
    for flight_id, flight_records in sorted(by_flight.items()):
        ordered = sorted(flight_records, key=lambda row: int(row.get("timestep_index", -1)))
        trajectory = [row.get("aircraft_state_before", {}) for row in ordered]
        exact = object_hash(trajectory)
        near = object_hash([
            {key: round(float(value), 3) for key, value in row.items() if isinstance(value, (int, float))}
            for row in trajectory
        ])
        if exact in trajectory_signatures:
            duplicate_trajectory_pairs.append([trajectory_signatures[exact], flight_id])
        else:
            trajectory_signatures[exact] = flight_id
        if near in near_trajectory_signatures:
            near_duplicate_trajectory_pairs.append([near_trajectory_signatures[near], flight_id])
        else:
            near_trajectory_signatures[near] = flight_id
    for first, second in duplicate_trajectory_pairs + near_duplicate_trajectory_pairs:
        first_split = flight_meta.get(first, {}).get("split")
        second_split = flight_meta.get(second, {}).get("split")
        if first_split != second_split:
            leakage_errors.append(f"duplicate_trajectory_across_splits:{first}:{second}")
    status_counts = dict(Counter(row["status"] for row in results))
    suppressed = suppress_outcomes_for_splits or set()
    analysis_records = [row for row in records if str(row.get("split")) not in suppressed]
    controls = np.asarray([row["expert_proposed_controls"] for row in analysis_records], dtype=float) if analysis_records else np.empty((0, 5))
    target_distributions = {
        name: {
            "minimum": float(np.min(controls[:, index])), "maximum": float(np.max(controls[:, index])),
            "mean": float(np.mean(controls[:, index])), "standard_deviation": float(np.std(controls[:, index])),
            "near_constant": bool(np.std(controls[:, index]) < 1e-6),
        }
        for index, name in enumerate(CONTROL_NAMES)
    } if len(controls) else {}
    scenario_counts = dict(Counter(row["scenario_id"] for row in analysis_records))
    outcome_records = analysis_records
    outcome_flights = [row for row in flights if str(row.get("split")) not in suppressed]
    outcome_counts = dict(Counter(row.get("outcome_label", "undefined") for row in outcome_records))
    failure_severity_counts = Counter()
    for row in analysis_records:
        for failure in row.get("active_failures", []):
            failure_severity_counts[f"{failure.get('failure_type')}:{failure.get('severity')}"] += 1
    report: dict[str, Any] = {
        "validator_version": VALIDATOR_VERSION,
        "dataset_id": manifest.get("dataset_id"),
        "record_count": len(records),
        "flight_count": len(flights),
        "integrity_errors": sorted(set(integrity_errors)),
        "leakage_errors": sorted(set(leakage_errors)),
        "record_status_counts": {name: status_counts.get(name, 0) for name in ("PASS", "FAIL", "REVIEW_REQUIRED")},
        "record_results": results if embed_record_results else None,
        "record_results_external_file": None if embed_record_results else "validation/record_results.jsonl",
        "quality_summary": {
            "target_distributions": target_distributions,
            "scenario_transition_counts": scenario_counts,
            "outcome_transition_counts": outcome_counts,
            "failure_severity_transition_counts": dict(sorted(failure_severity_counts.items())),
            "saturation_frequency": float(np.mean(np.any(np.abs(controls[:, :3]) >= .99, axis=1) | np.any((controls[:, 3:] <= .01) | (controls[:, 3:] >= .99), axis=1))) if len(controls) else 0.0,
            "normalization_statistics_generated": False,
            "normalization_requirement": "Export may fit statistics from approved training records only.",
            "plausibility_limit_status": "Simulator data-quality limits only; not certified real-aircraft limits.",
            "valid_failed_recovery_auto_rejected": False,
            "duplicate_record_hash_count": len(duplicate_content_hashes),
            "duplicate_trajectory_pairs": duplicate_trajectory_pairs,
            "near_duplicate_trajectory_pairs": near_duplicate_trajectory_pairs,
            "difficulty_tier_flight_counts": dict(Counter(str(row.get("difficulty_tier")) for row in flights)),
            "scenario_family_flight_counts": dict(Counter(str(row.get("scenario_family")) for row in flights)),
            "flight_outcome_counts": dict(Counter(str(row.get("outcome_label")) for row in outcome_flights)),
            "outcome_summary_suppressed_splits": sorted(suppressed),
        },
    }
    report["validator_report_hash"] = object_hash(report)
    validation_dir = directory / "validation"
    validation_dir.mkdir(exist_ok=True)
    (validation_dir / "record_results.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in results), encoding="utf-8"
    )
    write_json(validation_dir / "validation_report.json", report)
    leakage_audit = {
        "audit_version": 1, "dataset_id": manifest.get("dataset_id"),
        "status": "PASS" if not leakage_errors else "FAIL",
        "split_names": list(split_names),
        "flight_id_overlap": [item for item in leakage_errors if item.startswith("flight_overlap")],
        "seed_or_scenario_overlap": [item for item in leakage_errors if "seed" in item],
        "duplicate_record_hash_count": len(duplicate_content_hashes),
        "duplicate_trajectory_pairs": duplicate_trajectory_pairs,
        "near_duplicate_trajectory_pairs": near_duplicate_trajectory_pairs,
        "all_errors": sorted(set(leakage_errors)),
        "sealed_test_opened": False,
    }
    write_json(validation_dir / "split_leakage_audit.json", leakage_audit)
    (validation_dir / "split_leakage_audit.md").write_text(
        "# Split Leakage Audit\n\n"
        f"Status: **{leakage_audit['status']}**\n\n"
        f"Splits: {', '.join(split_names)}\n\n"
        f"Duplicate record hashes: {len(duplicate_content_hashes)}  \n"
        f"Exact duplicate trajectories: {len(duplicate_trajectory_pairs)}  \n"
        f"Near-duplicate trajectories: {len(near_duplicate_trajectory_pairs)}  \n"
        f"Errors: {leakage_audit['all_errors'] or 'none'}\n\n"
        "The validator read serialized candidate records from disk. No sealed policy-test record was opened.\n",
        encoding="utf-8",
    )
    return report
