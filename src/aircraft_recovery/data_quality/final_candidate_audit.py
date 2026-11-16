"""Post-generation integrity audits for the frozen v2.1 candidate dataset.

This module intentionally never evaluates controller outcomes.  Sealed rows are
used only for identifiers, frozen design fields, hashes, and provenance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from aircraft_recovery.data_quality.common import file_hash, object_hash, write_json
from aircraft_recovery.data_quality.final_generation import (
    ACTIVE_FREEZE_HASH,
    DATASET_ID,
    SPLITS,
    build_final_flight_registry,
)
from aircraft_recovery.research.holdout_audit import build_exposure_matrix


AUDITOR_VERSION = "final-candidate-auditor-v1-design-integrity-only"
SEALED = "final_sealed_test"
DESIGN_FIELDS = (
    "component_severity", "activation_seconds", "actuator_delay_seconds",
    "sensor_bias_kind", "sensor_bias_value", "turbulence_intensity",
    "crosswind_maximum_kts", "engine_side", "initial_grid_fraction",
    "terrain_clearance_ft", "warmup_vertical_speed_bin",
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter("null" if value is None else str(value) for value in values).items()))


def _pairwise_overlaps(values: dict[str, set[Any]]) -> dict[str, int]:
    return {
        f"{left}:{right}": len(values[left] & values[right])
        for left, right in combinations(SPLITS, 2)
    }


def _realized_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(row["split"] for row in rows)
    report: dict[str, Any] = {
        "flight_count": len(rows),
        "transition_count": sum(int(row["transition_count"]) for row in rows),
        "split_counts": dict(by_split),
        "tier_counts": _counts(row["difficulty_tier"] for row in rows),
        "family_counts": _counts(row["scenario_family"] for row in rows),
        "vertical_speed_bins": _counts(row["warmup_vertical_speed_bin"] for row in rows),
        "vertical_speed_bins_by_split": {
            split: _counts(row["warmup_vertical_speed_bin"] for row in rows if row["split"] == split)
            for split in SPLITS
        },
        "vertical_speed_bins_by_tier": {
            str(tier): _counts(row["warmup_vertical_speed_bin"] for row in rows if row["difficulty_tier"] == tier)
            for tier in range(6)
        },
        "vertical_speed_bins_by_family": {
            family: _counts(row["warmup_vertical_speed_bin"] for row in rows if row["scenario_family"] == family)
            for family in sorted({row["scenario_family"] for row in rows})
        },
    }
    for field in DESIGN_FIELDS:
        report[field] = _counts(row[field] for row in rows)
    for field in ("airspeed_kts", "roll_deg", "pitch_deg", "altitude_ft"):
        report[f"initial_{field}"] = _counts(row["initial_state"][field] for row in rows)
    report["derived_actuator_effectiveness"] = _counts(
        None if row["component_severity"] is None else round(1.0 - row["component_severity"], 12)
        for row in rows if "aileron" in row["scenario_family"] or "rudder" in row["scenario_family"]
    )
    report["derived_partial_thrust_fraction"] = _counts(
        None if row["component_severity"] is None else round(1.0 - row["component_severity"], 12)
        for row in rows if "partial_thrust" in row["scenario_family"]
    )
    return report


def _audit_leakage(dataset: Path, flights: list[dict[str, Any]]) -> dict[str, Any]:
    sets: dict[str, dict[str, set[Any]]] = {
        name: {split: set() for split in SPLITS}
        for name in (
            "flight_ids", "record_ids", "seeds", "scenario_instance_ids",
            "transition_hashes", "flight_hashes", "canonical_scenario_signatures",
            "source_configuration_hashes", "effective_scenario_configuration_hashes",
        )
    }
    for row in flights:
        split = row["split"]
        sets["flight_ids"][split].add(row["flight_id"])
        sets["seeds"][split].add(row["random_seed"])
        sets["scenario_instance_ids"][split].add(row["scenario_instance_id"])
        sets["flight_hashes"][split].add(row["flight_hash"])
        sets["canonical_scenario_signatures"][split].add(row["canonical_scenario_signature"])
        sets["source_configuration_hashes"][split].add(row["source_configuration_hash"])

    record_count = 0
    duplicate_record_ids = 0
    duplicate_record_hashes = 0
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for row in _read_jsonl(dataset / "raw/transitions.jsonl"):
        record_count += 1
        split = row["split"]
        record_id = row["record_id"]
        transition_hash = row["record_hash"]
        duplicate_record_ids += record_id in seen_ids
        duplicate_record_hashes += transition_hash in seen_hashes
        seen_ids.add(record_id)
        seen_hashes.add(transition_hash)
        sets["record_ids"][split].add(record_id)
        sets["transition_hashes"][split].add(transition_hash)
        sets["effective_scenario_configuration_hashes"][split].add(row["effective_scenario_configuration_hash"])

    overlaps = {name: _pairwise_overlaps(split_sets) for name, split_sets in sets.items()}
    unique_contracts = (
        "flight_ids", "record_ids", "seeds", "scenario_instance_ids",
        "transition_hashes", "flight_hashes",
    )
    prohibited = [
        f"{name}:{pair}:{count}"
        for name in unique_contracts
        for pair, count in overlaps[name].items() if count
    ]
    sealed_signature_overlaps = {
        split: len(sets["canonical_scenario_signatures"][SEALED] & sets["canonical_scenario_signatures"][split])
        for split in SPLITS if split != SEALED
    }
    prohibited.extend(
        f"sealed_signature:{split}:{count}"
        for split, count in sealed_signature_overlaps.items() if count
    )
    if duplicate_record_ids:
        prohibited.append(f"duplicate_record_ids:{duplicate_record_ids}")
    if duplicate_record_hashes:
        prohibited.append(f"duplicate_transition_hashes:{duplicate_record_hashes}")
    return {
        "audit_version": AUDITOR_VERSION,
        "dataset_id": DATASET_ID,
        "status": "PASS" if not prohibited else "FAIL",
        "record_count_scanned": record_count,
        "overlaps": overlaps,
        "sealed_signature_overlap_counts": sealed_signature_overlaps,
        "source_hash_overlap_classification": "EXPECTED: splits use the same frozen source scenario templates",
        "effective_configuration_overlap_classification": "REPORTED: not an identity key; exact scenario signature is the combination contract",
        "duplicate_record_id_count": duplicate_record_ids,
        "duplicate_transition_hash_count": duplicate_record_hashes,
        "prohibited_overlaps": prohibited,
        "ambiguous_overlaps": [],
    }


def _flight_statuses(validation: dict[str, Any], flights: list[dict[str, Any]]) -> dict[str, int]:
    status = validation["record_status_counts"]
    if status["FAIL"] == 0 and status["REVIEW_REQUIRED"] == 0 and status["PASS"] == validation["record_count"]:
        return {"PASS": len(flights), "FAIL": 0, "REVIEW_REQUIRED": 0}
    raise ValueError("non-PASS records require explicit grouped flight-level review")


def audit_final_candidate(root: Path) -> dict[str, Any]:
    dataset = root / "datasets/candidate_policy_data/final_policy_candidate_v2_1"
    manifests = dataset / "manifests"
    validation_dir = dataset / "validation"
    generation = json.loads((manifests / "generation_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((validation_dir / "validation_report.json").read_text(encoding="utf-8"))
    frozen = json.loads((dataset / "frozen_flight_registry.json").read_text(encoding="utf-8"))
    realized = list(_read_jsonl(dataset / "raw/flights.jsonl"))
    expected = build_final_flight_registry(root)
    frozen_hashes = [row["flight_spec_hash"] for row in frozen]
    expected_hashes = [row["flight_spec_hash"] for row in expected]
    realized_hashes = [row["flight_spec_hash"] for row in realized]
    if frozen_hashes != expected_hashes or realized_hashes != expected_hashes:
        raise ValueError("serialized flight registry does not exactly match the active frozen design")

    leakage = _audit_leakage(dataset, realized)
    write_json(validation_dir / "final_leakage_audit.json", leakage)

    static_holdout = build_exposure_matrix(root, use_v2_1=True)
    holdout = {
        "audit_version": AUDITOR_VERSION,
        "status": "PASS" if static_holdout["status"] == "PASS" and leakage["status"] == "PASS" else "FAIL",
        "realized_registry_exactly_matches_frozen_registry": True,
        "strict_sealed_collisions": static_holdout["strict_sealed_collisions"],
        "model_selection_collisions": static_holdout["model_selection_collisions"],
        "sealed_signature_duplicate_counts": leakage["sealed_signature_overlap_counts"],
        "seed_overlap_counts": leakage["overlaps"]["seeds"],
        "holdout_semantics_version": static_holdout["schedule_id"],
        "frozen_exposure_audit_hash": static_holdout["audit_hash"],
    }
    write_json(validation_dir / "final_holdout_compliance_audit.json", holdout)

    nonsealed = [row for row in realized if row["split"] != SEALED]
    sealed = [row for row in realized if row["split"] == SEALED]
    distribution = {
        "audit_version": AUDITOR_VERSION,
        "nonsealed": _realized_distributions(nonsealed),
        "sealed_design_and_integrity_only": _realized_distributions(sealed),
        "sealed_performance_metrics_present": False,
    }
    write_json(manifests / "realized_distribution_report.json", distribution)
    (manifests / "realized_distribution_report.md").write_text(
        "# Realized candidate distribution\n\n"
        f"- Nonsealed flights: {len(nonsealed)}\n"
        f"- Sealed flights (design/integrity counts only): {len(sealed)}\n"
        f"- Total transitions: {sum(row['transition_count'] for row in realized)}\n"
        f"- Warmup vertical-speed bins: `{json.dumps(distribution['nonsealed']['vertical_speed_bins'], sort_keys=True)}`\n"
        "- Sealed controller outcomes were not analyzed.\n",
        encoding="utf-8",
    )

    sealed_boundary = {
        "audit_version": AUDITOR_VERSION,
        "status": "PASS",
        "sealed_flight_count": len(sealed),
        "allowed_operations_performed": [
            "schema_validation", "hash_validation", "count_validation",
            "provenance_validation", "holdout_contract_validation", "duplicate_and_leakage_checks",
        ],
        "outcome_analysis_routines_invoked": [],
        "performance_metrics_computed": [],
        "validator_suppressed_splits": validation["quality_summary"]["outcome_summary_suppressed_splits"],
        "sealed_fields_accessed_by_post_generation_auditor": [
            "identity", "split", "seed", "tier", "family", "frozen_design_parameters",
            "initial_condition_parameters", "warmup_vertical_speed_bin", "provenance_hashes", "integrity_hashes",
        ],
        "sealed_controller_performance_accessed": False,
        "approved_training_export_created": False,
        "normalization_fitted": False,
        "training_invoked": False,
    }
    write_json(validation_dir / "sealed_boundary_audit.json", sealed_boundary)

    record_status = validation["record_status_counts"]
    flight_status = _flight_statuses(validation, realized)
    root_causes = {
        "status": "NOT_REQUIRED" if record_status["FAIL"] == record_status["REVIEW_REQUIRED"] == 0 else "REQUIRED",
        "groups": [],
        "record_status_counts": record_status,
        "flight_status_counts": flight_status,
    }
    write_json(validation_dir / "nonpass_root_causes.json", root_causes)

    checksums = json.loads((manifests / "file_checksums.json").read_text(encoding="utf-8"))
    dataset_logical_hash = object_hash({
        "dataset_id": DATASET_ID,
        "freeze_hash": ACTIVE_FREEZE_HASH,
        "flight_spec_hashes": realized_hashes,
        "consolidated_file_hashes": {
            "raw/flights.jsonl": checksums["raw/flights.jsonl"],
            "raw/transitions.jsonl": checksums["raw/transitions.jsonl"],
        },
        "validator_report_hash": validation["validator_report_hash"],
    })
    (manifests / "candidate_dataset_manifest.md").write_text(
        "# Final policy candidate v2.1\n\n"
        "- Status: CANDIDATE_AWAITING_EXPLICIT_FINAL_DATA_APPROVAL\n"
        f"- Freeze: `{ACTIVE_FREEZE_HASH}`\n"
        f"- Flights/transitions: {len(realized)} / {sum(row['transition_count'] for row in realized)}\n"
        f"- Record validation: `{json.dumps(record_status, sort_keys=True)}`\n"
        f"- Flight validation: `{json.dumps(flight_status, sort_keys=True)}`\n"
        f"- Leakage/holdout/sealed boundary: {leakage['status']} / {holdout['status']} / {sealed_boundary['status']}\n"
        f"- Dataset logical hash: `{dataset_logical_hash}`\n"
        "- This is not an approved training export. No normalization or training was run.\n",
        encoding="utf-8",
    )
    hash_exclusions = {"manifests/candidate_dataset_manifest.json"}
    artifact_hashes = {
        path.relative_to(dataset).as_posix(): file_hash(path)
        for path in sorted(dataset.rglob("*"))
        if path.is_file() and path.relative_to(dataset).as_posix() not in hash_exclusions
    }
    batch_hashes = {
        path.parent.parent.name: file_hash(path)
        for path in sorted((dataset / ".batches").glob("batch_*/manifests/batch_manifest.json"))
    }
    started = datetime.fromisoformat(generation["generation_started_utc"])
    completed = datetime.fromisoformat(generation["generation_completed_utc"])
    registry = json.loads((root / "artifacts/research_stage/final_generation_freeze_registry.json").read_text(encoding="utf-8"))
    machine_manifest = {
        "manifest_version": "final-candidate-manifest-v1",
        "dataset_id": DATASET_ID,
        "status": "CANDIDATE_AWAITING_EXPLICIT_FINAL_DATA_APPROVAL",
        "generation_freeze_id": generation["generation_freeze_id"],
        "generation_freeze_hash": ACTIVE_FREEZE_HASH,
        "generation_freeze_registry_hash": registry["registry_hash_excluding_this_field"],
        "seed_registry_hash": file_hash(root / "configs/research/final_seed_registry_v1.json"),
        "dataset_design_hash": file_hash(root / "configs/training/final_policy_dataset_design_v1.yaml"),
        "holdout_semantics_version": holdout["holdout_semantics_version"],
        "simulator_source_revision": generation["git_commit_hash"],
        "expert_controller_version": file_hash(root / "configs/research/final_fallback_v1.yaml"),
        "generation_started_utc": generation["generation_started_utc"],
        "generation_completed_utc": generation["generation_completed_utc"],
        "generation_duration_seconds": (completed - started).total_seconds(),
        "flight_count": generation["flight_count"],
        "transition_count": generation["record_count"],
        "split_counts": {split: len(generation["splits"][split]) for split in SPLITS},
        "tier_counts": generation["tier_flight_counts"],
        "family_counts": generation["family_flight_counts"],
        "engine_side_counts": _counts(row["engine_side"] for row in realized if row["engine_side"]),
        "validator_version": validation["validator_version"],
        "validator_report_hash": validation["validator_report_hash"],
        "record_status_counts": record_status,
        "flight_status_counts": flight_status,
        "leakage_audit_hash": file_hash(validation_dir / "final_leakage_audit.json"),
        "holdout_audit_hash": file_hash(validation_dir / "final_holdout_compliance_audit.json"),
        "sealed_boundary_audit_hash": file_hash(validation_dir / "sealed_boundary_audit.json"),
        "artifact_hashes": artifact_hashes,
        "artifact_hash_scope": "every dataset file present at manifest construction",
        "artifact_hash_exclusions": {
            "manifests/candidate_dataset_manifest.json": "self-referential manifest cannot hash itself",
        },
        "batch_manifest_hashes": batch_hashes,
        "disk_usage_bytes_before_final_manifest": sum(path.stat().st_size for path in dataset.rglob("*") if path.is_file()),
        "approved_training_export_created": False,
    }
    machine_manifest["dataset_logical_hash"] = dataset_logical_hash
    write_json(manifests / "candidate_dataset_manifest.json", machine_manifest)
    return machine_manifest
