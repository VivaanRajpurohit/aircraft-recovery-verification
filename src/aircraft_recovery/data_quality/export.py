"""Immutable export of explicitly accepted, independently validated records."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data_quality.common import (
    WORKFLOW_VERSION,
    assert_not_sealed_path,
    file_hash,
    load_jsonl,
    object_hash,
    write_json,
)


def export_approved_dataset(
    candidate_directory: str | Path,
    approved_dataset_id: str,
    output_root: str | Path = "datasets/approved_policy_data",
    failed_recovery_exclusion_justification: str | None = None,
    disposition_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    source = assert_not_sealed_path(candidate_directory)
    output = Path(output_root) / approved_dataset_id
    if output.exists():
        raise FileExistsError(f"Approved dataset already exists and is immutable: {output}")
    if disposition_manifest_path is not None:
        return _export_independently_dispositioned_dataset(
            source, output, approved_dataset_id, Path(disposition_manifest_path),
        )
    report = json.loads((source / "validation" / "validation_report.json").read_text(encoding="utf-8"))
    report_hash = object_hash({key: value for key, value in report.items() if key != "validator_report_hash"})
    if report_hash != report.get("validator_report_hash"):
        raise ValueError("Validation report hash mismatch")
    review_manifest = json.loads((source / "review" / "review_manifest.json").read_text(encoding="utf-8"))
    review_hash = object_hash({
        key: value for key, value in review_manifest.items() if key != "review_manifest_hash"
    })
    if review_hash != review_manifest.get("review_manifest_hash"):
        raise ValueError("Review manifest hash mismatch")
    if review_manifest["validator_report_hash"] != report_hash:
        raise ValueError("Review was performed against a different validator report")
    decisions_path = source / "review" / "decisions.jsonl"
    if file_hash(decisions_path) != review_manifest["decisions_file_hash"]:
        raise ValueError("Review decisions hash mismatch")
    records = load_jsonl(source / "raw" / "transitions.jsonl")
    decisions = load_jsonl(decisions_path)
    latest: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        latest[decision["record_id"]] = decision
    status = {row["record_id"]: row["status"] for row in report["record_results"]}
    included = [
        row for row in records
        if latest.get(row["record_id"], {}).get("decision") == "ACCEPT"
        and status.get(row["record_id"]) != "FAIL"
    ]
    if not included:
        raise ValueError("No explicitly accepted non-failing records are eligible for export")
    included_ids = {row["record_id"] for row in included}
    excluded_valid_failures = [
        row for row in records
        if row["record_id"] not in included_ids
        and row.get("outcome_label") == "unsuccessful"
        and status.get(row["record_id"]) != "FAIL"
    ]
    if excluded_valid_failures and not failed_recovery_exclusion_justification:
        raise ValueError(
            "A written failed-recovery exclusion justification is required; valid unsuccessful "
            "records are controller-performance evidence, not automatically bad data"
        )
    train_records = [row for row in included if row["split"] == "train"]
    if not train_records:
        raise ValueError("Approved export requires at least one accepted training record for normalization")
    observations = np.asarray([row["ordered_observation_42"] for row in train_records], dtype=float)
    normalization = {
        "fit_split": "train",
        "source_record_ids": [row["record_id"] for row in train_records],
        "feature_order": list(OBSERVATION_FEATURES),
        "mean": observations.mean(axis=0).tolist(),
        "standard_deviation": np.where(observations.std(axis=0) < 1e-6, 1.0, observations.std(axis=0)).tolist(),
        "validation_or_test_records_used": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    manifests = output / "manifests"
    raw.mkdir()
    manifests.mkdir()
    transitions = raw / "transitions.jsonl"
    transitions.write_text(
        "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in included),
        encoding="utf-8",
    )
    write_json(manifests / "normalization_provenance.json", normalization)
    excluded: list[dict[str, Any]] = []
    for row in records:
        if row in included:
            continue
        decision = latest.get(row["record_id"])
        if status.get(row["record_id"]) == "FAIL":
            reason = "fatal_validation_failure"
        elif decision:
            reason = f"human_{decision['decision'].lower()}:{decision.get('reason_code') or 'no_reason'}"
        else:
            reason = "unreviewed"
        excluded.append({"record_id": row["record_id"], "record_hash": row["record_hash"], "reason": reason})
    split_construction = {
        name: [row["record_id"] for row in included if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    manifest: dict[str, Any] = {
        "approved_dataset_id": approved_dataset_id,
        "source_candidate_dataset_id": report["dataset_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exporter_version": WORKFLOW_VERSION,
        "validator_version": report["validator_version"],
        "validator_report_hash": report_hash,
        "review_manifest_hash": review_manifest["review_manifest_hash"],
        "export_configuration": {
            "explicit_human_accept_required": True,
            "fatal_validation_failures_excluded": True,
            "sealed_test_excluded": True,
            "failed_recovery_exclusion_justification": failed_recovery_exclusion_justification,
        },
        "included_record_count": len(included),
        "included_record_hashes": [row["record_hash"] for row in included],
        "excluded_records": excluded,
        "split_construction": split_construction,
        "feature_order": list(OBSERVATION_FEATURES),
        "normalization_provenance": "manifests/normalization_provenance.json",
        "source_review_bias_summary": json.loads((source / "review" / "review_summary.json").read_text(encoding="utf-8")),
        "raw_candidate_records_preserved": True,
        "valid_unsuccessful_records_excluded": len(excluded_valid_failures),
    }
    manifest["approved_dataset_hash"] = object_hash({
        "source": report["dataset_id"],
        "included": manifest["included_record_hashes"],
        "splits": split_construction,
        "normalization": normalization,
    })
    payload_checksums = {
        str(path.relative_to(output)).replace("\\", "/"): file_hash(path)
        for path in (transitions, manifests / "normalization_provenance.json")
    }
    manifest["payload_file_checksums"] = payload_checksums
    write_json(manifests / "approved_manifest.json", manifest)
    checksums = {
        str(path.relative_to(output)).replace("\\", "/"): file_hash(path)
        for path in (
            transitions, manifests / "normalization_provenance.json",
            manifests / "approved_manifest.json",
        )
    }
    write_json(manifests / "file_checksums.json", checksums)
    manifest["file_checksums"] = checksums
    return manifest


def _export_independently_dispositioned_dataset(
    source: Path,
    output: Path,
    approved_dataset_id: str,
    disposition_path: Path,
) -> dict[str, Any]:
    """Export validator-PASS records backed by a hashed independent disposition audit."""
    report_path = source / "validation" / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_hash = object_hash({key: value for key, value in report.items() if key != "validator_report_hash"})
    if report_hash != report.get("validator_report_hash"):
        raise ValueError("Validation report hash mismatch")
    dispositions = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition_hash = object_hash({key: value for key, value in dispositions.items() if key != "manifest_hash"})
    if disposition_hash != dispositions.get("manifest_hash"):
        raise ValueError("Disposition manifest hash mismatch")
    if dispositions.get("validator_version") != report.get("validator_version"):
        raise ValueError("Disposition manifest targets a different validator version")
    disposition_rows = dispositions.get("dispositions", [])
    if any(row.get("disposition") != "APPROVE" for row in disposition_rows):
        raise ValueError("Disposition manifest contains a non-approved record")

    records = load_jsonl(source / "raw" / "transitions.jsonl")
    flights = load_jsonl(source / "raw" / "flights.jsonl")
    record_map = {row["record_id"]: row for row in records}
    if len(record_map) != len(records):
        raise ValueError("Candidate records are not one-to-one by record ID")
    status = {row["record_id"]: row["status"] for row in report["record_results"]}
    if set(status) != set(record_map):
        raise ValueError("Validator report and candidate record IDs differ")
    disposition_ids = [str(row.get("record_id")) for row in disposition_rows]
    if len(disposition_ids) != len(set(disposition_ids)) or not set(disposition_ids) <= set(record_map):
        raise ValueError("Disposition records are duplicated or absent from the candidate dataset")
    if any(status[record_id] != "PASS" for record_id in disposition_ids):
        raise ValueError("A disposition attempts to approve a non-PASS validator record")
    included = [row for row in records if status[row["record_id"]] == "PASS"]
    excluded = [
        {"record_id": row["record_id"], "record_hash": row["record_hash"], "reason": status[row["record_id"]]}
        for row in records if status[row["record_id"]] != "PASS"
    ]
    if any(status_value in {"FAIL", "REVIEW_REQUIRED"} for status_value in status.values()):
        raise ValueError("Validated export requires zero FAIL and zero REVIEW_REQUIRED records")

    included_flight_ids = {row["flight_id"] for row in included}
    included_flights = [row for row in flights if row["flight_id"] in included_flight_ids]
    if len(included_flights) != len(included_flight_ids):
        raise ValueError("Approved transitions do not map one-to-one to flight metadata")
    generation_path = source / "manifests" / "generation_manifest.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    schema_path = source / "manifests" / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    train_records = [row for row in included if row["split"] == "train"]
    if not train_records:
        raise ValueError("Approved export requires training records for normalization provenance")
    observations = np.asarray([row["ordered_observation_42"] for row in train_records], dtype=float)
    normalization = {
        "fit_split": "train",
        "source_record_ids": [row["record_id"] for row in train_records],
        "feature_order": list(OBSERVATION_FEATURES),
        "mean": observations.mean(axis=0).tolist(),
        "standard_deviation": np.where(observations.std(axis=0) < 1e-6, 1.0, observations.std(axis=0)).tolist(),
        "validation_or_test_records_used": False,
        "pilot_only_not_final_training_preprocessing": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    manifests = output / "manifests"
    raw.mkdir()
    manifests.mkdir()
    transitions_path = raw / "transitions.jsonl"
    flights_path = raw / "flights.jsonl"
    transitions_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in included),
        encoding="utf-8",
    )
    flights_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in included_flights),
        encoding="utf-8",
    )
    normalization_path = manifests / "normalization_provenance.json"
    write_json(normalization_path, normalization)
    split_names = ("train", "development", "challenge", "ood_stress", "validation", "test")
    split_construction = {
        name: [row["record_id"] for row in included if row["split"] == name]
        for name in split_names if any(row["split"] == name for row in included)
    }
    source_hashes = {
        "generation_manifest_sha256": file_hash(generation_path),
        "raw_transitions_sha256": file_hash(source / "raw" / "transitions.jsonl"),
        "raw_flights_sha256": file_hash(source / "raw" / "flights.jsonl"),
        "validator_report_file_sha256": file_hash(report_path),
        "validator_logical_hash": report_hash,
        "disposition_manifest_file_sha256": file_hash(disposition_path),
        "disposition_audit_logical_hash": disposition_hash,
    }
    manifest: dict[str, Any] = {
        "approved_dataset_id": approved_dataset_id,
        "source_candidate_dataset_id": report["dataset_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exporter_version": "approved-pilot-export-v1",
        "source_git_revision": generation["git_commit_hash"],
        "schema_version": schema["schema_version"],
        "source_hashes": source_hashes,
        "validator_version": report["validator_version"],
        "validator_report_hash": report_hash,
        "disposition_audit_hash": disposition_hash,
        "disposition_counts": dispositions["counts"],
        "approved_flight_count": len(included_flights),
        "approved_transition_count": len(included),
        "rejected_flight_count": len(flights) - len(included_flights),
        "rejected_transition_count": len(excluded),
        "included_record_count": len(included),
        "included_record_ids": [row["record_id"] for row in included],
        "included_record_hashes": [row["record_hash"] for row in included],
        "excluded_records": excluded,
        "feature_contract": generation["feature_contract"],
        "action_contract": generation["action_contract"],
        "scenario_inventory": generation["scenario_inventory"],
        "difficulty_tier_inventory": generation["tier_flight_counts"],
        "scenario_family_inventory": generation["family_flight_counts"],
        "split_construction": split_construction,
        "normalization_provenance": "manifests/normalization_provenance.json",
        "raw_candidate_records_preserved": True,
        "sealed_test_opened": False,
        "pilot_only_not_final_training_corpus": True,
    }
    manifest["approved_dataset_hash"] = object_hash({
        "source_hashes": source_hashes,
        "included_record_ids": manifest["included_record_ids"],
        "included_record_hashes": manifest["included_record_hashes"],
        "flight_ids": [row["flight_id"] for row in included_flights],
        "splits": split_construction,
    })
    manifest_path = manifests / "approved_manifest.json"
    write_json(manifest_path, manifest)
    checksums = {
        path.relative_to(output).as_posix(): file_hash(path)
        for path in (transitions_path, flights_path, normalization_path, manifest_path)
    }
    write_json(manifests / "file_checksums.json", checksums)
    manifest["file_checksums"] = checksums
    return manifest
