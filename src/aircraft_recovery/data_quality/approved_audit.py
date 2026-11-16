"""Independent one-to-one audit of an approved export against candidate source."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from aircraft_recovery.data_quality.common import (
    assert_not_sealed_path,
    file_hash,
    load_jsonl,
    object_hash,
    write_json,
)


def audit_approved_export(
    candidate_directory: str | Path,
    approved_directory: str | Path,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    source = assert_not_sealed_path(candidate_directory)
    approved = assert_not_sealed_path(approved_directory)
    source_records = load_jsonl(source / "raw" / "transitions.jsonl")
    source_flights = load_jsonl(source / "raw" / "flights.jsonl")
    approved_records = load_jsonl(approved / "raw" / "transitions.jsonl")
    approved_flights = load_jsonl(approved / "raw" / "flights.jsonl")
    report = json.loads((source / "validation" / "validation_report.json").read_text(encoding="utf-8"))
    manifest_path = approved / "manifests" / "approved_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    source_map = {row["record_id"]: row for row in source_records}
    approved_map = {row["record_id"]: row for row in approved_records}
    source_flight_map = {row["flight_id"]: row for row in source_flights}
    approved_flight_map = {row["flight_id"]: row for row in approved_flights}
    status = {row["record_id"]: row["status"] for row in report["record_results"]}
    errors: list[str] = []
    if len(source_map) != len(source_records):
        errors.append("duplicate_source_record_id")
    if len(approved_map) != len(approved_records):
        errors.append("duplicate_approved_record_id")
    if len({row["record_hash"] for row in approved_records}) != len(approved_records):
        errors.append("duplicate_approved_record_hash")
    expected_ids = {record_id for record_id, value in status.items() if value == "PASS"}
    if set(approved_map) - expected_ids:
        errors.append("unapproved_record_added")
    if expected_ids - set(approved_map):
        errors.append("approved_record_omitted")
    modified_ids = sorted(record_id for record_id in set(source_map) & set(approved_map)
                          if source_map[record_id] != approved_map[record_id])
    if modified_ids:
        errors.append("approved_record_value_mismatch")
    if set(approved_flight_map) != {source_map[record_id]["flight_id"] for record_id in approved_map}:
        errors.append("approved_flight_id_set_mismatch")
    modified_flights = sorted(flight_id for flight_id in set(source_flight_map) & set(approved_flight_map)
                              if source_flight_map[flight_id] != approved_flight_map[flight_id])
    if modified_flights:
        errors.append("approved_flight_value_mismatch")
    if manifest["included_record_ids"] != [row["record_id"] for row in approved_records]:
        errors.append("manifest_record_id_order_mismatch")
    if manifest["included_record_hashes"] != [row["record_hash"] for row in approved_records]:
        errors.append("manifest_record_hash_order_mismatch")
    source_hashes = manifest["source_hashes"]
    expected_source_hashes = {
        "generation_manifest_sha256": file_hash(source / "manifests" / "generation_manifest.json"),
        "raw_transitions_sha256": file_hash(source / "raw" / "transitions.jsonl"),
        "raw_flights_sha256": file_hash(source / "raw" / "flights.jsonl"),
        "validator_report_file_sha256": file_hash(source / "validation" / "validation_report.json"),
    }
    for name, expected in expected_source_hashes.items():
        if source_hashes.get(name) != expected:
            errors.append(f"source_hash_mismatch:{name}")
    checksum_path = approved / "manifests" / "file_checksums.json"
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        path = approved / relative
        if not path.is_file() or file_hash(path) != expected:
            errors.append(f"approved_file_checksum_mismatch:{relative}")
    field_equality = {
        "record_ids": set(approved_map) == expected_ids,
        "flight_ids": set(approved_flight_map) == set(source_flight_map),
        "feature_values": not modified_ids,
        "expert_proposed_actions": not modified_ids,
        "executed_controls": not modified_ids,
        "state_values": not modified_ids,
        "scenario_and_failure_metadata": not modified_ids,
        "sensor_metadata": not modified_ids,
        "actuator_metadata": not modified_ids,
    }
    audit: dict[str, Any] = {
        "audit_version": 1,
        "approved_dataset_id": manifest["approved_dataset_id"],
        "source_candidate_dataset_id": manifest["source_candidate_dataset_id"],
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "expected_flight_count": len(source_flights),
        "approved_flight_count": len(approved_flights),
        "expected_transition_count": len(expected_ids),
        "approved_transition_count": len(approved_records),
        "record_id_set_match": set(approved_map) == expected_ids,
        "flight_id_set_match": set(approved_flight_map) == set(source_flight_map),
        "duplicate_approved_record_ids": len(approved_records) - len(approved_map),
        "duplicate_approved_record_hashes": len(approved_records) - len({row["record_hash"] for row in approved_records}),
        "modified_record_ids": modified_ids,
        "modified_flight_ids": modified_flights,
        "unapproved_record_ids": sorted(set(approved_map) - expected_ids),
        "omitted_approved_record_ids": sorted(expected_ids - set(approved_map)),
        "field_equality": field_equality,
        "source_status_counts": dict(Counter(status.values())),
        "source_hashes_verified": not any(error.startswith("source_hash_mismatch") for error in errors),
        "approved_payload_checksums_verified": not any(error.startswith("approved_file_checksum_mismatch") for error in errors),
        "sealed_test_opened": False,
    }
    audit["audit_hash"] = object_hash(audit)
    if write_artifacts:
        validation = approved / "validation"
        validation.mkdir(exist_ok=False)
        json_path = validation / "approved_export_integrity_audit.json"
        markdown_path = validation / "approved_export_integrity_audit.md"
        write_json(json_path, audit)
        markdown_path.write_text(
            "# Approved Export Integrity Audit\n\n"
            f"Status: **{audit['status']}**\n\n"
            f"Flights: {audit['approved_flight_count']}/{audit['expected_flight_count']}  \n"
            f"Transitions: {audit['approved_transition_count']}/{audit['expected_transition_count']}  \n"
            f"Record IDs match: {audit['record_id_set_match']}  \n"
            f"Flight IDs match: {audit['flight_id_set_match']}  \n"
            f"Modified records: {len(modified_ids)}  \n"
            f"Duplicate records/hashes: {audit['duplicate_approved_record_ids']}/{audit['duplicate_approved_record_hashes']}  \n"
            f"Unapproved added: {len(audit['unapproved_record_ids'])}  \n"
            f"Approved omitted: {len(audit['omitted_approved_record_ids'])}  \n"
            f"Errors: {audit['errors'] or 'none'}\n\n"
            "The comparison is exact over complete serialized mappings; no simulator was invoked. "
            "No sealed-test record was opened.\n",
            encoding="utf-8",
        )
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        checksums[json_path.relative_to(approved).as_posix()] = file_hash(json_path)
        checksums[markdown_path.relative_to(approved).as_posix()] = file_hash(markdown_path)
        write_json(checksum_path, checksums)
    if errors:
        raise ValueError(f"Approved export integrity audit failed: {sorted(set(errors))}")
    return audit


def validate_pilot_approval_freeze(freeze_path: str | Path, approved_manifest: dict[str, Any]) -> None:
    """Fail if a Markdown approval freeze omits its immutable identities."""
    text = Path(freeze_path).read_text(encoding="utf-8")
    required = (
        approved_manifest["approved_dataset_id"],
        approved_manifest["source_candidate_dataset_id"],
        approved_manifest["approved_dataset_hash"],
        approved_manifest["validator_version"],
        approved_manifest["validator_report_hash"],
        approved_manifest["disposition_audit_hash"],
        approved_manifest["source_git_revision"],
        "sealed final test was untouched",
    )
    missing = [value for value in required if str(value) not in text]
    if missing:
        raise ValueError(f"Pilot approval freeze is incomplete: {missing}")
