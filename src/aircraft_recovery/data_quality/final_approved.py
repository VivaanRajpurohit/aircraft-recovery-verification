"""Final nonsealed approval, exact-copy export, and train-only preprocessing."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data.preprocessing import BINARY_OR_MASK_FEATURES
from aircraft_recovery.data_quality.common import CONTROL_NAMES, file_hash, object_hash, write_json


CANDIDATE_ID = "final_policy_candidate_v2_1"
APPROVED_ID = "final_policy_approved_v2_1"
CANDIDATE_LOGICAL_HASH = "4db2d75ab341e4573019d09176507b03851d7d26dd2eb360d42ecfc2adc28706"
FREEZE_ID = "final_generation_freeze_v2_1"
FREEZE_HASH = "c0b831ede8a5d75027dcce70ec1c116fc5ec58d479cf500939f2075e3d16d0dc"
TRAINING_ELIGIBLE_SPLITS = ("train", "development", "challenge", "ood_stress")
SEALED_SPLIT = "final_sealed_test"
ROLE_SPLITS = {
    "normalization": frozenset({"train"}),
    "training": frozenset({"train"}),
    "model_selection": frozenset({"development"}),
    "challenge_evaluation": frozenset({"challenge"}),
    "ood_evaluation": frozenset({"ood_stress"}),
}
AUDITOR_VERSION = "final-approved-export-auditor-v1-streaming-exact-copy"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_role_split(role: str, split: str) -> None:
    if split == SEALED_SPLIT:
        raise PermissionError("final_sealed_test is forbidden to training and model-selection loaders")
    allowed = ROLE_SPLITS.get(role)
    if allowed is None or split not in allowed:
        raise PermissionError(f"split {split!r} is not permitted for loader role {role!r}")


def validate_approved_manifest_contract(manifest: dict[str, Any]) -> None:
    expected_splits = {"train": 576, "development": 144, "challenge": 144, "ood_stress": 144}
    errors = []
    if manifest.get("approved_dataset_id") != APPROVED_ID:
        errors.append("approved_dataset_id")
    if manifest.get("candidate_logical_hash") != CANDIDATE_LOGICAL_HASH:
        errors.append("candidate_logical_hash")
    if manifest.get("approved_flight_count") != 1008 or manifest.get("approved_transition_count") != 262080:
        errors.append("approved_counts")
    if manifest.get("split_counts") != expected_splits:
        errors.append("split_counts")
    if manifest.get("sealed_records_in_approved_export") != 0:
        errors.append("sealed_records")
    feature_names = manifest.get("feature_order_contract", {}).get("order")
    if feature_names != list(OBSERVATION_FEATURES):
        errors.append("feature_order_contract")
    action_names = manifest.get("action_contract", {}).get("order")
    if action_names != list(CONTROL_NAMES):
        errors.append("action_contract")
    if errors:
        raise ValueError(f"approved manifest contract failed: {errors}")


def validate_normalization_contract(statistics: dict[str, Any]) -> None:
    errors = []
    if statistics.get("fit_split") != "train":
        errors.append("fit_split")
    if statistics.get("fitted_transition_count") != 149760 or statistics.get("fitted_flight_count") != 576:
        errors.append("fit_counts")
    if statistics.get("feature_names") != list(OBSERVATION_FEATURES):
        errors.append("feature_order")
    excluded = {"development", "challenge", "ood_stress", SEALED_SPLIT}
    if set(statistics.get("excluded_splits", [])) != excluded:
        errors.append("excluded_splits")
    if statistics.get("excluded_split_record_counts_used") != {name: 0 for name in excluded}:
        errors.append("excluded_split_usage")
    declared = statistics.get("normalization_artifact_hash")
    payload = {key: value for key, value in statistics.items() if key != "normalization_artifact_hash"}
    if declared != object_hash(payload):
        errors.append("normalization_hash")
    if errors:
        raise ValueError(f"normalization contract failed: {errors}")


def validate_approved_freeze(path: Path, manifest: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    required = (
        manifest["approved_dataset_id"], manifest["approved_dataset_logical_hash"],
        manifest["candidate_logical_hash"], manifest["active_freeze_hash"],
        manifest["export_integrity_audit_hash"], manifest["source_git_revision"],
        "144 / 37440", "excluded from training, normalization, and model selection",
    )
    missing = [value for value in required if str(value) not in text]
    if missing:
        raise ValueError(f"approved dataset freeze is incomplete: {missing}")


def verify_approved_checksums(approved: Path) -> None:
    checksums = _json(approved / "manifests/file_checksums.json")
    mismatches = [
        relative for relative, expected in checksums.items()
        if not (approved / relative).is_file() or file_hash(approved / relative) != expected
    ]
    if mismatches:
        raise ValueError(f"approved artifact checksum mismatch: {mismatches}")


def verify_candidate_source_immutable(root: Path, approved: Path) -> None:
    candidate = root / "datasets/candidate_policy_data" / CANDIDATE_ID
    candidate_manifest = _json(candidate / "manifests/candidate_dataset_manifest.json")
    provenance = _json(approved / "manifests/export_provenance.json")
    current = {
        "transitions": file_hash(candidate / "raw/transitions.jsonl"),
        "flights": file_hash(candidate / "raw/flights.jsonl"),
    }
    expected = {
        "transitions": candidate_manifest["artifact_hashes"]["raw/transitions.jsonl"],
        "flights": candidate_manifest["artifact_hashes"]["raw/flights.jsonl"],
    }
    if current != expected or current != provenance["candidate_raw_hashes_before_export"]:
        raise ValueError("candidate source changed before, during, or after approved export")


def iter_approved_records(dataset: str | Path, *, split: str, role: str) -> Iterable[dict[str, Any]]:
    """Yield only the split explicitly allowed for a future loader role."""
    assert_role_split(role, split)
    path = Path(dataset) / "raw/transitions.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == SEALED_SPLIT:
                raise PermissionError("sealed record detected inside approved training export")
            if row["split"] == split:
                yield row


def create_final_approval_record(root: Path) -> dict[str, Any]:
    candidate = root / "datasets/candidate_policy_data" / CANDIDATE_ID
    manifest = _json(candidate / "manifests/candidate_dataset_manifest.json")
    validation = _json(candidate / "validation/validation_report.json")
    leakage = candidate / "validation/final_leakage_audit.json"
    holdout = candidate / "validation/final_holdout_compliance_audit.json"
    sealed = candidate / "validation/sealed_boundary_audit.json"
    if manifest["dataset_logical_hash"] != CANDIDATE_LOGICAL_HASH:
        raise ValueError("candidate logical hash changed since approval authorization")
    if manifest["record_status_counts"] != {"PASS": 299520, "FAIL": 0, "REVIEW_REQUIRED": 0}:
        raise ValueError("candidate validation status is no longer approval-eligible")
    if _json(leakage)["status"] != "PASS" or _json(holdout)["status"] != "PASS" or _json(sealed)["status"] != "PASS":
        raise ValueError("candidate approval audits are not all PASS")
    record: dict[str, Any] = {
        "record_id": "final_policy_candidate_v2_1_data_approval",
        "dataset_id": CANDIDATE_ID,
        "dataset_logical_hash": CANDIDATE_LOGICAL_HASH,
        "source_freeze_id": FREEZE_ID,
        "source_freeze_hash": FREEZE_HASH,
        "validator_version": validation["validator_version"],
        "validator_report_hash": validation["validator_report_hash"],
        "leakage_audit_hash": file_hash(leakage),
        "holdout_audit_hash": file_hash(holdout),
        "sealed_boundary_audit_hash": file_hash(sealed),
        "flight_count": 1152,
        "transition_count": 299520,
        "record_status_counts": {"PASS": 299520, "FAIL": 0, "REVIEW_REQUIRED": 0},
        "review_status": "reviewed",
        "approval_status": "approved",
        "review_method": "independent_contract_review",
        "approval_method": "independent_contract_review",
        "approval_timestamp_utc": _utc(),
        "statement": (
            "The candidate underwent independent contract-based review and passed all required "
            "integrity, validation, leakage, provenance, and holdout checks."
        ),
        "approved_scope": list(TRAINING_ELIGIBLE_SPLITS),
        "sealed_split_approval": "excluded_from_training_export",
    }
    record["approval_record_hash"] = object_hash(record)
    path = root / "artifacts/research_stage/final_policy_candidate_v2_1_approval_record.json"
    write_json(path, record)
    return record


def _copy_eligible_jsonl(source: Path, destination: Path, *, kind: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    families: Counter[str] = Counter()
    record_hashes: list[str] = []
    item_hashes: list[str] = []
    sealed = 0
    seen_ids: set[str] = set()
    id_field = "record_id" if kind == "transition" else "flight_id"
    hash_field = "record_hash" if kind == "transition" else "flight_hash"
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        for line in input_stream:
            row = json.loads(line)
            split = row["split"]
            if split == SEALED_SPLIT:
                sealed += 1
                continue
            if split not in TRAINING_ELIGIBLE_SPLITS:
                raise ValueError(f"unexpected split in candidate: {split}")
            identifier = row[id_field]
            if identifier in seen_ids:
                raise ValueError(f"duplicate {id_field}: {identifier}")
            seen_ids.add(identifier)
            if kind == "transition" and SEALED_SPLIT.encode() in line:
                raise ValueError("fatal sealed-record guard triggered during approved export")
            output_stream.write(line)
            counts[split] += 1
            tiers[str(row["difficulty_tier"])] += 1
            families[row["scenario_family"]] += 1
            item_hashes.append(row[hash_field])
            if kind == "transition":
                record_hashes.append(row["record_hash"])
    return {
        "counts": dict(counts), "tier_counts": dict(tiers), "family_counts": dict(families),
        "item_hashes": item_hashes, "record_hashes": record_hashes,
        "sealed_excluded": sealed, "included": sum(counts.values()),
    }


def _next_line(stream: BinaryIO) -> bytes | None:
    value = stream.readline()
    return value if value else None


def _audit_exact_copy(candidate_path: Path, approved_path: Path, *, kind: str) -> dict[str, Any]:
    approved_count = 0
    sealed_count = 0
    mismatch_count = 0
    omitted_count = 0
    extra_count = 0
    with candidate_path.open("rb") as candidate, approved_path.open("rb") as approved:
        approved_line = _next_line(approved)
        for candidate_line in candidate:
            row = json.loads(candidate_line)
            if row["split"] == SEALED_SPLIT:
                sealed_count += 1
                continue
            if approved_line is None:
                omitted_count += 1
                continue
            if candidate_line != approved_line:
                mismatch_count += 1
            approved_count += 1
            approved_line = _next_line(approved)
        while approved_line is not None:
            extra_count += 1
            approved_line = _next_line(approved)
    return {
        "kind": kind, "approved_count": approved_count, "sealed_source_items_excluded": sealed_count,
        "byte_mismatch_count": mismatch_count, "omitted_count": omitted_count, "extra_count": extra_count,
        "status": "PASS" if mismatch_count == omitted_count == extra_count == 0 else "FAIL",
    }


def audit_final_approved_export(candidate: Path, approved: Path) -> dict[str, Any]:
    transition = _audit_exact_copy(
        candidate / "raw/transitions.jsonl", approved / "raw/transitions.jsonl", kind="transition",
    )
    flight = _audit_exact_copy(candidate / "raw/flights.jsonl", approved / "raw/flights.jsonl", kind="flight")
    sealed_in_export = 0
    duplicate_ids = 0
    duplicate_hashes = 0
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    with (approved / "raw/transitions.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sealed_in_export += row["split"] == SEALED_SPLIT
            duplicate_ids += row["record_id"] in seen_ids
            duplicate_hashes += row["record_hash"] in seen_hashes
            seen_ids.add(row["record_id"])
            seen_hashes.add(row["record_hash"])
    errors = []
    if transition["status"] != "PASS" or flight["status"] != "PASS":
        errors.append("exact_copy_mismatch")
    if transition["approved_count"] != 262080 or flight["approved_count"] != 1008:
        errors.append("approved_count_mismatch")
    if transition["sealed_source_items_excluded"] != 37440 or flight["sealed_source_items_excluded"] != 144:
        errors.append("sealed_exclusion_count_mismatch")
    if sealed_in_export:
        errors.append("sealed_record_in_approved_export")
    if duplicate_ids or duplicate_hashes:
        errors.append("duplicate_approved_transition")
    audit: dict[str, Any] = {
        "audit_version": AUDITOR_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "candidate_dataset_id": CANDIDATE_ID,
        "approved_dataset_id": APPROVED_ID,
        "candidate_logical_hash": CANDIDATE_LOGICAL_HASH,
        "transition_exact_copy": transition,
        "flight_exact_copy": flight,
        "sealed_records_in_approved_training_export": sealed_in_export,
        "duplicate_record_ids": duplicate_ids,
        "duplicate_record_hashes": duplicate_hashes,
        "complete_record_equality_scope": [
            "record_ids", "flight_ids", "features", "actions", "states", "failures",
            "sensor_metadata", "actuator_metadata", "scenario_and_tier_metadata", "seeds", "hashes",
        ],
        "candidate_training_eligible_pass_records_omitted": transition["omitted_count"],
        "errors": errors,
    }
    audit["audit_hash"] = object_hash(audit)
    validation = approved / "validation"
    write_json(validation / "approved_final_export_integrity_audit.json", audit)
    (validation / "approved_final_export_integrity_audit.md").write_text(
        "# Approved Final Export Integrity Audit\n\n"
        f"- Status: {audit['status']}\n"
        f"- Flights: {flight['approved_count']} exact copies; {flight['sealed_source_items_excluded']} sealed excluded\n"
        f"- Transitions: {transition['approved_count']} exact copies; {transition['sealed_source_items_excluded']} sealed excluded\n"
        f"- Byte mismatches: {transition['byte_mismatch_count'] + flight['byte_mismatch_count']}\n"
        f"- Sealed records in export: {sealed_in_export}\n"
        f"- Duplicates: IDs={duplicate_ids}, hashes={duplicate_hashes}\n"
        f"- Errors: {errors or 'none'}\n",
        encoding="utf-8",
    )
    if errors:
        raise ValueError(f"approved export integrity audit failed: {errors}")
    return audit


def _stream_train_statistics(approved: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_sum = np.zeros(42, dtype=np.float64)
    feature_sumsq = np.zeros(42, dtype=np.float64)
    feature_min = np.full(42, np.inf)
    feature_max = np.full(42, -np.inf)
    feature_nonfinite = np.zeros(42, dtype=np.int64)
    action_sum = np.zeros(5, dtype=np.float64)
    action_sumsq = np.zeros(5, dtype=np.float64)
    action_min = np.full(5, np.inf)
    action_max = np.full(5, -np.inf)
    action_nonfinite = np.zeros(5, dtype=np.int64)
    saturation = np.zeros(5, dtype=np.int64)
    count = 0
    flights: set[str] = set()
    feature_vectors: set[bytes] = set()
    duplicate_vectors = 0
    source_ids = hashlib.sha256()
    for row in iter_approved_records(approved, split="train", role="normalization"):
        features = np.asarray(row["ordered_observation_42"], dtype=np.float64)
        actions = np.asarray(row["expert_proposed_controls"], dtype=np.float64)
        if features.shape != (42,) or tuple(row["observation_feature_names"]) != tuple(OBSERVATION_FEATURES):
            raise ValueError("42-feature contract mismatch during normalization")
        if actions.shape != (5,):
            raise ValueError("five-output action contract mismatch during normalization")
        feature_nonfinite += ~np.isfinite(features)
        action_nonfinite += ~np.isfinite(actions)
        if not np.isfinite(features).all() or not np.isfinite(actions).all():
            raise ValueError("non-finite train data cannot be normalized")
        feature_sum += features
        feature_sumsq += features * features
        feature_min = np.minimum(feature_min, features)
        feature_max = np.maximum(feature_max, features)
        action_sum += actions
        action_sumsq += actions * actions
        action_min = np.minimum(action_min, actions)
        action_max = np.maximum(action_max, actions)
        saturation += np.asarray([
            abs(actions[0]) >= 0.999, abs(actions[1]) >= 0.999, abs(actions[2]) >= 0.999,
            actions[3] <= 0.001 or actions[3] >= 0.999,
            actions[4] <= 0.001 or actions[4] >= 0.999,
        ])
        signature = features.tobytes()
        duplicate_vectors += signature in feature_vectors
        feature_vectors.add(signature)
        source_ids.update(row["record_id"].encode("utf-8") + b"\n")
        flights.add(row["flight_id"])
        count += 1
    if count != 149760 or len(flights) != 576:
        raise ValueError(f"train-only normalization count mismatch: {len(flights)} flights, {count} transitions")
    feature_mean = feature_sum / count
    feature_std = np.sqrt(np.maximum(feature_sumsq / count - feature_mean * feature_mean, 0.0))
    action_mean = action_sum / count
    action_std = np.sqrt(np.maximum(action_sumsq / count - action_mean * action_mean, 0.0))
    normalized = np.asarray([name not in BINARY_OR_MASK_FEATURES for name in OBSERVATION_FEATURES])
    normalization_mean = feature_mean.copy()
    normalization_std = feature_std.copy()
    normalization_mean[~normalized] = 0.0
    normalization_std[~normalized] = 1.0
    normalization_std[(normalization_std < 1e-6) & normalized] = 1.0
    statistics: dict[str, Any] = {
        "artifact_version": "final-train-normalization-v1",
        "approved_dataset_id": APPROVED_ID,
        "fit_split": "train",
        "fitted_transition_count": count,
        "fitted_flight_count": len(flights),
        "source_record_id_order_hash": source_ids.hexdigest(),
        "feature_names": list(OBSERVATION_FEATURES),
        "means": normalization_mean.tolist(),
        "standard_deviations": normalization_std.tolist(),
        "normalized": normalized.tolist(),
        "clip_standard_deviations": 5.0,
        "observation_vector_version": "phase1-42-v1",
        "excluded_splits": ["development", "challenge", "ood_stress", SEALED_SPLIT],
        "excluded_split_record_counts_used": {name: 0 for name in ("development", "challenge", "ood_stress", SEALED_SPLIT)},
    }
    statistics["normalization_artifact_hash"] = object_hash(statistics)
    features = [
        {"name": name, "min": float(feature_min[i]), "max": float(feature_max[i]), "mean": float(feature_mean[i]),
         "standard_deviation": float(feature_std[i]), "nonfinite_count": int(feature_nonfinite[i])}
        for i, name in enumerate(OBSERVATION_FEATURES)
    ]
    actions = [
        {"name": name, "min": float(action_min[i]), "max": float(action_max[i]), "mean": float(action_mean[i]),
         "standard_deviation": float(action_std[i]), "nonfinite_count": int(action_nonfinite[i]),
         "saturation_count": int(saturation[i]), "saturation_frequency": float(saturation[i] / count)}
        for i, name in enumerate(CONTROL_NAMES)
    ]
    sanity = {
        "audit_version": "final-train-input-sanity-v1",
        "status": "PASS",
        "scope": "approved train split only",
        "training_flight_count": len(flights),
        "training_transition_count": count,
        "features": features,
        "targets": actions,
        "total_feature_nonfinite_count": int(feature_nonfinite.sum()),
        "total_target_nonfinite_count": int(action_nonfinite.sum()),
        "constant_features": [row["name"] for row in features if row["standard_deviation"] < 1e-12],
        "near_constant_features": [row["name"] for row in features if row["standard_deviation"] < 1e-6],
        "highly_imbalanced_action_outputs": [row["name"] for row in actions if row["standard_deviation"] < 0.02],
        "duplicate_feature_vector_count": duplicate_vectors,
        "duplicate_feature_vector_fraction": duplicate_vectors / count,
        "rare_or_unusual_valid_records_removed": False,
    }
    sanity["audit_hash"] = object_hash(sanity)
    return statistics, sanity


def _write_final_manifest(root: Path, approved: Path, approval: dict[str, Any], export: dict[str, Any], audit: dict[str, Any], normalization: dict[str, Any], sanity: dict[str, Any]) -> dict[str, Any]:
    candidate = root / "datasets/candidate_policy_data" / CANDIDATE_ID
    candidate_manifest = _json(candidate / "manifests/candidate_dataset_manifest.json")
    generation = _json(candidate / "manifests/generation_manifest.json")
    schema = _json(candidate / "manifests/schema.json")
    artifact_hashes = {
        path.relative_to(approved).as_posix(): file_hash(path)
        for path in sorted(approved.rglob("*")) if path.is_file() and path.name not in {"approved_manifest.json", "file_checksums.json"}
    }
    logical_hash = object_hash({
        "approved_dataset_id": APPROVED_ID,
        "candidate_logical_hash": CANDIDATE_LOGICAL_HASH,
        "record_hashes": export["transitions"]["record_hashes"],
        "flight_hashes": export["flights"]["item_hashes"],
        "raw_file_hashes": {key: value for key, value in artifact_hashes.items() if key.startswith("raw/")},
        "normalization_hash": normalization["normalization_artifact_hash"],
    })
    manifest: dict[str, Any] = {
        "manifest_version": "final-approved-dataset-manifest-v1",
        "approved_dataset_id": APPROVED_ID,
        "candidate_source_dataset_id": CANDIDATE_ID,
        "candidate_logical_hash": CANDIDATE_LOGICAL_HASH,
        "active_freeze_id": FREEZE_ID,
        "active_freeze_hash": FREEZE_HASH,
        "review_status": approval["review_status"],
        "approval_status": approval["approval_status"],
        "review_method": approval["review_method"],
        "approval_record_hash": approval["approval_record_hash"],
        "validator_version": candidate_manifest["validator_version"],
        "validator_report_hash": candidate_manifest["validator_report_hash"],
        "leakage_audit_hash": candidate_manifest["leakage_audit_hash"],
        "holdout_audit_hash": candidate_manifest["holdout_audit_hash"],
        "sealed_boundary_audit_hash": candidate_manifest["sealed_boundary_audit_hash"],
        "export_integrity_audit_hash": audit["audit_hash"],
        "approved_dataset_logical_hash": logical_hash,
        "approved_flight_count": export["flights"]["included"],
        "approved_transition_count": export["transitions"]["included"],
        "split_counts": export["flights"]["counts"],
        "tier_counts": export["flights"]["tier_counts"],
        "family_counts": export["flights"]["family_counts"],
        "sealed_flights_excluded": export["flights"]["sealed_excluded"],
        "sealed_transitions_excluded": export["transitions"]["sealed_excluded"],
        "sealed_records_in_approved_export": 0,
        "record_hashes": export["transitions"]["record_hashes"],
        "schema_version": schema["schema_version"],
        "feature_order_contract": generation["feature_contract"],
        "action_contract": generation["action_contract"],
        "source_git_revision": generation["git_commit_hash"],
        "normalization_artifact_hash": normalization["normalization_artifact_hash"],
        "training_input_sanity_audit_hash": sanity["audit_hash"],
        "artifact_hashes": artifact_hashes,
        "artifact_hash_exclusions": {
            "manifests/approved_manifest.json": "self-referential manifest",
            "manifests/file_checksums.json": "checksum inventory is written last",
        },
        "created_utc": _utc(),
        "candidate_raw_data_modified": False,
    }
    write_json(approved / "manifests/approved_manifest.json", manifest)
    checksums = {
        path.relative_to(approved).as_posix(): file_hash(path)
        for path in sorted(approved.rglob("*")) if path.is_file() and path.name != "file_checksums.json"
    }
    write_json(approved / "manifests/file_checksums.json", checksums)
    return manifest


def export_final_approved_dataset(root: Path) -> dict[str, Any]:
    candidate = root / "datasets/candidate_policy_data" / CANDIDATE_ID
    approved = root / "datasets/approved_policy_data" / APPROVED_ID
    if approved.exists():
        raise FileExistsError(f"approved dataset is immutable and already exists: {approved}")
    approval = create_final_approval_record(root)
    before_hashes = {
        "transitions": file_hash(candidate / "raw/transitions.jsonl"),
        "flights": file_hash(candidate / "raw/flights.jsonl"),
    }
    for directory in (approved / "raw", approved / "manifests", approved / "validation", approved / "preprocessing"):
        directory.mkdir(parents=True, exist_ok=True)
    transitions = _copy_eligible_jsonl(
        candidate / "raw/transitions.jsonl", approved / "raw/transitions.jsonl", kind="transition",
    )
    flights = _copy_eligible_jsonl(
        candidate / "raw/flights.jsonl", approved / "raw/flights.jsonl", kind="flight",
    )
    if transitions["counts"] != {"train": 149760, "development": 37440, "challenge": 37440, "ood_stress": 37440}:
        raise ValueError(f"approved transition split counts mismatch: {transitions['counts']}")
    if flights["counts"] != {"train": 576, "development": 144, "challenge": 144, "ood_stress": 144}:
        raise ValueError(f"approved flight split counts mismatch: {flights['counts']}")
    if transitions["sealed_excluded"] != 37440 or flights["sealed_excluded"] != 144:
        raise ValueError("sealed exclusion count mismatch")
    export = {"transitions": transitions, "flights": flights, "exported_utc": _utc()}
    write_json(approved / "manifests/export_provenance.json", {
        "approved_dataset_id": APPROVED_ID,
        "candidate_dataset_id": CANDIDATE_ID,
        "candidate_logical_hash": CANDIDATE_LOGICAL_HASH,
        "exported_utc": export["exported_utc"],
        "copy_method": "binary_exact_line_copy_with_nonsealed_split_allowlist",
        "eligible_splits": list(TRAINING_ELIGIBLE_SPLITS),
        "sealed_split_fatal_guard": True,
        "candidate_raw_hashes_before_export": before_hashes,
    })
    audit = audit_final_approved_export(candidate, approved)
    normalization, sanity = _stream_train_statistics(approved)
    write_json(approved / "preprocessing/train_only_normalization.json", normalization)
    write_json(approved / "validation/training_input_sanity_audit.json", sanity)
    (approved / "validation/training_input_sanity_audit.md").write_text(
        "# Approved Train Input Sanity Audit\n\n"
        f"- Status: {sanity['status']}\n"
        f"- Flights/transitions: {sanity['training_flight_count']} / {sanity['training_transition_count']}\n"
        f"- Feature/target non-finite values: {sanity['total_feature_nonfinite_count']} / {sanity['total_target_nonfinite_count']}\n"
        f"- Constant features: {sanity['constant_features']}\n"
        f"- Near-constant features: {sanity['near_constant_features']}\n"
        f"- Highly imbalanced actions: {sanity['highly_imbalanced_action_outputs']}\n"
        f"- Duplicate feature vectors: {sanity['duplicate_feature_vector_count']}\n",
        encoding="utf-8",
    )
    after_hashes = {
        "transitions": file_hash(candidate / "raw/transitions.jsonl"),
        "flights": file_hash(candidate / "raw/flights.jsonl"),
    }
    if after_hashes != before_hashes:
        raise ValueError("candidate raw data changed during approved export")
    manifest = _write_final_manifest(root, approved, approval, export, audit, normalization, sanity)
    freeze_path = root / "docs/research/final_approved_dataset_freeze.md"
    export_command = (
        '.\\.venv311\\Scripts\\python.exe scripts\\export_final_approved_v2_1.py'
    )
    freeze_path.write_text(
        "# Final Approved Dataset Freeze\n\n"
        f"- Approved dataset ID: `{APPROVED_ID}`\n"
        f"- Approved logical hash: `{manifest['approved_dataset_logical_hash']}`\n"
        f"- Candidate logical hash: `{CANDIDATE_LOGICAL_HASH}`\n"
        f"- Generation freeze hash: `{FREEZE_HASH}`\n"
        f"- Approved flights/transitions: {manifest['approved_flight_count']} / {manifest['approved_transition_count']}\n"
        f"- Split counts: `{json.dumps(manifest['split_counts'], sort_keys=True)}`\n"
        f"- Excluded sealed flights/transitions: {manifest['sealed_flights_excluded']} / {manifest['sealed_transitions_excluded']}\n"
        f"- Integrity-audit hash: `{manifest['export_integrity_audit_hash']}`\n"
        f"- Approved-manifest SHA-256: `{file_hash(approved / 'manifests/approved_manifest.json')}`\n"
        f"- Export command: `{export_command}`\n"
        f"- Source Git revision: `{manifest['source_git_revision']}`\n"
        f"- Freeze timestamp: `{_utc()}`\n\n"
        "The final sealed test is excluded from training, normalization, and model selection.\n",
        encoding="utf-8",
    )
    return {
        "approval": approval, "manifest": manifest, "audit": audit,
        "normalization": normalization, "sanity": sanity,
        "candidate_raw_hashes": after_hashes, "approved_path": str(approved),
        "freeze_path": str(freeze_path),
    }
