from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys

import pytest

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data_quality.candidate_generation import (
    CandidateFlightConfig,
    CandidateGenerationConfig,
    generate_candidate_dataset,
)
from aircraft_recovery.data_quality.common import file_hash, load_jsonl, object_hash, record_hash, write_json
from aircraft_recovery.data_quality.approved_audit import (
    audit_approved_export,
    validate_pilot_approval_freeze,
)
from aircraft_recovery.data_quality.export import export_approved_dataset
from aircraft_recovery.data_quality.review import append_review_decision
from aircraft_recovery.data_quality.validation import validate_candidate_dataset, validate_record


def _record(step: int = 0, *, flight_id: str = "flight-a", split: str = "train") -> dict:
    before = {
        "altitude_ft": 8000.0 + step,
        "airspeed_kts": 140.0,
        "pitch_deg": 2.0,
        "roll_deg": -3.0,
        "yaw_deg": 0.0,
        "vertical_speed_fpm": 60.0,
        "heading_deg": 250.0,
        "latitude_deg": 37.0,
        "longitude_deg": -120.0,
    }
    after = {**before, "altitude_ft": before["altitude_ft"] + 1.0}
    observation = [0.0] * 42
    for name, value in before.items():
        observation[OBSERVATION_FEATURES.index(name)] = value
    observation[OBSERVATION_FEATURES.index("wind_direction_deg")] = 300.0
    row = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "record_id": f"{flight_id}:step:{step:06d}",
        "flight_id": flight_id,
        "scenario_id": "scenario-a",
        "random_seed": 7,
        "split": split,
        "timestep_index": step,
        "simulation_timestamp_seconds": step * 0.1,
        "state_after_timestamp_seconds": (step + 1) * 0.1,
        "physics_timestep_seconds": 0.1,
        "aircraft_state_before": before,
        "ordered_observation_42": observation,
        "observation_feature_names": list(OBSERVATION_FEATURES),
        "expert_proposed_controls": [0.1, -0.1, 0.0, 0.5, 0.5],
        "label_semantics": "expert_proposed_action",
        "monitor_status": None,
        "executed_controls": [0.1, -0.1, 0.0, 0.5, 0.5],
        "actuator_state": {
            "elevator_effectiveness": 1.0, "aileron_effectiveness": 1.0,
            "rudder_effectiveness": 1.0, "actuator_delay_ms": 0.0,
            "elevator_stuck": False, "aileron_stuck": False, "rudder_stuck": False,
        },
        "engine_state": {
            "left_engine_thrust_available": 1.0,
            "right_engine_thrust_available": 1.0,
        },
        "active_failures": [],
        "disturbance_values": {
            "wind_speed_kts": 10.0, "wind_direction_deg": 300.0,
            "turbulence_intensity": 0.1, "terrain_clearance_ft": 2000.0,
        },
        "aircraft_state_after": after,
        "termination_status": {"terminal": False, "reason": None},
        "recoverability_label": "recoverable",
        "outcome_label": "successful",
        "source_configuration_path": "configs/scenarios/fixture.yaml",
        "source_configuration_hash": "a" * 64,
        "git_commit_hash": "b" * 40,
        "generator_version": "candidate-data-v1",
        "declared_synthetic_corruption": None,
    }
    row["record_hash"] = record_hash(row)
    return row


def _dataset(tmp_path: Path, records: list[dict] | None = None) -> Path:
    records = records or [_record()]
    root = tmp_path / "candidate"
    for name in ("raw", "manifests", "validation", "review"):
        (root / name).mkdir(parents=True, exist_ok=True)
    flights = []
    for flight_id in dict.fromkeys(row["flight_id"] for row in records):
        first = next(row for row in records if row["flight_id"] == flight_id)
        flights.append({
            "dataset_id": "fixture", "flight_id": flight_id,
            "scenario_id": first["scenario_id"], "random_seed": first["random_seed"],
            "split": first["split"], "transition_count": sum(row["flight_id"] == flight_id for row in records),
            "outcome_label": first["outcome_label"],
        })
    transition_path = root / "raw" / "transitions.jsonl"
    flight_path = root / "raw" / "flights.jsonl"
    transition_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    flight_path.write_text("".join(json.dumps(row) + "\n" for row in flights), encoding="utf-8")
    write_json(root / "manifests" / "schema.json", {"schema_version": 1})
    splits = {name: [] for name in ("train", "validation", "test")}
    for flight in flights:
        splits[flight["split"]].append(flight["flight_id"])
    write_json(root / "manifests" / "generation_manifest.json", {
        "schema_version": 1, "dataset_id": "fixture", "record_count": len(records),
        "flight_count": len(flights), "splits": splits,
    })
    paths = [
        transition_path, flight_path, root / "manifests" / "schema.json",
        root / "manifests" / "generation_manifest.json",
    ]
    write_json(root / "manifests" / "file_checksums.json", {
        path.relative_to(root).as_posix(): file_hash(path) for path in paths
    })
    return root


def _rewrite_records(root: Path, records: list[dict]) -> None:
    path = root / "raw" / "transitions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    checksums = json.loads((root / "manifests" / "file_checksums.json").read_text())
    checksums["raw/transitions.jsonl"] = file_hash(path)
    write_json(root / "manifests" / "file_checksums.json", checksums)


def test_raw_candidate_dataset_cannot_be_overwritten(tmp_path: Path) -> None:
    (tmp_path / "existing").mkdir()
    config = CandidateGenerationConfig(
        dataset_id="existing", output_root=str(tmp_path), aircraft_config="missing",
        fallback_config="missing", flights=[CandidateFlightConfig(scenario="missing", seed=1, split="train")],
    )
    with pytest.raises(FileExistsError):
        generate_candidate_dataset(config)


@pytest.mark.parametrize("mutation,reason", [
    (lambda row: row.pop("scenario_id"), "missing:scenario_id"),
    (lambda row: row["observation_feature_names"].reverse(), "wrong_feature_order"),
    (lambda row: row["ordered_observation_42"].__setitem__(0, float("nan")), "nan_or_infinity"),
    (lambda row: row["ordered_observation_42"].__setitem__(0, float("inf")), "nan_or_infinity"),
    (lambda row: row.__setitem__("expert_proposed_controls", [0.0] * 4), "proposed_action_dimension_not_5"),
])
def test_record_validation_rejects_malformed_values(mutation, reason: str) -> None:
    row = _record()
    mutation(row)
    if "record_hash" in row and reason not in {"nan_or_infinity"}:
        row["record_hash"] = record_hash(row)
    result = validate_record(row)
    assert result["status"] == "FAIL"
    assert any(reason in item for item in result["reasons"])


def test_proposed_and_executed_actions_remain_separate() -> None:
    row = _record()
    row["actuator_state"]["aileron_effectiveness"] = 0.5
    row["executed_controls"][1] = -0.05
    row["record_hash"] = record_hash(row)
    assert row["expert_proposed_controls"] != row["executed_controls"]
    assert validate_record(row)["status"] == "REVIEW_REQUIRED"


def test_failure_timing_inconsistency_fails() -> None:
    row = _record()
    row["active_failures"] = [{"failure_id": "late", "activation_time_seconds": 10.0}]
    row["record_hash"] = record_hash(row)
    assert "active_failure_precedes_activation" in validate_record(row)["reasons"]


def test_nonmonotonic_and_discontinuous_transition_fails() -> None:
    previous = _record(0)
    current = _record(1)
    current["simulation_timestamp_seconds"] = 0.05
    current["aircraft_state_before"]["altitude_ft"] += 10.0
    current["record_hash"] = record_hash(current)
    reasons = validate_record(current, previous)["reasons"]
    assert "nonmonotonic_or_gapped_timestamp" in reasons
    assert "state_after_before_discontinuity" in reasons


def test_duplicate_ids_and_broken_hashes_fail(tmp_path: Path) -> None:
    first = _record(0)
    second = _record(1)
    second["record_id"] = first["record_id"]
    second["record_hash"] = record_hash(second)
    root = _dataset(tmp_path, [first, second])
    report = validate_candidate_dataset(root)
    assert "duplicate_record_ids" in report["integrity_errors"]
    assert report["record_status_counts"]["FAIL"] == 2
    records = [first]
    records[0]["airspeed_kts"] = 999  # Extra mutation also invalidates the stored hash.
    _rewrite_records(root, records)
    report = validate_candidate_dataset(root)
    assert any("record_hash_mismatch" in row["reasons"] for row in report["record_results"])


def test_train_validation_flight_overlap_is_reported(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    manifest_path = root / "manifests" / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["validation"] = ["flight-a"]
    write_json(manifest_path, manifest)
    checksums_path = root / "manifests" / "file_checksums.json"
    checksums = json.loads(checksums_path.read_text())
    checksums["manifests/generation_manifest.json"] = file_hash(manifest_path)
    write_json(checksums_path, checksums)
    assert "flight_overlap:train:validation" in validate_candidate_dataset(root)["leakage_errors"]


def test_manual_decisions_append_and_override_requires_reason(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    validate_candidate_dataset(root)
    record_id = _record()["record_id"]
    append_review_decision(root, record_id, "ACCEPT", "reviewer")
    with pytest.raises(ValueError):
        append_review_decision(root, record_id, "REJECT", "reviewer")
    append_review_decision(root, record_id, "REJECT", "reviewer", "manual_context")
    assert len((root / "review" / "decisions.jsonl").read_text().splitlines()) == 2
    assert len((root / "raw" / "transitions.jsonl").read_text().splitlines()) == 1


def test_export_contains_only_latest_explicitly_accepted_records(tmp_path: Path) -> None:
    rows = [_record(0), _record(0, flight_id="flight-b", split="validation")]
    root = _dataset(tmp_path, rows)
    validate_candidate_dataset(root)
    append_review_decision(root, rows[0]["record_id"], "ACCEPT", "reviewer")
    append_review_decision(root, rows[1]["record_id"], "REJECT", "reviewer", "scope", "preserved")
    manifest = export_approved_dataset(root, "approved", tmp_path / "approved")
    assert manifest["included_record_hashes"] == [rows[0]["record_hash"]]
    assert len((tmp_path / "approved" / "approved" / "raw" / "transitions.jsonl").read_text().splitlines()) == 1
    assert (root / "raw" / "transitions.jsonl").read_text().count("\n") == 2


def test_valid_failed_recovery_is_not_automatically_rejected(tmp_path: Path) -> None:
    row = _record()
    row["outcome_label"] = "unsuccessful"
    row["record_hash"] = record_hash(row)
    report = validate_candidate_dataset(_dataset(tmp_path, [row]))
    assert report["record_status_counts"]["PASS"] == 1
    assert report["quality_summary"]["valid_failed_recovery_auto_rejected"] is False


def test_sealed_policy_test_path_cannot_be_opened(tmp_path: Path) -> None:
    sealed = tmp_path / "datasets" / "policy_smoke_v1"
    sealed.mkdir(parents=True)
    with pytest.raises(PermissionError):
        validate_candidate_dataset(sealed)


def test_validation_is_deterministic(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    first = validate_candidate_dataset(root)
    second = validate_candidate_dataset(root)
    assert first == second
    assert first["validator_report_hash"] == object_hash({
        key: value for key, value in first.items() if key != "validator_report_hash"
    })


def test_review_import_does_not_create_ui_or_import_panda3d() -> None:
    before = set(sys.modules)
    importlib.import_module("aircraft_recovery.data_quality.review")
    added = set(sys.modules) - before
    assert not any(name == "panda3d" or name.startswith("panda3d.") for name in added)


def _audited_export_source(tmp_path: Path, status_override: str | None = None) -> tuple[Path, Path]:
    root = _dataset(tmp_path)
    manifest_path = root / "manifests" / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "git_commit_hash": "b" * 40,
        "feature_contract": {"count": 42, "order": list(OBSERVATION_FEATURES)},
        "action_contract": {"count": 5, "order": ["elevator", "aileron", "rudder", "throttle_left", "throttle_right"]},
        "scenario_inventory": [{"scenario_id": "scenario-a", "seed": 7, "tier": 0}],
        "tier_flight_counts": {"0": 1},
        "family_flight_counts": {"fixture": 1},
    })
    write_json(manifest_path, manifest)
    checksums_path = root / "manifests" / "file_checksums.json"
    checksums = json.loads(checksums_path.read_text())
    checksums["manifests/generation_manifest.json"] = file_hash(manifest_path)
    write_json(checksums_path, checksums)
    report = validate_candidate_dataset(root)
    if status_override:
        report["record_results"][0]["status"] = status_override
        report["record_status_counts"] = {
            "PASS": int(status_override == "PASS"),
            "FAIL": int(status_override == "FAIL"),
            "REVIEW_REQUIRED": int(status_override == "REVIEW_REQUIRED"),
        }
        report["validator_report_hash"] = object_hash({
            key: value for key, value in report.items() if key != "validator_report_hash"
        })
        write_json(root / "validation" / "validation_report.json", report)
    disposition = {
        "version": 1, "validator_version": report["validator_version"],
        "counts": {"APPROVE": 1},
        "dispositions": [{"record_id": "flight-a:step:000000", "disposition": "APPROVE"}],
    }
    disposition["manifest_hash"] = object_hash(disposition)
    disposition_path = tmp_path / "dispositions.json"
    write_json(disposition_path, disposition)
    return root, disposition_path


def test_audited_approved_export_preserves_source_values_and_hashes(tmp_path: Path) -> None:
    source, disposition = _audited_export_source(tmp_path)
    source_hash = file_hash(source / "raw" / "transitions.jsonl")
    manifest = export_approved_dataset(
        source, "approved-v1", tmp_path / "approved", disposition_manifest_path=disposition,
    )
    approved = tmp_path / "approved" / "approved-v1"
    audit = audit_approved_export(source, approved, write_artifacts=False)
    assert audit["status"] == "PASS"
    assert audit["record_id_set_match"] is True
    assert not audit["modified_record_ids"]
    assert file_hash(source / "raw" / "transitions.jsonl") == source_hash
    assert load_jsonl(approved / "raw" / "transitions.jsonl") == load_jsonl(source / "raw" / "transitions.jsonl")
    assert manifest["approved_flight_count"] == 1
    assert manifest["approved_transition_count"] == 1
    assert manifest["feature_contract"]["count"] == 42
    assert manifest["action_contract"]["count"] == 5


@pytest.mark.parametrize("status", ["FAIL", "REVIEW_REQUIRED"])
def test_nonpass_record_cannot_enter_audited_export(tmp_path: Path, status: str) -> None:
    source, disposition = _audited_export_source(tmp_path, status)
    with pytest.raises(ValueError, match="non-PASS|zero FAIL"):
        export_approved_dataset(
            source, "approved-v1", tmp_path / "approved", disposition_manifest_path=disposition,
        )


def test_audited_dataset_versioning_prevents_overwrite(tmp_path: Path) -> None:
    source, disposition = _audited_export_source(tmp_path)
    export_approved_dataset(
        source, "approved-v1", tmp_path / "approved", disposition_manifest_path=disposition,
    )
    with pytest.raises(FileExistsError):
        export_approved_dataset(
            source, "approved-v1", tmp_path / "approved", disposition_manifest_path=disposition,
        )


def test_pilot_approval_freeze_validation(tmp_path: Path) -> None:
    manifest = {
        "approved_dataset_id": "approved-v1", "source_candidate_dataset_id": "candidate-v1",
        "approved_dataset_hash": "a" * 64, "validator_version": "validator-v3",
        "validator_report_hash": "b" * 64, "disposition_audit_hash": "c" * 64,
        "source_git_revision": "d" * 40,
    }
    freeze = tmp_path / "freeze.md"
    freeze.write_text("\n".join(map(str, manifest.values())) + "\nsealed final test was untouched\n")
    validate_pilot_approval_freeze(freeze, manifest)
    freeze.write_text("incomplete", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        validate_pilot_approval_freeze(freeze, manifest)


def test_audited_export_keeps_sealed_path_protection(tmp_path: Path) -> None:
    sealed = tmp_path / "datasets" / "policy_smoke_v1"
    sealed.mkdir(parents=True)
    with pytest.raises(PermissionError):
        audit_approved_export(sealed, tmp_path / "approved", write_artifacts=False)
