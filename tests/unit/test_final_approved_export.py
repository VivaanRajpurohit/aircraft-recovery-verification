import json
from pathlib import Path

import pytest

from aircraft_recovery.data_quality.final_approved import (
    APPROVED_ID,
    CANDIDATE_LOGICAL_HASH,
    SEALED_SPLIT,
    _audit_exact_copy,
    _copy_eligible_jsonl,
    assert_role_split,
    validate_approved_freeze,
    validate_approved_manifest_contract,
    validate_normalization_contract,
    verify_candidate_source_immutable,
)
from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data_quality.common import CONTROL_NAMES, object_hash


def _row(split: str, index: int) -> dict:
    return {
        "record_id": f"record-{index}", "flight_id": f"flight-{index}",
        "record_hash": f"hash-{index}", "flight_hash": f"flight-hash-{index}",
        "split": split, "difficulty_tier": 0, "scenario_family": "nominal",
    }


def test_exact_copy_export_excludes_sealed_without_renumbering(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    approved = tmp_path / "approved.jsonl"
    rows = [_row("train", 1), _row(SEALED_SPLIT, 2), _row("development", 3)]
    source.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    result = _copy_eligible_jsonl(source, approved, kind="transition")
    exported = [json.loads(line) for line in approved.read_text(encoding="utf-8").splitlines()]
    assert [row["record_id"] for row in exported] == ["record-1", "record-3"]
    assert result["sealed_excluded"] == 1
    assert _audit_exact_copy(source, approved, kind="transition")["status"] == "PASS"


def test_training_and_normalization_roles_are_train_only() -> None:
    assert_role_split("training", "train")
    assert_role_split("normalization", "train")
    with pytest.raises(PermissionError):
        assert_role_split("training", "development")
    with pytest.raises(PermissionError):
        assert_role_split("normalization", "challenge")


@pytest.mark.parametrize("role", ["normalization", "training", "model_selection", "challenge_evaluation", "ood_evaluation"])
def test_every_loader_role_fatally_rejects_sealed(role: str) -> None:
    with pytest.raises(PermissionError, match="final_sealed_test"):
        assert_role_split(role, SEALED_SPLIT)


def test_approved_manifest_requires_exact_feature_action_and_split_contracts() -> None:
    manifest = {
        "approved_dataset_id": APPROVED_ID, "candidate_logical_hash": CANDIDATE_LOGICAL_HASH,
        "approved_flight_count": 1008, "approved_transition_count": 262080,
        "split_counts": {"train": 576, "development": 144, "challenge": 144, "ood_stress": 144},
        "sealed_records_in_approved_export": 0,
        "feature_order_contract": {"order": list(OBSERVATION_FEATURES)},
        "action_contract": {"order": list(CONTROL_NAMES)},
    }
    validate_approved_manifest_contract(manifest)
    manifest["action_contract"]["order"] = ["wrong"] * 5
    with pytest.raises(ValueError, match="action_contract"):
        validate_approved_manifest_contract(manifest)


def test_normalization_contract_is_train_only_and_self_hashed() -> None:
    statistics = {
        "fit_split": "train", "fitted_transition_count": 149760, "fitted_flight_count": 576,
        "feature_names": list(OBSERVATION_FEATURES),
        "excluded_splits": ["development", "challenge", "ood_stress", SEALED_SPLIT],
        "excluded_split_record_counts_used": {
            "development": 0, "challenge": 0, "ood_stress": 0, SEALED_SPLIT: 0,
        },
    }
    statistics["normalization_artifact_hash"] = object_hash(statistics)
    validate_normalization_contract(statistics)
    statistics["excluded_split_record_counts_used"]["development"] = 1
    with pytest.raises(ValueError, match="excluded_split_usage"):
        validate_normalization_contract(statistics)


def test_approved_freeze_requires_immutable_identities(tmp_path: Path) -> None:
    manifest = {
        "approved_dataset_id": APPROVED_ID, "approved_dataset_logical_hash": "approved-hash",
        "candidate_logical_hash": CANDIDATE_LOGICAL_HASH, "active_freeze_hash": "freeze-hash",
        "export_integrity_audit_hash": "audit-hash", "source_git_revision": "git-hash",
    }
    path = tmp_path / "freeze.md"
    path.write_text("\n".join(str(value) for value in manifest.values()) +
                    "\n144 / 37440\nexcluded from training, normalization, and model selection\n", encoding="utf-8")
    validate_approved_freeze(path, manifest)


def test_candidate_source_immutability_checks_pre_and_post_export_hashes(tmp_path: Path) -> None:
    candidate = tmp_path / "datasets/candidate_policy_data/final_policy_candidate_v2_1"
    approved = tmp_path / "approved"
    (candidate / "raw").mkdir(parents=True)
    (candidate / "manifests").mkdir()
    (approved / "manifests").mkdir(parents=True)
    (candidate / "raw/transitions.jsonl").write_text("transition\n", encoding="utf-8")
    (candidate / "raw/flights.jsonl").write_text("flight\n", encoding="utf-8")
    from aircraft_recovery.data_quality.common import file_hash
    hashes = {
        "transitions": file_hash(candidate / "raw/transitions.jsonl"),
        "flights": file_hash(candidate / "raw/flights.jsonl"),
    }
    (candidate / "manifests/candidate_dataset_manifest.json").write_text(json.dumps({
        "artifact_hashes": {
            "raw/transitions.jsonl": hashes["transitions"], "raw/flights.jsonl": hashes["flights"],
        }
    }), encoding="utf-8")
    (approved / "manifests/export_provenance.json").write_text(json.dumps({
        "candidate_raw_hashes_before_export": hashes,
    }), encoding="utf-8")
    verify_candidate_source_immutable(tmp_path, approved)
    (candidate / "raw/flights.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate source changed"):
        verify_candidate_source_immutable(tmp_path, approved)
