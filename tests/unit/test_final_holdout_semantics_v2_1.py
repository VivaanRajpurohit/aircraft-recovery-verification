import json
from pathlib import Path

from aircraft_recovery.data_quality.common import object_hash
from aircraft_recovery.research.design_amendment import (
    audit_v2_1_amendment,
    build_v2_1_freeze_manifest,
)
from aircraft_recovery.research.holdout_audit import build_exposure_matrix


def test_original_v2_fails_strict_holdout_semantics() -> None:
    result = build_exposure_matrix(Path.cwd(), use_v2_1=False)
    assert result["status"] == "FAIL"
    assert 2.5 in result["strict_sealed_collisions"]["failure_activation_time"]
    assert result["strict_sealed_collisions"]["component_severity"] == [0.34, 0.46, 0.48]


def test_v2_1_strict_and_model_selection_holdouts_pass() -> None:
    result = build_exposure_matrix(Path.cwd(), use_v2_1=True)
    assert result["status"] == "PASS"
    assert all(not values for values in result["strict_sealed_collisions"].values())
    assert all(not values for values in result["model_selection_collisions"].values())
    assert all(not values for values in result["challenge_tier_model_selection_collisions"].values())
    assert all(not values for values in result["ood_tier_model_selection_collisions"].values())


def test_sealed_combination_and_seed_holdouts_pass() -> None:
    result = build_exposure_matrix(Path.cwd(), use_v2_1=True)
    assert result["seed_collisions"] == {}
    assert result["sealed_signature_duplicate_counts"] == {
        "train": 0, "development": 0, "challenge": 0, "ood_stress": 0,
    }
    assert len(result["scenario_families_seen_in_all_splits"]) == 20


def test_v2_1_static_audit_preserves_counts_and_is_guarded() -> None:
    result = audit_v2_1_amendment(Path.cwd())
    assert result["status"] == "PASS"
    assert result["total_flights"] == 1152
    assert result["generation_authorized"] is False
    assert result["sealed_outcomes_accessed"] is False


def test_v2_1_freeze_is_self_consistent_and_previous_v2_is_preserved() -> None:
    manifest = build_v2_1_freeze_manifest(Path.cwd())
    declared = manifest.pop("freeze_hash")
    assert object_hash(manifest) == declared
    previous = json.loads(Path("artifacts/research_stage/final_generation_manifest_v2.json").read_text(encoding="utf-8"))
    assert previous["freeze_hash"] == "92d5a0137c7e185ce49060d0d1ada6ab631410a2914ba465d377aa43b83004bc"
    assert manifest["superseded_v2_proposal_hash"] == previous["freeze_hash"]
