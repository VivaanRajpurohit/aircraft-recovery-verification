import copy
import json
from pathlib import Path

from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.candidate_generation import CandidateFlightConfig
from aircraft_recovery.data_quality.common import object_hash
from aircraft_recovery.research.design_amendment import (
    _check_schedule,
    audit_v2_amendment,
    build_v2_freeze_manifest,
)


def test_v2_static_amendment_audit_passes() -> None:
    result = audit_v2_amendment(Path.cwd())
    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["total_flights"] == 1152
    assert result["tier_counts"] == {str(tier): 192 for tier in range(6)}
    assert result["scenario_family_count"] == 20


def test_out_of_range_severity_and_activation_are_rejected() -> None:
    errors: list[str] = []
    _check_schedule(errors, 1, "component_severity", {"train": [0.20], "challenge": [0.40]}, (0.20, 0.35))
    _check_schedule(errors, 1, "activation_seconds", {"train": [3.0], "final_sealed_test": [2.75]}, (3.0, 8.0))
    assert any("out_of_tier_range:tier_1:component_severity:challenge" in error for error in errors)
    assert any("out_of_tier_range:tier_1:activation_seconds:final_sealed_test" in error for error in errors)


def test_required_value_holdout_collision_is_rejected() -> None:
    errors: list[str] = []
    _check_schedule(errors, 2, "component_severity", {"train": [0.50], "challenge": [0.50]}, (0.45, 0.60))
    assert any("required_value_holdout_collision" in error for error in errors)


def test_every_scenario_family_has_parameter_semantics() -> None:
    schedule = load_yaml("configs/research/final_heldout_factor_matrix_v2.yaml")
    design = load_yaml("configs/training/final_policy_dataset_design_v1.yaml")
    assert set(schedule["family_parameter_semantics"]) == {row["family"] for row in design["scenario_matrix"]}


def test_final_sealed_split_name_is_supported_with_legacy_aliases() -> None:
    common = {"scenario": "fixture.yaml", "seed": 1}
    assert CandidateFlightConfig(**common, split="final_sealed_test").split == "final_sealed_test"
    assert CandidateFlightConfig(**common, split="test").split == "test"


def test_v2_freeze_is_historical_while_v2_1_is_the_active_freeze() -> None:
    manifest = build_v2_freeze_manifest(Path.cwd())
    declared = manifest.pop("freeze_hash")
    assert object_hash(manifest) == declared
    registry_path = Path("artifacts/research_stage/final_generation_freeze_registry.json")
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        active = registry["active_generation_freeze"]
        assert active["freeze_id"] == "final_generation_freeze_v2_1"
        assert active["freeze_hash"] == "c0b831ede8a5d75027dcce70ec1c116fc5ec58d479cf500939f2075e3d16d0dc"
        assert registry["withdrawn_v2_proposal"]["freeze_hash"] == (
            "92d5a0137c7e185ce49060d0d1ada6ab631410a2914ba465d377aa43b83004bc"
        )
        assert registry["withdrawn_v2_proposal"]["status"].startswith("not_activated")
        assert registry["historical_v1"]["status"].startswith("superseded_before_generation")
