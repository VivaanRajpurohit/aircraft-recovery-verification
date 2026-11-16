import json
from pathlib import Path

import pytest

from aircraft_recovery.config import load_config, load_yaml
from aircraft_recovery.controllers.fallback import FallbackConfig
from aircraft_recovery.research.pre_generation import (
    build_generation_manifest,
    classify_difficulty_tier,
    expand_seed_registry,
    validate_tier_contract,
)
from aircraft_recovery.safety.config import SafetyConfig
from aircraft_recovery.data_quality.common import object_hash


def test_seed_registry_is_complete_disjoint_and_sealed() -> None:
    registry = json.loads(Path("configs/research/final_seed_registry_v1.json").read_text())
    expanded = expand_seed_registry(registry)
    assert {key: len(value) for key, value in expanded.items()} == {
        "train": 576, "development": 144, "challenge": 144,
        "ood_stress": 144, "final_sealed_test": 144,
    }
    assert registry["final_sealed_test_execution_authorized"] is False
    declared = registry.pop("logical_hash_excluding_this_field")
    assert object_hash(registry) == declared


def test_seed_registry_rejects_overlap() -> None:
    registry = json.loads(Path("configs/research/final_seed_registry_v1.json").read_text())
    registry["dataset_generation"]["development"] = {"first": 20575, "last": 20718, "count": 144}
    with pytest.raises(ValueError, match="overlap"):
        expand_seed_registry(registry)


def test_all_six_tiers_are_numeric_complete_and_ordered() -> None:
    contract = load_yaml("configs/research/final_tiers_v1.yaml")
    assert validate_tier_contract(contract) == []
    assert contract["tiers"][1]["surface_failure_severity"][1] < contract["tiers"][2]["surface_failure_severity"][0]
    assert contract["tiers"][4]["component_severity"][1] <= contract["tiers"][5]["component_severity"][1]
    probes = [(0, 0.0, False), (1, 0.30, False), (1, 0.50, False),
              (2, 0.50, False), (3, 0.75, False), (2, 0.82, True)]
    assert [classify_difficulty_tier(count, severity, boundary_case=boundary)
            for count, severity, boundary in probes] == list(range(6))


def test_heldout_factor_values_are_isolated_and_sealed() -> None:
    matrix = load_yaml("configs/research/final_heldout_factor_matrix_v1.yaml")
    names = ("challenge", "ood_stress", "final_sealed_test")
    severity = [set(matrix["splits"][name]["severity_values"]) for name in names]
    activation = [set(matrix["splits"][name]["activation_seconds"]) for name in names]
    assert all(not severity[i] & severity[j] for i in range(3) for j in range(i + 1, 3))
    assert all(not activation[i] & activation[j] for i in range(3) for j in range(i + 1, 3))
    assert matrix["splits"]["final_sealed_test"]["execution_authorized"] is False


def test_frozen_monitor_and_fallback_configs_validate() -> None:
    monitor = load_config("configs/research/final_monitor_v1.yaml", SafetyConfig)
    fallback = load_config("configs/research/final_fallback_v1.yaml", FallbackConfig)
    assert monitor.thresholds.maximum_absolute_bank_deg == 60.0
    assert monitor.dynamics.horizon_steps == 1
    assert fallback.minimum_mode_duration_seconds == 2.0


def test_generation_manifest_is_guarded_and_hashes_all_contracts() -> None:
    manifest = build_generation_manifest(Path.cwd())
    assert manifest["expected_flights"] == 1152
    assert manifest["generation_authorized"] is False
    assert manifest["training_authorized"] is False
    assert manifest["final_sealed_test_authorized"] is False
    assert manifest["sealed_test_opened"] is False
    assert "configs/research/final_heldout_factor_matrix_v1.yaml" in manifest["files"]
    assert len(manifest["freeze_hash"]) == 64


def test_metric_contract_hash_is_stable() -> None:
    assert manifest_hash("configs/research/final_metrics_v1.yaml") == (
        "6e2632629e9363a8732fb6c826f011de95fbc1ea5d62e7040dd4f35674f32254"
    )


def manifest_hash(path: str) -> str:
    from aircraft_recovery.data_quality.common import file_hash
    return file_hash(path)
