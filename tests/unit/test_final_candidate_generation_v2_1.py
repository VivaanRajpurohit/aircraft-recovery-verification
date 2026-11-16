from collections import Counter
import json
from pathlib import Path

from aircraft_recovery.config import AircraftConfig, load_config, load_yaml
from aircraft_recovery.controllers import FallbackConfig
from aircraft_recovery.data_quality.common import object_hash
from aircraft_recovery.data_quality.final_generation import (
    ACTIVE_FREEZE_HASH,
    _write_batch,
    build_final_flight_registry,
)


def test_active_registry_is_hashed_and_has_exactly_one_active_freeze() -> None:
    registry = json.loads(Path("artifacts/research_stage/final_generation_freeze_registry.json").read_text())
    declared = registry.pop("registry_hash_excluding_this_field")
    assert object_hash(registry) == declared
    assert registry["active_generation_freeze"]["freeze_hash"] == ACTIVE_FREEZE_HASH
    assert registry["proposed_v2_1"]["status"] == "active_approved_for_candidate_generation"
    assert registry["historical_v1"]["status"].startswith("superseded")
    assert registry["withdrawn_v2_proposal"]["status"].startswith("not_activated")


def test_frozen_flight_registry_has_exact_allocations() -> None:
    specs = build_final_flight_registry(Path.cwd())
    assert len(specs) == 1152
    assert Counter(row["split"] for row in specs) == {
        "train": 576, "development": 144, "challenge": 144,
        "ood_stress": 144, "final_sealed_test": 144,
    }
    assert Counter(row["tier"] for row in specs) == {tier: 192 for tier in range(6)}
    assert Counter(row["engine_side"] for row in specs if row["engine_side"]) == {"left": 152, "right": 152}
    assert len({row["seed"] for row in specs}) == 1152
    assert all(not row["gaussian_sensor_noise_enabled"] for row in specs)


def test_realized_registry_strict_sealed_values_are_unseen_earlier() -> None:
    specs = build_final_flight_registry(Path.cwd())
    earlier = [row for row in specs if row["split"] != "final_sealed_test"]
    sealed = [row for row in specs if row["split"] == "final_sealed_test"]
    for field in ("component_severity", "activation_seconds", "crosswind_maximum_kts"):
        before = {row[field] for row in earlier if row[field] is not None}
        after = {row[field] for row in sealed if row[field] is not None}
        assert not before & after
    assert not {row["canonical_signature"] for row in earlier} & {row["canonical_signature"] for row in sealed}


def test_small_mixed_batch_passes_independent_validation(tmp_path: Path) -> None:
    root = Path.cwd()
    specs = build_final_flight_registry(root)
    families = ("nominal", "isolated_aileron_mild", "deterministic_sensor_bias", "isolated_actuator_delay")
    sample = [next(row for row in specs if row["family"] == family and row["split"] == "train") for family in families]
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    fallback = load_config("configs/research/final_fallback_v1.yaml", FallbackConfig)
    metrics = load_yaml("configs/research/final_metrics_v1.yaml")
    manifest = _write_batch(root, tmp_path / "batch", sample, 0, aircraft, fallback, metrics, "d" * 40)
    assert manifest["status"] == "PASS"
    assert manifest["record_status_counts"]["FAIL"] == 0
    assert manifest["record_status_counts"]["REVIEW_REQUIRED"] == 0
