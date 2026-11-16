from pathlib import Path

from aircraft_recovery.research.dataset_design import audit_dataset_design


def test_final_dataset_design_arithmetic_and_guards_pass() -> None:
    result = audit_dataset_design("configs/training/final_policy_dataset_design_v1.yaml")
    assert result["status"] == "PASS"
    assert result["total_flights"] == 1152
    assert result["split_totals"] == {
        "train": 576, "development": 144, "challenge": 144,
        "ood_stress": 144, "final_sealed_test": 144,
    }
    assert result["tier_totals"] == {str(tier): 192 for tier in range(6)}
    assert result["seed_overlap"] is False
    assert result["engine_side_balanced"] is True
    assert set(result["severity_band_counts"].values()) == {192}
    assert result["large_dataset_generated"] is False
    assert result["sealed_test_opened"] is False


def test_design_audit_detects_enabled_generation_guard(tmp_path: Path) -> None:
    source = Path("configs/training/final_policy_dataset_design_v1.yaml").read_text(encoding="utf-8")
    changed = source.replace("generate_large_dataset: false", "generate_large_dataset: true")
    path = tmp_path / "invalid.yaml"
    path.write_text(changed, encoding="utf-8")
    result = audit_dataset_design(path)
    assert result["status"] == "FAIL"
    assert "design_only_guard_enabled" in result["errors"]
