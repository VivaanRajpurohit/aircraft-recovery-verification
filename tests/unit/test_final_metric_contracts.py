import pytest

from aircraft_recovery.config import load_yaml
from aircraft_recovery.research.metric_contracts import (
    classify_termination,
    inside_recovery_envelope,
    safety_summary,
    safety_violation_magnitudes,
    safety_violations,
    strict_recovery_result,
)


CONFIG = load_yaml("configs/research/final_metrics_v1.yaml")


def _sample(timestamp: float, **changes: float) -> dict[str, float]:
    row = {
        "timestamp_seconds": timestamp, "stall_margin_kts": 10.0,
        "overspeed_margin_kts": 10.0, "bank_deg": 10.0, "pitch_deg": 7.5,
        "vertical_speed_fpm": 750.0, "terrain_clearance_ft": 0.0,
        "angle_of_attack_deg": 14.0, "load_factor_g": 3.5,
    }
    row.update(changes)
    return row


def test_recovery_boundary_equality_is_inside() -> None:
    assert inside_recovery_envelope(_sample(0.0), CONFIG["recovery"])


def test_five_second_rule_requires_consecutive_samples_and_rearms() -> None:
    samples = [_sample(index * 0.1) for index in range(60)]
    samples[25]["bank_deg"] = 10.01
    result = strict_recovery_result(samples, CONFIG["recovery"], 0.0, 0.1)
    assert result["success"] is False
    samples.extend(_sample(index * 0.1) for index in range(60, 76))
    result = strict_recovery_result(samples, CONFIG["recovery"], 0.0, 0.1)
    assert result["success"] is True
    assert result["achievement_time_seconds"] == 2.6


def test_recovery_before_reference_does_not_count() -> None:
    samples = [_sample(index * 0.1) for index in range(82)]
    samples[30]["bank_deg"] = 10.01
    result = strict_recovery_result(samples, CONFIG["recovery"], 3.0, 0.1)
    assert result["success"] is True
    assert result["achievement_time_seconds"] == pytest.approx(0.1)


def test_recovery_loss_is_reported_without_revoking_achievement() -> None:
    samples = [_sample(index * 0.1) for index in range(55)]
    samples[-1]["pitch_deg"] = 8.0
    result = strict_recovery_result(samples, CONFIG["recovery"], 0.0, 0.1)
    assert result == {"success": True, "achievement_time_seconds": 0.0, "lost_after_achievement": True}


def test_safety_boundary_equality_is_safe_and_simultaneous_count_is_explicit() -> None:
    assert not any(safety_violations(_sample(0.0), CONFIG["safety"]).values())
    unsafe = _sample(0.1, stall_margin_kts=-0.01, bank_deg=60.01)
    summary = safety_summary([_sample(0.0), unsafe], CONFIG["safety"], 0.1)
    assert summary["violation_property_step_count"] == 2
    assert summary["any_violation_step_count"] == 1
    assert summary["violation_duration_seconds"] == 0.1
    magnitudes = safety_violation_magnitudes(unsafe, CONFIG["safety"])
    assert magnitudes["stall_kts"] == pytest.approx(0.01)
    assert magnitudes["bank_deg"] == pytest.approx(0.01)


def test_all_supported_termination_reasons_are_classified() -> None:
    expected = {
        "ground_impact": "recovery_failure", "duration_complete": "evaluate_primary_endpoint",
        "numerical_instability": "invalid_experiment", "monitor_timeout": "unsupported_as_termination",
    }
    for reason, classification in expected.items():
        assert classify_termination(reason, CONFIG["termination"]) == classification
