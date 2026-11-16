from aircraft_recovery.research.pilot_difficulty import _recovery_time, aggregate


def test_recovery_requires_continuous_hold_after_failure() -> None:
    samples = [(index * 0.1, index >= 20) for index in range(80)]
    assert _recovery_time(samples, reference=1.0, dt=0.1) == 1.0


def test_recovery_does_not_bridge_an_unsafe_sample() -> None:
    samples = [(index * 0.1, index >= 10 and index != 40) for index in range(80)]
    assert _recovery_time(samples, reference=1.0, dt=0.1) is None


def test_aggregate_uses_flights_as_experimental_units() -> None:
    rows = [
        {"controller": "fixed", "difficulty_tier": 0, "strict_recovery_success": True,
         "strict_recovery_time_seconds": 2.0, "safety_violation_count": 0,
         "safety_violation_duration_seconds": 0.0, "command_saturation_frequency": 0.0,
         "control_oscillation_mean_absolute_delta": 0.0, "runtime_fallback_count": 0},
        {"controller": "fixed", "difficulty_tier": 1, "strict_recovery_success": False,
         "strict_recovery_time_seconds": None, "safety_violation_count": 2,
         "safety_violation_duration_seconds": 0.2, "command_saturation_frequency": 1.0,
         "control_oscillation_mean_absolute_delta": 0.2, "runtime_fallback_count": 0},
    ]
    result = aggregate(rows)["fixed"]
    assert result["flights"] == 2
    assert result["recovery_rate"] == 0.5
    assert result["by_tier"]["1"]["recovery_rate"] == 0.0
