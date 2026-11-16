"""Frozen trajectory-level metric calculations for the final design."""

from __future__ import annotations

from math import ceil
from typing import Any


def inside_recovery_envelope(sample: dict[str, float], recovery: dict[str, Any]) -> bool:
    return (
        sample["stall_margin_kts"] >= float(recovery["minimum_stall_margin_kts"])
        and abs(sample["bank_deg"]) <= float(recovery["maximum_absolute_bank_deg"])
        and float(recovery["minimum_pitch_deg"]) <= sample["pitch_deg"] <= float(recovery["maximum_pitch_deg"])
        and abs(sample["vertical_speed_fpm"]) <= float(recovery["maximum_absolute_vertical_speed_fpm"])
        and sample["terrain_clearance_ft"] >= float(recovery["minimum_terrain_clearance_ft"])
    )


def strict_recovery_result(
    samples: list[dict[str, float]],
    recovery: dict[str, Any],
    reference_seconds: float,
    dt_seconds: float,
) -> dict[str, Any]:
    required = max(1, ceil(float(recovery["continuous_hold_seconds"]) / dt_seconds - 1e-12))
    run = 0
    achievement_index: int | None = None
    for index, sample in enumerate(samples):
        if sample["timestamp_seconds"] < reference_seconds:
            continue
        run = run + 1 if inside_recovery_envelope(sample, recovery) else 0
        if run >= required:
            achievement_index = index
            break
    if achievement_index is None:
        return {"success": False, "achievement_time_seconds": None, "lost_after_achievement": False}
    first_index = achievement_index - required + 1
    achievement_time = max(0.0, samples[first_index]["timestamp_seconds"] - reference_seconds)
    lost = any(not inside_recovery_envelope(sample, recovery) for sample in samples[achievement_index + 1:])
    return {"success": True, "achievement_time_seconds": achievement_time, "lost_after_achievement": lost}


def safety_violations(sample: dict[str, float], safety: dict[str, Any]) -> dict[str, bool]:
    return {
        "stall": sample["stall_margin_kts"] < float(safety["minimum_stall_margin_kts"]),
        "overspeed": sample["overspeed_margin_kts"] < float(safety["minimum_overspeed_margin_kts"]),
        "bank": abs(sample["bank_deg"]) > float(safety["maximum_absolute_bank_deg"]),
        "pitch": not (float(safety["minimum_pitch_deg"]) <= sample["pitch_deg"] <= float(safety["maximum_pitch_deg"])),
        "angle_of_attack": sample["angle_of_attack_deg"] > float(safety["maximum_angle_of_attack_deg"]),
        "load_factor": not (float(safety["minimum_load_factor_g"]) <= sample["load_factor_g"] <= float(safety["maximum_load_factor_g"])),
        "terrain_clearance": sample["terrain_clearance_ft"] < float(safety["minimum_terrain_clearance_ft"]),
    }


def safety_violation_magnitudes(sample: dict[str, float], safety: dict[str, Any]) -> dict[str, float]:
    """Return nonnegative raw threshold exceedance in each property's native unit."""
    return {
        "stall_kts": max(0.0, float(safety["minimum_stall_margin_kts"]) - sample["stall_margin_kts"]),
        "overspeed_kts": max(0.0, float(safety["minimum_overspeed_margin_kts"]) - sample["overspeed_margin_kts"]),
        "bank_deg": max(0.0, abs(sample["bank_deg"]) - float(safety["maximum_absolute_bank_deg"])),
        "pitch_deg": max(float(safety["minimum_pitch_deg"]) - sample["pitch_deg"], sample["pitch_deg"] - float(safety["maximum_pitch_deg"]), 0.0),
        "angle_of_attack_deg": max(0.0, sample["angle_of_attack_deg"] - float(safety["maximum_angle_of_attack_deg"])),
        "load_factor_g": max(float(safety["minimum_load_factor_g"]) - sample["load_factor_g"], sample["load_factor_g"] - float(safety["maximum_load_factor_g"]), 0.0),
        "terrain_clearance_ft": max(0.0, float(safety["minimum_terrain_clearance_ft"]) - sample["terrain_clearance_ft"]),
    }


def safety_summary(samples: list[dict[str, float]], safety: dict[str, Any], dt_seconds: float) -> dict[str, Any]:
    rows = [safety_violations(sample, safety) for sample in samples]
    property_steps = sum(sum(flags.values()) for flags in rows)
    any_steps = sum(any(flags.values()) for flags in rows)
    by_property = {name: sum(flags[name] for flags in rows) for name in rows[0]} if rows else {}
    magnitudes = [safety_violation_magnitudes(sample, safety) for sample in samples]
    maximum_exceedance = {
        name: max((row[name] for row in magnitudes), default=0.0)
        for name in (magnitudes[0] if magnitudes else {})
    }
    return {
        "violation_property_step_count": property_steps,
        "any_violation_step_count": any_steps,
        "violation_duration_seconds": any_steps * dt_seconds,
        "violations_by_property": by_property,
        "maximum_raw_exceedance_by_property": maximum_exceedance,
        "flight_has_violation": any_steps > 0,
    }


def classify_termination(reason: str, termination: dict[str, list[str]]) -> str:
    for classification in ("recovery_failure", "evaluate_primary_endpoint", "invalid_experiment", "unsupported_as_termination"):
        if reason in termination[classification]:
            return classification
    raise ValueError(f"Unclassified termination reason: {reason}")
