"""Run the bounded command-rate response ablation without changing defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import FallbackConfig, MonitoredNeuralController
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.safety import SafetyConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator


CONTROL_NAMES = ("elevator", "aileron", "rudder", "throttle_left", "throttle_right")


class AblationVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    maximum_control_rate_per_second: float = Field(gt=0.0)


class AblationScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    seed: int


class ResponseAblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ablation_name: str
    output_directory: str
    checkpoint: str
    aircraft_config: str
    fallback_config: str
    safety_config: str
    device: str = "cpu"
    recovery_bank_bound_deg: float = Field(gt=0.0)
    recovery_pitch_bound_deg: float = Field(gt=0.0)
    acceptable_vertical_speed_fpm: float = Field(gt=0.0)
    stabilization_dwell_seconds: float = Field(gt=0.0)
    variants: list[AblationVariant] = Field(min_length=2)
    scenarios: list[AblationScenario] = Field(min_length=1)


def _first_sustained_time(
    rows: list[dict[str, Any]], start_index: int, dwell_steps: int, predicate: Any,
) -> float | None:
    for index in range(start_index, len(rows) - dwell_steps + 1):
        if all(predicate(row) for row in rows[index:index + dwell_steps]):
            return float(rows[index]["observation"]["timestamp_seconds"])
    return None


def _metrics(run_directory: Path, config: ResponseAblationConfig) -> dict[str, Any]:
    rows = [json.loads(line) for line in (run_directory / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    activation_index = next((
        index for index, row in enumerate(rows)
        if row["ground_truth_before"].get("ground_truth_failures")
        or row["observation"].get("active_failures")
    ), None)
    activation_index = 0 if activation_index is None else activation_index
    activation_time = float(rows[activation_index]["observation"]["timestamp_seconds"])
    state = lambda row: row["observation"]["aircraft_state"]
    envelope = lambda row: row["observation"]["flight_envelope"]
    post = rows[activation_index:]

    def first_offset(predicate: Any) -> float | None:
        value = next((float(row["observation"]["timestamp_seconds"]) for row in post if predicate(row)), None)
        return None if value is None else value - activation_time

    envelope_safe = lambda row: (
        envelope(row)["stall_margin_kts"] >= 15.0
        and envelope(row)["overspeed_margin_kts"] >= 15.0
        and abs(state(row)["roll_deg"]) <= config.recovery_bank_bound_deg
        and abs(state(row)["pitch_deg"]) <= config.recovery_pitch_bound_deg
        and row["observation"]["environment"]["terrain_clearance_ft"] >= 0.0
    )
    safe = lambda row: envelope_safe(row) and (
        abs(state(row)["vertical_speed_fpm"]) <= config.acceptable_vertical_speed_fpm
    )
    dt = float(rows[0]["next_observation"]["timestamp_seconds"] - rows[0]["observation"]["timestamp_seconds"])
    dwell_steps = max(1, round(config.stabilization_dwell_seconds / dt))
    recovery_start = _first_sustained_time(rows, activation_index, dwell_steps, safe)
    peak_index = max(
        range(activation_index, len(rows)),
        key=lambda index: abs(state(rows[index])["roll_deg"]),
    )
    strict = lambda row: (
        abs(state(row)["roll_deg"]) <= 10.0
        and abs(state(row)["pitch_deg"]) <= 8.0
        and abs(state(row)["vertical_speed_fpm"]) <= config.acceptable_vertical_speed_fpm
        and envelope(row)["stall_margin_kts"] >= 15.0
        and envelope(row)["overspeed_margin_kts"] >= 15.0
    )
    post_peak_recovery = _first_sustained_time(rows, peak_index, dwell_steps, strict)
    commands = np.asarray([
        [row["final_action"]["control_commands"][name] for name in CONTROL_NAMES]
        for row in rows
    ], dtype=np.float64)
    executed_commands = np.asarray([
        [row["ground_truth_after"]["applied_control_commands"][name] for name in CONTROL_NAMES]
        for row in rows
    ], dtype=np.float64)
    first_proposal_time = float(rows[activation_index]["proposed_action"]["timestamp_seconds"])
    prior_executed = executed_commands[max(0, activation_index - 1)]
    first_executed_index = next((
        index for index in range(activation_index, len(rows))
        if np.max(np.abs(executed_commands[index] - prior_executed)) > 1e-9
    ), activation_index)
    first_executed_time = float(rows[first_executed_index]["observation"]["timestamp_seconds"])
    deltas = np.diff(commands, axis=0)
    reversals = np.sign(deltas[1:]) != np.sign(deltas[:-1]) if len(deltas) > 1 else np.zeros((0, 5), dtype=bool)
    saturation = np.any(np.abs(commands[:, :3]) >= 0.99, axis=1) | np.any(
        (commands[:, 3:] <= 0.01) | (commands[:, 3:] >= 0.99), axis=1
    )
    violations = [not envelope_safe(row) for row in rows]
    monitor_counts = summary["monitor_statistics"]
    return {
        "failure_activation_time_seconds": activation_time,
        "first_neural_proposal_after_failure_seconds": first_proposal_time,
        "neural_proposal_response_delay_seconds": first_proposal_time - activation_time,
        "first_executed_correction_after_failure_seconds": first_executed_time,
        "executed_correction_response_delay_seconds": first_executed_time - activation_time,
        "time_to_recovery_bank_bound_seconds": first_offset(
            lambda row: abs(state(row)["roll_deg"]) <= config.recovery_bank_bound_deg
        ),
        "time_to_recovery_pitch_bound_seconds": first_offset(
            lambda row: abs(state(row)["pitch_deg"]) <= config.recovery_pitch_bound_deg
        ),
        "time_to_acceptable_vertical_speed_seconds": first_offset(
            lambda row: abs(state(row)["vertical_speed_fpm"]) <= config.acceptable_vertical_speed_fpm
        ),
        "full_recovery_time_seconds": None if recovery_start is None else recovery_start + config.stabilization_dwell_seconds - activation_time,
        "post_peak_strict_recovery_time_seconds": None if post_peak_recovery is None else post_peak_recovery + config.stabilization_dwell_seconds - activation_time,
        "maximum_absolute_bank_deg": max(abs(state(row)["roll_deg"]) for row in rows),
        "maximum_absolute_pitch_deg": max(abs(state(row)["pitch_deg"]) for row in rows),
        "minimum_airspeed_kts": min(state(row)["airspeed_kts"] for row in rows),
        "minimum_terrain_clearance_ft": min(row["observation"]["environment"]["terrain_clearance_ft"] for row in rows),
        "control_oscillation_mean_absolute_delta": float(np.mean(np.abs(deltas))) if len(deltas) else 0.0,
        "control_reversal_frequency": float(np.mean(reversals)) if len(reversals) else 0.0,
        "command_saturation_frequency": float(np.mean(saturation)),
        "monitor_modification_count": int(monitor_counts["modify"]),
        "fallback_count": int(monitor_counts["reject_fallback"] + monitor_counts["unknown_fallback"]),
        "safety_envelope_violation_steps": int(sum(violations)),
        "safety_envelope_violation_seconds": float(sum(violations) * dt),
        "controller_inference_and_monitor_latency_ms_mean": summary["controller_latency_ms_mean"],
        "termination_reason": summary["termination_reason"],
    }


def run_ablation(config: ResponseAblationConfig) -> dict[str, Any]:
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    aircraft = load_config(config.aircraft_config, AircraftConfig)
    fallback = load_config(config.fallback_config, FallbackConfig)
    base_safety = load_config(config.safety_config, SafetyConfig)
    results: dict[str, list[dict[str, Any]]] = {}
    for variant in config.variants:
        increase = variant.maximum_control_rate_per_second / config.variants[0].maximum_control_rate_per_second - 1.0
        if increase > 0.2000001:
            raise ValueError(f"{variant.name} exceeds the approved 20% rate increase")
        safety = base_safety.model_copy(deep=True)
        safety.configuration_name = f"{base_safety.configuration_name}_{variant.name}"
        safety.thresholds.maximum_control_rate_per_second = variant.maximum_control_rate_per_second
        variant_results: list[dict[str, Any]] = []
        for scenario_item in config.scenarios:
            base_scenario = load_config(scenario_item.scenario, ScenarioConfig)
            scenario = ScenarioConfig.model_validate({**base_scenario.model_dump(), "seed": scenario_item.seed})
            controller = MonitoredNeuralController(
                config.checkpoint, config.device, fallback, scenario.airports, safety,
            )
            run_directory = output / "runs" / f"{scenario.scenario_id}_{scenario.seed}_{variant.name}"
            if run_directory.exists():
                raise FileExistsError(f"Ablation run already exists: {run_directory}")
            run = ScenarioRunner(SimpleAircraftSimulator(aircraft, scenario), controller).run(
                scenario.seed, run_directory, scenario.scenario_id,
            )
            variant_results.append({
                "scenario_id": scenario.scenario_id,
                "seed": scenario.seed,
                **_metrics(run.run_directory, config),
            })
        results[variant.name] = variant_results

    aggregates: dict[str, dict[str, float]] = {}
    for variant, rows in results.items():
        numeric = (
            "neural_proposal_response_delay_seconds", "executed_correction_response_delay_seconds",
            "time_to_recovery_bank_bound_seconds", "time_to_recovery_pitch_bound_seconds",
            "time_to_acceptable_vertical_speed_seconds", "full_recovery_time_seconds",
            "post_peak_strict_recovery_time_seconds", "minimum_airspeed_kts",
            "maximum_absolute_bank_deg", "maximum_absolute_pitch_deg",
            "minimum_terrain_clearance_ft",
            "control_oscillation_mean_absolute_delta", "control_reversal_frequency",
            "command_saturation_frequency", "monitor_modification_count", "fallback_count",
            "safety_envelope_violation_steps", "controller_inference_and_monitor_latency_ms_mean",
        )
        aggregates[variant] = {
            name: float(np.mean([row[name] for row in rows if row[name] is not None]))
            for name in numeric
        }
    baseline_name = config.variants[0].name
    baseline = aggregates[baseline_name]
    selected = None
    for variant in config.variants[1:]:
        candidate = aggregates[variant.name]
        improved = candidate["full_recovery_time_seconds"] < baseline["full_recovery_time_seconds"] - 1e-9
        no_new_violations = candidate["safety_envelope_violation_steps"] <= baseline["safety_envelope_violation_steps"]
        oscillation_ok = candidate["control_oscillation_mean_absolute_delta"] <= baseline["control_oscillation_mean_absolute_delta"] * 1.10
        if improved and no_new_violations and oscillation_ok:
            selected = variant.name
            break
    report = {
        "ablation_name": config.ablation_name,
        "default_configuration_unchanged": True,
        "per_step_maximum_deltas": {
            item.name: item.maximum_control_rate_per_second * base_safety.dynamics.time_step_seconds
            for item in config.variants
        },
        "results": results,
        "aggregates": aggregates,
        "selected_candidate": selected,
        "selection_rule": "Faster full recovery, no additional safety violations, and <=10% worse mean command oscillation.",
    }
    (output / "response_ablation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/response_speed_ablation.yaml")
    args = parser.parse_args()
    config = load_config(args.config, ResponseAblationConfig)
    print(json.dumps(run_ablation(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
