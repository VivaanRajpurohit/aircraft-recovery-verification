"""Fair paired three-controller experiment batches and explicit outcomes."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Literal

import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict, Field
import torch

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import (
    DeterministicFallbackController,
    FallbackConfig,
    MonitoredNeuralController,
    NeuralBaselineController,
)
from aircraft_recovery.data.hashing import canonical_hash, file_hash
from aircraft_recovery.experiments.runner import ScenarioRunner
from aircraft_recovery.safety import SafetyConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator


class SuccessDefinitions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum_stall_margin_kts: float
    minimum_overspeed_margin_kts: float
    maximum_absolute_bank_deg: float
    maximum_absolute_pitch_deg: float
    stabilization_dwell_seconds: float = Field(gt=0.0)
    diversion_arrival_distance_nm: float = Field(gt=0.0)
    diversion_heading_tolerance_deg: float = Field(gt=0.0)


class BatchScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    category: str
    distribution: Literal["in_distribution", "out_of_distribution"]


class ExperimentBatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_name: str
    output_directory: str
    checkpoint: str
    safety_config: str
    aircraft_config: str
    fallback_config: str
    device: str = "auto"
    repetitions: int = Field(gt=0)
    seed_stride: int = Field(default=1000, gt=0)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_resamples: int = Field(default=2000, gt=99)
    success_definitions: SuccessDefinitions
    scenarios: list[BatchScenario] = Field(min_length=1)


FAILURE_TERMINATIONS = {
    "ground_impact", "stall_not_recovered", "overspeed_not_recovered",
    "loss_of_control", "terrain_clearance_violation", "invalid_state",
    "numerical_instability", "unrecoverable_condition",
}


def _extract_metrics(run_directory: Path, definitions: SuccessDefinitions) -> dict[str, Any]:
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_directory / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    dt = 0.1
    stable_steps_required = max(1, int(round(definitions.stabilization_dwell_seconds / dt)))
    stable_count = 0
    stabilization_time = None
    actual_violations = 0
    selected_airport = None
    diversion_selection_time = None
    commands: list[list[float]] = []
    monitor_decisions: list[dict[str, Any]] = []
    max_pitch = 0.0
    max_load = float("-inf")
    for row in rows:
        observation = row["next_observation"]
        state = observation["aircraft_state"]
        envelope = observation["flight_envelope"]
        stable = (
            envelope["stall_margin_kts"] >= definitions.minimum_stall_margin_kts
            and envelope["overspeed_margin_kts"] >= definitions.minimum_overspeed_margin_kts
            and abs(state["roll_deg"]) <= definitions.maximum_absolute_bank_deg
            and abs(state["pitch_deg"]) <= definitions.maximum_absolute_pitch_deg
        )
        stable_count = stable_count + 1 if stable else 0
        if stabilization_time is None and stable_count >= stable_steps_required:
            stabilization_time = max(0.0, observation["timestamp_seconds"] - definitions.stabilization_dwell_seconds + dt)
        if not stable:
            actual_violations += 1
        max_pitch = max(max_pitch, abs(state["pitch_deg"]))
        max_load = max(max_load, envelope["load_factor_g"])
        navigation = row["final_action"]["navigation_commands"]
        if navigation["selected_airport"] and selected_airport is None:
            selected_airport = navigation["selected_airport"]
            diversion_selection_time = observation["timestamp_seconds"]
        command = row["final_action"]["control_commands"]
        commands.append([command[name] for name in (
            "elevator", "aileron", "rudder", "throttle_left", "throttle_right"
        )])
        if row.get("monitor_decision"):
            monitor_decisions.append(row["monitor_decision"])
    final_distances = rows[-1].get("ground_truth_after", {}).get("airport_distances_nm", {}) if rows else {}
    final_distance = final_distances.get(selected_airport) if selected_airport else None
    final_heading_error = None
    if selected_airport and rows:
        airports = rows[-1]["next_observation"]["navigation"]["nearest_airports"]
        selected = next((airport for airport in airports if airport["airport_id"] == selected_airport), None)
        if selected:
            heading = rows[-1]["next_observation"]["aircraft_state"]["heading_deg"]
            final_heading_error = abs((heading - selected["bearing_deg"] + 180.0) % 360.0 - 180.0)
    diversion_success = bool(
        selected_airport
        and final_distance is not None
        and final_distance <= definitions.diversion_arrival_distance_nm
        and final_heading_error is not None
        and final_heading_error <= definitions.diversion_heading_tolerance_deg
    )
    stabilization_success = stabilization_time is not None
    recovery_success = stabilization_success and summary["termination_reason"] not in FAILURE_TERMINATIONS
    if summary["termination_reason"] == "duration_complete":
        outcome_classification = (
            "duration_complete_after_stabilization" if stabilization_success
            else "duration_complete_without_stabilization"
        )
    elif selected_airport is None and summary["termination_reason"] == "duration_complete":
        outcome_classification = "diversion_infeasible"
    else:
        outcome_classification = summary["termination_reason"]
    command_array = np.asarray(commands, dtype=np.float64)
    saturation = (
        np.any(np.abs(command_array[:, :3]) >= 0.99, axis=1)
        | np.any((command_array[:, 3:] <= 0.01) | (command_array[:, 3:] >= 0.99), axis=1)
    ) if len(command_array) else np.asarray([], dtype=bool)
    status_counts = {
        status: sum(item["status"] == status for item in monitor_decisions)
        for status in ("accept", "modify", "reject_fallback", "unknown_fallback")
    }
    return {
        "stabilization_success": stabilization_success,
        "recovery_success": recovery_success,
        "diversion_selected": selected_airport is not None,
        "diversion_success": diversion_success,
        "ground_impact": summary["termination_reason"] == "ground_impact",
        "unrecoverable_termination": summary["termination_reason"] == "unrecoverable_condition",
        "time_to_stabilization_seconds": stabilization_time,
        "time_to_diversion_selection_seconds": diversion_selection_time,
        "minimum_stall_margin_kts": summary["minimum_stall_margin_kts"],
        "maximum_angle_of_attack_deg": summary["maximum_angle_of_attack_deg"],
        "maximum_absolute_roll_deg": summary["maximum_absolute_roll_deg"],
        "maximum_absolute_pitch_deg": max_pitch,
        "maximum_load_factor_g": max_load,
        "minimum_terrain_clearance_ft": summary["minimum_terrain_clearance_ft"],
        "safety_violation_count": actual_violations,
        "safety_violation_duration_seconds": actual_violations * dt,
        "neural_action_rejection_count": status_counts["reject_fallback"],
        "neural_action_modification_count": status_counts["modify"],
        "fallback_activation_count": sum(item["fallback_activated"] for item in monitor_decisions),
        "solver_timeout_count": sum(item["timeout"] for item in monitor_decisions),
        "solver_unknown_count": sum(item["unknown"] for item in monitor_decisions),
        "monitor_latency_ms_mean": float(np.mean([item["monitor_duration_ms"] for item in monitor_decisions])) if monitor_decisions else 0.0,
        "solver_latency_ms_mean": float(np.mean([item["solver_duration_ms"] for item in monitor_decisions])) if monitor_decisions else 0.0,
        "controller_latency_ms_mean": summary["controller_latency_ms_mean"],
        "total_loop_latency_ms_mean": summary["controller_latency_ms_mean"],
        "command_smoothness": float(np.abs(np.diff(command_array, axis=0)).mean()) if len(command_array) > 1 else 0.0,
        "command_saturation_frequency": float(np.mean(saturation)) if len(saturation) else 0.0,
        "termination_reason": summary["termination_reason"],
        "outcome_classification": outcome_classification,
        "selected_airport": selected_airport,
        "final_distance_from_selected_airport_nm": final_distance,
        "final_heading_error_to_selected_airport_deg": final_heading_error,
        "step_count": len(rows),
    }


def run_experiment_batch(config: ExperimentBatchConfig) -> dict[str, Any]:
    """Run paired scenarios and write flat records plus a reproducibility manifest."""
    output = Path(config.output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Experiment output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    aircraft = load_config(config.aircraft_config, AircraftConfig)
    fallback_config = load_config(config.fallback_config, FallbackConfig)
    safety_config = load_config(config.safety_config, SafetyConfig)
    process = psutil.Process(os.getpid())
    records: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    batch_started = datetime.now(timezone.utc)
    checkpoint_metadata = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    for scenario_index, entry in enumerate(config.scenarios):
        base = load_config(entry.scenario, ScenarioConfig)
        for repetition in range(config.repetitions):
            seed = base.seed + (scenario_index + 1) * config.seed_stride + repetition
            scenario = ScenarioConfig.model_validate({**base.model_dump(), "seed": seed})
            pair_id = f"{scenario.scenario_id}:{seed}"
            controllers = {
                "expert": DeterministicFallbackController(fallback_config, scenario.airports),
                "neural": NeuralBaselineController(config.checkpoint, config.device),
                "monitored": MonitoredNeuralController(
                    config.checkpoint, config.device, fallback_config,
                    scenario.airports, safety_config,
                ),
            }
            for controller_name, controller in controllers.items():
                run_directory = output / "runs" / f"{scenario.scenario_id}_{seed}_{controller_name}"
                cpu_before = process.cpu_times()
                wall_started = perf_counter()
                result = ScenarioRunner(SimpleAircraftSimulator(aircraft, scenario), controller).run(
                    seed, run_directory, scenario.scenario_id
                )
                wall_seconds = perf_counter() - wall_started
                cpu_after = process.cpu_times()
                cpu_seconds = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
                metrics = _extract_metrics(result.run_directory, config.success_definitions)
                records.append({
                    "pair_id": pair_id,
                    "scenario_id": scenario.scenario_id,
                    "scenario_category": entry.category,
                    "distribution": entry.distribution,
                    "expected_recoverability": scenario.expected_recoverability,
                    "seed": seed,
                    "repetition": repetition,
                    "controller": controller_name,
                    "run_directory": str(result.run_directory),
                    "cpu_seconds": cpu_seconds,
                    "cpu_utilization_percent_process": cpu_seconds / max(wall_seconds, 1e-9) * 100.0,
                    "wall_seconds": wall_seconds,
                    "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    **metrics,
                })
    batch_ended = datetime.now(timezone.utc)
    (output / "batch_records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    if records:
        with (output / "batch_records.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    manifest = {
        "batch_name": config.batch_name,
        "format_version": 1,
        "batch_start_utc": batch_started.isoformat(),
        "batch_end_utc": batch_ended.isoformat(),
        "configuration": config.model_dump(mode="json"),
        "configuration_hash": canonical_hash(config.model_dump(mode="json")),
        "checkpoint_hash": file_hash(config.checkpoint),
        "dataset_identifier": checkpoint_metadata.get("dataset_identifier"),
        "split_manifest_identifier": checkpoint_metadata.get("split_manifest_identifier"),
        "safety_configuration_hash": canonical_hash(safety_config.model_dump(mode="json")),
        "record_count": len(records),
        "paired_scenario_count": len(records) // 3,
        "controllers": ["expert", "neural", "monitored"],
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "torch": torch.__version__,
            "z3_solver": version("z3-solver"),
            "matplotlib": version("matplotlib"),
        },
        "hardware": {
            "cpu_logical_count": psutil.cpu_count(),
            "system_memory_bytes": psutil.virtual_memory().total,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "claim_boundary": "Batch outcomes are simulator evidence, not proof of airworthiness or universal safety.",
    }
    (output / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
