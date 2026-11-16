"""Frozen representative-only v2.1B Challenge and OOD-stress evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch

from aircraft_recovery.common.observations import observation_to_vector
from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config, load_yaml
from aircraft_recovery.controllers import DeterministicFallbackController, FallbackConfig
from aircraft_recovery.data_quality.common import CONTROL_NAMES, file_hash, object_hash
from aircraft_recovery.data_quality.final_approved import (
    APPROVED_ID,
    assert_role_split,
    iter_approved_records,
)
from aircraft_recovery.data_quality.final_generation import _metric_sample, _scenario
from aircraft_recovery.models import (
    ControlCommands,
    ControllerInput,
    ControllerOutput,
    NavigationCommands,
    SafetyDecision,
    StabilizationTargets,
)
from aircraft_recovery.research.metric_contracts import (
    classify_termination,
    safety_summary,
    strict_recovery_result,
)
from aircraft_recovery.safety import RuntimeSafetyMonitor, SafetyConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.training.v2_1a_architecture_screen import (
    APPROVED_DATASET_LOGICAL_HASH,
    NORMALIZATION_LOGICAL_HASH,
    TARGET_ORDER,
    V21AControlPolicyNetwork,
    _normalize,
)
from aircraft_recovery.training.v2_1a_optimizer_selection import WINNING_ARCHITECTURE


REPRESENTATIVE_SEED = 17001
REPRESENTATIVE_CHECKPOINT_HASH = "175064a9d978edff8542e41d626ecad2e0cb04b4552ed055668043389b369ef7"
MODEL_SELECTION_FREEZE_LOGICAL_HASH = "50cd2c0b355d5776ac3fb11064734a49dc5a76f0c6fd802b8d81e71d4bac5efc"
FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH = "288b96bbeed75d4ce19793e28198be25430f8dd260604fb654276c405467917a"
FIVE_SEED_PROVENANCE_LOGICAL_HASH = "b3c06be0a6f519f5860dae4acd18d758a22c2ba3178d5a5d52152a4d1a58c9de"
AUTHORIZED_SPLITS = {
    "challenge": "challenge_evaluation",
    "ood_stress": "ood_evaluation",
}
EXPECTED_FLIGHTS = 144
EXPECTED_TRANSITIONS = 37440
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 19001


class V21AControlPolicyController:
    """The frozen five-control checkpoint as a simulator controller."""

    name = "v2_1b_representative_unmonitored"

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self.device = device
        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = V21AControlPolicyNetwork(WINNING_ARCHITECTURE).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.normalization = checkpoint["normalization"]
        self.model_checkpoint_identifier = file_hash(checkpoint_path)
        self.last_inference_latency_ms = 0.0
        self.last_proposed_action: ControllerOutput | None = None

    def reset(self) -> None:
        self.last_inference_latency_ms = 0.0
        self.last_proposed_action = None

    def act(self, observation: ControllerInput) -> ControllerOutput:
        vector = observation_to_vector(observation).astype(np.float32)
        normalized = _normalize(vector.reshape(1, -1), self.normalization)
        tensor = torch.from_numpy(normalized).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        with torch.inference_mode():
            controls = self.model(tensor)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_inference_latency_ms = (perf_counter() - started) * 1000.0
        values = controls.detach().cpu().tolist()
        state = observation.aircraft_state
        output = ControllerOutput(
            timestamp_seconds=observation.timestamp_seconds,
            control_commands=ControlCommands(**dict(zip(TARGET_ORDER, values, strict=True))),
            stabilization_targets=StabilizationTargets(
                target_pitch_deg=state.pitch_deg,
                target_roll_deg=state.roll_deg,
                target_airspeed_kts=state.airspeed_kts,
                target_altitude_ft=state.altitude_ft,
            ),
            navigation_commands=NavigationCommands(
                target_heading_deg=state.heading_deg,
                selected_airport=None,
                selected_runway_heading_deg=None,
                approach_phase="none",
            ),
            emergency_mode="stabilize" if observation.active_failures else "normal",
            safety_decision=SafetyDecision(
                status="not_checked",
                ai_action_accepted=True,
                fallback_activated=False,
                violated_constraints=[],
                explanation="Frozen v2.1B representative neural proposal without monitor.",
            ),
            confidence=1.0,
        )
        self.last_proposed_action = output
        return output


class V21AMonitoredController:
    """Frozen representative checkpoint with frozen monitor and fallback."""

    name = "v2_1b_representative_monitored"

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        fallback_config: FallbackConfig,
        scenario: ScenarioConfig,
        safety_config: SafetyConfig,
    ) -> None:
        self.neural = V21AControlPolicyController(checkpoint_path, device)
        self.fallback = DeterministicFallbackController(fallback_config, scenario.airports)
        self.monitor = RuntimeSafetyMonitor(safety_config)
        self.model_checkpoint_identifier = self.neural.model_checkpoint_identifier
        self.last_proposed_action: ControllerOutput | None = None
        self.last_monitor_decision = None

    @property
    def last_inference_latency_ms(self) -> float:
        return self.neural.last_inference_latency_ms

    def reset(self) -> None:
        self.neural.reset()
        self.fallback.reset()
        self.monitor.reset()
        self.last_proposed_action = None
        self.last_monitor_decision = None

    def act(self, observation: ControllerInput) -> ControllerOutput:
        proposed = self.neural.act(observation)
        fallback = self.fallback.act(observation)
        decision = self.monitor.decide(
            observation, proposed.control_commands, fallback.control_commands
        )
        self.last_proposed_action = proposed
        self.last_monitor_decision = decision
        chosen = fallback.model_copy(deep=True) if decision.fallback_activated else proposed.model_copy(deep=True)
        chosen.control_commands = decision.final_action
        chosen.safety_decision = SafetyDecision(
            status=(
                "fallback" if decision.fallback_activated
                else "accepted" if decision.status == "accept" else "modified"
            ),
            ai_action_accepted=decision.status == "accept",
            fallback_activated=decision.fallback_activated,
            violated_constraints=list(decision.violated_properties),
            explanation=decision.reason,
        )
        return chosen


def _canonical(document: dict[str, Any], field: str) -> str:
    return object_hash({key: value for key, value in document.items() if key != field})


def _load_flight_metadata(approved: Path, split: str, role: str) -> list[dict[str, Any]]:
    assert_role_split(role, split)
    rows: list[dict[str, Any]] = []
    with (approved / "raw/flights.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == "final_sealed_test":
                raise PermissionError("sealed flight found in approved nonsealed export")
            if row["split"] == split:
                rows.append(row)
    if len(rows) != EXPECTED_FLIGHTS:
        raise ValueError(f"{split} flight count mismatch: {len(rows)}")
    return rows


def _expected_scenario_hashes(approved: Path, split: str, role: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    count = 0
    for row in iter_approved_records(approved, split=split, role=role):
        count += 1
        hashes.setdefault(row["flight_id"], row["effective_scenario_configuration_hash"])
    if count != EXPECTED_TRANSITIONS or len(hashes) != EXPECTED_FLIGHTS:
        raise ValueError(f"{split} transition geometry mismatch: {len(hashes)}/{count}")
    return hashes


def _flight_spec(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": row["split"],
        "seed": row["random_seed"],
        "tier": row["difficulty_tier"],
        "family": row["scenario_family"],
        "severity_band": row["severity_band"],
        "scenario_instance_id": row["scenario_instance_id"],
        "duration_seconds": 26.0,
        "initial_state": row["initial_state"],
        "terrain_clearance_ft": row["terrain_clearance_ft"],
        "component_severity": row["component_severity"],
        "activation_seconds": row["activation_seconds"],
        "actuator_delay_seconds": row["actuator_delay_seconds"],
        "sensor_bias_kind": row["sensor_bias_kind"],
        "sensor_bias_value": row["sensor_bias_value"],
        "turbulence_intensity": row["turbulence_intensity"],
        "crosswind_maximum_kts": row["crosswind_maximum_kts"],
        "engine_side": row["engine_side"],
    }


def nonsealed_preflight(root: str | Path, split: str) -> dict[str, Any]:
    root = Path(root).resolve()
    if split not in AUTHORIZED_SPLITS:
        raise PermissionError(f"unauthorized nonsealed split: {split}")
    role = AUTHORIZED_SPLITS[split]
    assert_role_split(role, split)
    freeze_path = root / "artifacts/research_stage/final_model_selection_freeze_v2_1b.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("freeze_logical_sha256") != MODEL_SELECTION_FREEZE_LOGICAL_HASH
        or _canonical(freeze, "freeze_logical_sha256") != MODEL_SELECTION_FREEZE_LOGICAL_HASH
    ):
        raise ValueError("model-selection freeze hash mismatch")
    if freeze.get("nonsealed_evaluation_checkpoint_set") != "representative_seed_17001_only":
        raise ValueError("representative-only held-out path is not frozen")
    checkpoint = root / freeze["representative"]["checkpoint_path"]
    if file_hash(checkpoint) != REPRESENTATIVE_CHECKPOINT_HASH:
        raise ValueError("representative checkpoint hash mismatch")
    configuration_path = root / "configs/evaluation/final_nonsealed_evaluation_v2_1b.yaml"
    configuration = load_yaml(configuration_path)
    required = {
        "authorized_checkpoint_set": "representative_seed_17001_only",
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_checkpoint_sha256": REPRESENTATIVE_CHECKPOINT_HASH,
        "model_selection_freeze_logical_sha256": MODEL_SELECTION_FREEZE_LOGICAL_HASH,
        "approved_dataset_logical_sha256": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "authorized_sequence": ["challenge", "ood_stress"],
        "flight_count_per_split": EXPECTED_FLIGHTS,
        "transition_count_per_split": EXPECTED_TRANSITIONS,
        "sealed_access_authorized": False,
        "all_five_seed_heldout_evaluation_authorized": False,
    }
    for key, value in required.items():
        if configuration.get(key) != value:
            raise ValueError(f"nonsealed configuration mismatch: {key}")
    if split == "ood_stress":
        challenge_path = root / "results/final_nonsealed_evaluation_v2_1b/challenge/summary.json"
        if not challenge_path.is_file():
            raise PermissionError("Challenge must complete before OOD-style simulator stress")
        challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
        if challenge.get("status") != "COMPLETE" or challenge.get("sealed_opened") is not False:
            raise ValueError("Challenge completion provenance is invalid")
    return {
        "status": "PASS",
        "split": split,
        "role": role,
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_checkpoint_sha256": REPRESENTATIVE_CHECKPOINT_HASH,
        "model_selection_freeze_logical_sha256": MODEL_SELECTION_FREEZE_LOGICAL_HASH,
        "model_selection_freeze_file_sha256": file_hash(freeze_path),
        "evaluation_configuration_sha256": file_hash(configuration_path),
        "approved_dataset_logical_sha256": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "final_training_configuration_logical_sha256": FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH,
        "five_seed_provenance_logical_sha256": FIVE_SEED_PROVENANCE_LOGICAL_HASH,
        "challenge_opened": split == "challenge",
        "ood_opened": split == "ood_stress",
        "sealed_opened": False,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int]:
    if total <= 0:
        return {"successes": successes, "total": total, "rate": 0.0, "lower": 0.0, "upper": 1.0}
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return {"successes": successes, "total": total, "rate": rate, "lower": center - margin, "upper": center + margin}


def continuous_summary(values: Iterable[float | int]) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {"count": 0, "mean": None, "median": None, "sample_standard_deviation": None,
                "minimum": None, "maximum": None, "mean_bootstrap_95_ci": None,
                "median_bootstrap_95_ci": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = data[rng.integers(0, data.size, size=(BOOTSTRAP_RESAMPLES, data.size))]
    means = samples.mean(axis=1)
    medians = np.median(samples, axis=1)
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "sample_standard_deviation": float(data.std(ddof=1)) if data.size > 1 else 0.0,
        "minimum": float(data.min()),
        "maximum": float(data.max()),
        "mean_bootstrap_95_ci": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "median_bootstrap_95_ci": [float(x) for x in np.quantile(medians, [0.025, 0.975])],
    }


def latency_summary(values: list[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return {"steps": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "maximum_ms": None}
    return {
        "steps": len(data),
        "p50_ms": float(np.quantile(data, 0.50)),
        "p95_ms": float(np.quantile(data, 0.95)),
        "p99_ms": float(np.quantile(data, 0.99)),
        "maximum_ms": float(data.max()),
    }


def _controls(value: ControlCommands | dict[str, float]) -> dict[str, float]:
    return value if isinstance(value, dict) else value.model_dump(mode="json")


def _run_flight(
    *,
    root: Path,
    flight: dict[str, Any],
    scenario: ScenarioConfig,
    condition: str,
    checkpoint: Path,
    device: torch.device,
    aircraft: AircraftConfig,
    fallback_config: FallbackConfig,
    safety_config: SafetyConfig,
    metrics: dict[str, Any],
    trace_stream,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    simulator = SimpleAircraftSimulator(aircraft, scenario)
    if condition == "unmonitored":
        controller: Any = V21AControlPolicyController(checkpoint, device)
    elif condition == "monitored":
        controller = V21AMonitoredController(
            checkpoint, device, fallback_config, scenario, safety_config
        )
    else:
        raise ValueError(condition)
    controller.reset()
    observation = simulator.reset(int(flight["random_seed"]))
    samples: list[dict[str, float]] = []
    executed: list[list[float]] = []
    proposed: list[list[float]] = []
    selected: list[list[float]] = []
    controller_latencies: list[float] = []
    inference_latencies: list[float] = []
    monitor_latencies: list[float] = []
    fallback_flags: list[bool] = []
    monitor_modified: list[bool] = []
    statuses: Counter[str] = Counter()
    solver_results: Counter[str] = Counter()
    violated_properties: Counter[str] = Counter()
    step = 0
    while not simulator.is_terminal():
        truth_before = simulator.audit_snapshot()
        sample = _metric_sample(
            truth_before["true_state"], observation.environment.terrain_clearance_ft, aircraft
        )
        sample["timestamp_seconds"] = float(observation.timestamp_seconds)
        samples.append(sample)
        started = perf_counter()
        action = controller.act(observation)
        full_latency = (perf_counter() - started) * 1000.0
        controller_latencies.append(full_latency)
        inference_latencies.append(float(controller.last_inference_latency_ms))
        proposed_output = controller.last_proposed_action or action
        proposed_values = [float(getattr(proposed_output.control_commands, name)) for name in CONTROL_NAMES]
        selected_values = [float(getattr(action.control_commands, name)) for name in CONTROL_NAMES]
        next_observation = simulator.step(action)
        truth_after = simulator.audit_snapshot()
        executed_values = [float(truth_after["applied_control_commands"][name]) for name in CONTROL_NAMES]
        decision = getattr(controller, "last_monitor_decision", None)
        fallback = bool(decision.fallback_activated) if decision else False
        modified = any(abs(a - b) > 1e-12 for a, b in zip(proposed_values, selected_values, strict=True))
        fallback_flags.append(fallback)
        monitor_modified.append(modified)
        if decision:
            statuses[decision.status] += 1
            solver_results[decision.solver_result] += 1
            violated_properties.update(decision.violated_properties)
            monitor_latencies.append(float(decision.monitor_duration_ms))
        proposed.append(proposed_values)
        selected.append(selected_values)
        executed.append(executed_values)
        trace = {
            "flight_id": flight["flight_id"],
            "scenario_instance_id": flight["scenario_instance_id"],
            "split": flight["split"],
            "controller_condition": condition,
            "timestep_index": step,
            "timestamp_seconds": float(observation.timestamp_seconds),
            "proposed_action": dict(zip(CONTROL_NAMES, proposed_values, strict=True)),
            "monitor_selected_action": dict(zip(CONTROL_NAMES, selected_values, strict=True)),
            "executed_action": dict(zip(CONTROL_NAMES, executed_values, strict=True)),
            "monitor_decision": decision.to_dict() if decision else None,
            "inference_latency_ms": float(controller.last_inference_latency_ms),
            "full_controller_latency_ms": full_latency,
        }
        trace_stream.write(json.dumps(trace, separators=(",", ":"), allow_nan=False) + "\n")
        observation = next_observation
        step += 1
    reference_candidates = [
        float(item.activation_time_seconds)
        for item in scenario.failures
        if item.observable and item.detected and item.activation_time_seconds is not None
    ]
    reference_seconds = max(reference_candidates, default=0.0)
    recovery = strict_recovery_result(
        samples, metrics["recovery"], reference_seconds, aircraft.time_step_seconds
    )
    safety = safety_summary(samples, metrics["safety"], aircraft.time_step_seconds)
    reason = simulator.termination_reason() or "unknown"
    termination_class = classify_termination(reason, metrics["termination"])
    commands = np.asarray(executed, dtype=np.float64)
    saturated = np.concatenate(
        (np.abs(commands[:, :3]) >= 1.0 - 1e-12,
         (commands[:, 3:] <= 1e-12) | (commands[:, 3:] >= 1.0 - 1e-12)), axis=1
    )
    oscillation = float(np.abs(np.diff(commands, axis=0)).sum()) if len(commands) > 1 else 0.0
    fallback_activations = sum(
        flag and (index == 0 or not fallback_flags[index - 1])
        for index, flag in enumerate(fallback_flags)
    )
    failure_count = len(scenario.failures)
    concurrency = "nominal" if failure_count == 0 else "isolated" if failure_count == 1 else "concurrent"
    result = {
        "flight_id": flight["flight_id"],
        "scenario_instance_id": flight["scenario_instance_id"],
        "split": flight["split"],
        "controller_condition": condition,
        "random_seed": flight["random_seed"],
        "difficulty_tier": flight["difficulty_tier"],
        "scenario_family": flight["scenario_family"],
        "severity_band": flight["severity_band"],
        "isolated_vs_concurrent": concurrency,
        "failure_count": failure_count,
        "termination_reason": reason,
        "termination_class": termination_class,
        "invalid_experiment": termination_class == "invalid_experiment",
        "strict_recovery_success": bool(recovery["success"]),
        "strict_recovery_time_seconds": recovery["achievement_time_seconds"],
        "lost_after_recovery": recovery["lost_after_achievement"],
        "recovery_timer_reference_seconds": reference_seconds,
        "minimum_airspeed_kts": min(item["stall_margin_kts"] + aircraft.stall_speed_kts for item in samples),
        "maximum_absolute_bank_deg": max(abs(item["bank_deg"]) for item in samples),
        "maximum_absolute_pitch_deg": max(abs(item["pitch_deg"]) for item in samples),
        "minimum_terrain_clearance_ft": min(item["terrain_clearance_ft"] for item in samples),
        "safety": safety,
        "command_saturation_frequency": float(saturated.mean()),
        "command_saturation_per_output": {
            name: float(saturated[:, index].mean()) for index, name in enumerate(CONTROL_NAMES)
        },
        "control_oscillation_total": oscillation,
        "control_oscillation_per_second": oscillation / max(step * aircraft.time_step_seconds, aircraft.time_step_seconds),
        "monitor_modification_count": sum(monitor_modified),
        "monitor_intervention_frequency": sum(monitor_modified) / step,
        "fallback_activation_count": int(fallback_activations),
        "fallback_step_count": sum(fallback_flags),
        "fallback_frequency": sum(fallback_flags) / step,
        "fallback_duration_seconds": sum(fallback_flags) * aircraft.time_step_seconds,
        "intervention_taxonomy": {
            "status_counts": dict(statuses),
            "solver_result_counts": dict(solver_results),
            "violated_property_counts": dict(violated_properties),
        },
        "latency": {
            "inference": latency_summary(inference_latencies),
            "monitor": latency_summary(monitor_latencies),
            "full_controller": latency_summary(controller_latencies),
        },
        "step_count": step,
        "proposed_action_trace_preserved": True,
        "executed_action_trace_preserved": True,
    }
    return result, {
        "inference": inference_latencies,
        "monitor": monitor_latencies,
        "full_controller": controller_latencies,
    }


def _aggregate(rows: list[dict[str, Any]], latency: dict[str, list[float]]) -> dict[str, Any]:
    valid = [row for row in rows if not row["invalid_experiment"]]
    successes = sum(row["strict_recovery_success"] for row in valid)
    recovery_times = [
        row["strict_recovery_time_seconds"] for row in valid
        if row["strict_recovery_time_seconds"] is not None
    ]
    violation_flights = sum(row["safety"]["flight_has_violation"] for row in valid)
    return {
        "flight_count": len(rows),
        "valid_flight_count": len(valid),
        "invalid_experiment_count": len(rows) - len(valid),
        "strict_recovery": wilson_interval(successes, len(valid)),
        "strict_recovery_time_seconds": {
            **continuous_summary(recovery_times),
            "right_censored_count": len(valid) - len(recovery_times),
        },
        "safety_violation_flights": wilson_interval(violation_flights, len(valid)),
        "safety_violation_property_step_count": sum(row["safety"]["violation_property_step_count"] for row in valid),
        "safety_violation_duration_seconds": sum(row["safety"]["violation_duration_seconds"] for row in valid),
        "fallback_activation_count": sum(row["fallback_activation_count"] for row in valid),
        "fallback_step_frequency": (
            sum(row["fallback_step_count"] for row in valid) / sum(row["step_count"] for row in valid)
            if valid else 0.0
        ),
        "monitor_intervention_frequency": (
            sum(row["monitor_modification_count"] for row in valid) / sum(row["step_count"] for row in valid)
            if valid else 0.0
        ),
        "latency": {name: latency_summary(values) for name, values in latency.items()},
        "termination_reason_counts": dict(Counter(row["termination_reason"] for row in rows)),
    }


def _grouped(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("difficulty_tier", "isolated_vs_concurrent", "scenario_family"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result[field] = {
            key: _aggregate(value, {name: [] for name in ("inference", "monitor", "full_controller")})
            for key, value in sorted(groups.items())
        }
    return result


def _paired(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[row["flight_id"]][row["controller_condition"]] = row
    if len(pairs) != EXPECTED_FLIGHTS or any(set(pair) != {"unmonitored", "monitored"} for pair in pairs.values()):
        raise ValueError("paired flight identity mismatch")
    both = monitored_only = unmonitored_only = neither = 0
    time_differences = []
    violation_duration_differences = []
    for pair in pairs.values():
        unmonitored = pair["unmonitored"]
        monitored = pair["monitored"]
        left = unmonitored["strict_recovery_success"]
        right = monitored["strict_recovery_success"]
        both += left and right
        monitored_only += (not left) and right
        unmonitored_only += left and (not right)
        neither += (not left) and (not right)
        if unmonitored["strict_recovery_time_seconds"] is not None and monitored["strict_recovery_time_seconds"] is not None:
            time_differences.append(
                monitored["strict_recovery_time_seconds"] - unmonitored["strict_recovery_time_seconds"]
            )
        violation_duration_differences.append(
            monitored["safety"]["violation_duration_seconds"]
            - unmonitored["safety"]["violation_duration_seconds"]
        )
    return {
        "pair_count": len(pairs),
        "both_recovered": both,
        "monitored_only_recovered": monitored_only,
        "unmonitored_only_recovered": unmonitored_only,
        "neither_recovered": neither,
        "discordant_pair_count": monitored_only + unmonitored_only,
        "recovery_time_paired_difference_monitored_minus_unmonitored": continuous_summary(time_differences),
        "violation_duration_paired_difference_monitored_minus_unmonitored": continuous_summary(violation_duration_differences),
    }


def run_nonsealed_evaluation(root: str | Path, split: str) -> dict[str, Any]:
    root = Path(root).resolve()
    preflight = nonsealed_preflight(root, split)
    output = root / "results/final_nonsealed_evaluation_v2_1b" / split
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonsealed evaluation: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    approved = root / "datasets/approved_policy_data" / APPROVED_ID
    role = AUTHORIZED_SPLITS[split]
    flights = _load_flight_metadata(approved, split, role)
    expected_hashes = _expected_scenario_hashes(approved, split, role)
    aircraft = load_config(root / "configs/aircraft/simple_twin.yaml", AircraftConfig)
    fallback = load_config(root / "configs/research/final_fallback_v1.yaml", FallbackConfig)
    safety_config = load_config(root / "configs/research/final_monitor_v1.yaml", SafetyConfig)
    metrics = load_yaml(root / "configs/research/final_metrics_v1.yaml")
    checkpoint = root / "checkpoints/final_policy_v1/moderate/final_training_v2_1a/seed_17001/best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traces_path = output / "action_traces.jsonl"
    flight_records: list[dict[str, Any]] = []
    all_latency: dict[str, dict[str, list[float]]] = {
        condition: {name: [] for name in ("inference", "monitor", "full_controller")}
        for condition in ("unmonitored", "monitored")
    }
    started = datetime.now(timezone.utc)
    started_timer = perf_counter()
    with traces_path.open("x", encoding="utf-8", newline="\n") as trace_stream:
        for flight in flights:
            scenario = _scenario(root, _flight_spec(flight))
            observed_hash = object_hash(scenario.model_dump(mode="json"))
            if observed_hash != expected_hashes[flight["flight_id"]]:
                raise ValueError(f"scenario reconstruction hash mismatch: {flight['flight_id']}")
            for condition in ("unmonitored", "monitored"):
                record, latencies = _run_flight(
                    root=root,
                    flight=flight,
                    scenario=scenario,
                    condition=condition,
                    checkpoint=checkpoint,
                    device=device,
                    aircraft=aircraft,
                    fallback_config=fallback,
                    safety_config=safety_config,
                    metrics=metrics,
                    trace_stream=trace_stream,
                )
                flight_records.append(record)
                for name, values in latencies.items():
                    all_latency[condition][name].extend(values)
    flight_records_path = output / "flight_records.jsonl"
    with flight_records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in flight_records:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    by_condition = {
        condition: [row for row in flight_records if row["controller_condition"] == condition]
        for condition in ("unmonitored", "monitored")
    }
    summary = {
        "result_version": "final-nonsealed-evaluation-v2.1b-v1",
        "status": "COMPLETE",
        "split": split,
        "split_wording": "Challenge" if split == "challenge" else "OOD-style simulator stress",
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_checkpoint_sha256": REPRESENTATIVE_CHECKPOINT_HASH,
        "controller_conditions": ["unmonitored", "monitored"],
        "overall": {
            condition: _aggregate(by_condition[condition], all_latency[condition])
            for condition in ("unmonitored", "monitored")
        },
        "paired": _paired(flight_records),
        "grouped": {
            condition: _grouped(by_condition[condition])
            for condition in ("unmonitored", "monitored")
        },
        "intervention_taxonomy": {
            condition: {
                "status_counts": dict(sum((Counter(row["intervention_taxonomy"]["status_counts"]) for row in by_condition[condition]), Counter())),
                "solver_result_counts": dict(sum((Counter(row["intervention_taxonomy"]["solver_result_counts"]) for row in by_condition[condition]), Counter())),
                "violated_property_counts": dict(sum((Counter(row["intervention_taxonomy"]["violated_property_counts"]) for row in by_condition[condition]), Counter())),
            }
            for condition in ("unmonitored", "monitored")
        },
        "flight_record_count": len(flight_records),
        "trace_record_count": sum(row["step_count"] for row in flight_records),
        "flight_records_path": str(flight_records_path.relative_to(root)).replace("\\", "/"),
        "flight_records_sha256": file_hash(flight_records_path),
        "action_traces_path": str(traces_path.relative_to(root)).replace("\\", "/"),
        "action_traces_sha256": file_hash(traces_path),
        "evaluation_start_utc": started.isoformat(),
        "evaluation_end_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": perf_counter() - started_timer,
        "source_git_revision": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "source_file_sha256": file_hash(root / "src/aircraft_recovery/evaluation/v2_1b_nonsealed.py"),
        "evaluation_configuration_sha256": preflight["evaluation_configuration_sha256"],
        "model_selection_freeze_logical_sha256": MODEL_SELECTION_FREEZE_LOGICAL_HASH,
        "approved_dataset_logical_sha256": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "challenge_opened": True,
        "ood_opened": split == "ood_stress",
        "sealed_opened": False,
        "model_changed_from_results": False,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
