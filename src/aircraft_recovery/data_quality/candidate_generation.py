"""Stage 1: immutable raw candidate generation from simulator and expert."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aircraft_recovery.common.observations import OBSERVATION_FEATURES, observation_to_vector
from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config, load_yaml
from aircraft_recovery.controllers import DeterministicFallbackController, FallbackConfig
from aircraft_recovery.data_quality.common import (
    CONTROL_NAMES,
    SCHEMA_VERSION,
    WORKFLOW_VERSION,
    file_hash,
    object_hash,
    record_hash,
    write_json,
)
from aircraft_recovery.simulator import SimpleAircraftSimulator


class CandidateFlightConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    seed: int
    split: Literal[
        "train", "development", "challenge", "ood_stress", "final_sealed_test",
        "validation", "test",
    ]
    difficulty_tier: int = Field(default=0, ge=0, le=5)
    scenario_family: str = "legacy"
    severity_band: str = "unspecified"
    scenario_instance_id: str | None = None
    initial_state_overrides: dict[str, float] = Field(default_factory=dict)
    environment_overrides: dict[str, float] = Field(default_factory=dict)
    failure_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SyntheticCorruptionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flight_index: int = Field(ge=0)
    timestep_index: int = Field(ge=0)
    kind: Literal["wrong_feature_order"] = "wrong_feature_order"


class CandidateGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]+$")
    output_root: str = "datasets/candidate_policy_data"
    aircraft_config: str
    fallback_config: str
    maximum_steps_per_flight: int | None = Field(default=None, gt=0)
    label_semantics: Literal["expert_proposed_action", "runtime_assurance_imitation"] = "expert_proposed_action"
    flights: list[CandidateFlightConfig] = Field(min_length=1)
    synthetic_corruption: SyntheticCorruptionConfig | None = None


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _controls(action: Any) -> list[float]:
    return [float(getattr(action.control_commands, name)) for name in CONTROL_NAMES]


def _outcome(reason: str | None, truncated: bool) -> str:
    if truncated:
        return "truncated_smoke"
    if reason in {"duration_complete", "stable_recovery_achieved", "diversion_target_reached", "approach_completed"}:
        return "successful"
    return "unsuccessful"


def _effective_scenario(flight: CandidateFlightConfig) -> ScenarioConfig:
    payload = load_yaml(flight.scenario)
    payload["seed"] = flight.seed
    payload["initial_state"] = {**payload["initial_state"], **flight.initial_state_overrides}
    payload["environment"] = {**payload["environment"], **flight.environment_overrides}
    failures = []
    for failure in payload.get("failures", []):
        override = flight.failure_overrides.get(str(failure["failure_id"]), {})
        merged = {**failure, **{key: value for key, value in override.items() if key != "parameters"}}
        if "parameters" in override:
            merged["parameters"] = {**failure.get("parameters", {}), **override["parameters"]}
        failures.append(merged)
    payload["failures"] = failures
    return ScenarioConfig.model_validate(payload)


def generate_candidate_dataset(config: CandidateGenerationConfig) -> dict[str, Any]:
    output = Path(config.output_root) / config.dataset_id
    if output.exists():
        raise FileExistsError(f"Candidate dataset already exists and is immutable: {output}")
    raw = output / "raw"
    manifests = output / "manifests"
    validation = output / "validation"
    review = output / "review"
    for directory in (raw, manifests, validation, review):
        directory.mkdir(parents=True, exist_ok=False)

    aircraft = load_config(config.aircraft_config, AircraftConfig)
    fallback_config = load_config(config.fallback_config, FallbackConfig)
    git_commit = _git_hash()
    records: list[dict[str, Any]] = []
    flights: list[dict[str, Any]] = []
    split_names = ("train", "development", "challenge", "ood_stress", "validation", "test")
    split_flights: dict[str, list[str]] = {name: [] for name in split_names}
    for flight_index, flight_config in enumerate(config.flights):
        scenario_path = Path(flight_config.scenario)
        scenario = _effective_scenario(flight_config)
        source_hash = file_hash(scenario_path)
        scenario_instance_id = flight_config.scenario_instance_id or f"{scenario.scenario_id}:{flight_config.seed}:{flight_index:03d}"
        flight_id = f"{config.dataset_id}:{scenario_instance_id}"
        split_flights[flight_config.split].append(flight_id)
        simulator = SimpleAircraftSimulator(aircraft, scenario)
        expert = DeterministicFallbackController(fallback_config, scenario.airports)
        observation = simulator.reset(flight_config.seed)
        expert.reset()
        buffered: list[dict[str, Any]] = []
        step = 0
        while not simulator.is_terminal() and (
            config.maximum_steps_per_flight is None or step < config.maximum_steps_per_flight
        ):
            truth_before = simulator.audit_snapshot()
            proposed = expert.act(observation)
            next_observation = simulator.step(proposed)
            truth_after = simulator.audit_snapshot()
            feature_names = list(OBSERVATION_FEATURES)
            declared_corruption = None
            corrupt = config.synthetic_corruption
            if corrupt and corrupt.flight_index == flight_index and corrupt.timestep_index == step:
                feature_names[0], feature_names[1] = feature_names[1], feature_names[0]
                declared_corruption = corrupt.kind
            record_id = f"{flight_id}:step:{step:06d}"
            active_failures = truth_before["ground_truth_failures"]
            active_failures_after = truth_after["ground_truth_failures"]
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": config.dataset_id,
                "record_id": record_id,
                "flight_id": flight_id,
                "scenario_id": scenario.scenario_id,
                "scenario_instance_id": scenario_instance_id,
                "difficulty_tier": flight_config.difficulty_tier,
                "scenario_family": flight_config.scenario_family,
                "severity_band": flight_config.severity_band,
                "random_seed": flight_config.seed,
                "split": flight_config.split,
                "timestep_index": step,
                "simulation_timestamp_seconds": float(observation.timestamp_seconds),
                "state_after_timestamp_seconds": float(next_observation.timestamp_seconds),
                "physics_timestep_seconds": float(aircraft.time_step_seconds),
                "aircraft_state_before": truth_before["true_state"],
                "ordered_observation_42": observation_to_vector(observation).astype(float).tolist(),
                "observation_feature_names": feature_names,
                "expert_proposed_controls": _controls(proposed),
                "label_semantics": config.label_semantics,
                "monitor_status": None,
                "monitor_modified_controls": None,
                "executed_controls": [
                    float(truth_after["applied_control_commands"][name]) for name in CONTROL_NAMES
                ],
                "actuator_state": observation.control_health.model_dump(mode="json"),
                "engine_state": observation.engine_health.model_dump(mode="json"),
                "active_failures": active_failures,
                "controller_known_failures": list(observation.active_failures),
                "ground_truth_failures": active_failures,
                "ground_truth_failures_after": active_failures_after,
                "failure_schedule": [failure.model_dump(mode="json") for failure in scenario.failures],
                "actuator_effectiveness": {
                    "elevator": observation.control_health.elevator_effectiveness,
                    "aileron": observation.control_health.aileron_effectiveness,
                    "rudder": observation.control_health.rudder_effectiveness,
                },
                "actuator_delay_seconds": observation.control_health.actuator_delay_ms / 1000.0,
                "sensor_error_failures": [item for item in active_failures if "sensor" in str(item.get("affected_subsystem", "")) or item.get("failure_type") in {"airspeed_bias", "frozen_sensor_value", "temporary_missing_sensor_data"}],
                "disturbance_values": observation.environment.model_dump(mode="json"),
                "disturbance_realization_id": f"{scenario_instance_id}:seed:{flight_config.seed}",
                "aircraft_state_after": truth_after["true_state"],
                "termination_status": {"terminal": False, "reason": None},
                "recoverability_label": scenario.expected_recoverability,
                "outcome_label": None,
                "source_configuration_path": flight_config.scenario,
                "source_configuration_hash": source_hash,
                "effective_scenario_configuration_hash": object_hash(scenario.model_dump(mode="json")),
                "simulator_configuration_hash": file_hash(config.aircraft_config),
                "expert_configuration_hash": file_hash(config.fallback_config),
                "git_commit_hash": git_commit,
                "generator_version": WORKFLOW_VERSION,
                "simulator_version": "simple-aircraft-simulator-v1",
                "declared_synthetic_corruption": declared_corruption,
            }
            record["record_hash"] = record_hash(record)
            buffered.append(record)
            observation = next_observation
            step += 1
        truncated = not simulator.is_terminal()
        reason = "smoke_step_limit" if truncated else simulator.termination_reason()
        outcome = _outcome(reason, truncated)
        if buffered:
            buffered[-1]["termination_status"] = {"terminal": True, "reason": reason}
        for record in buffered:
            record["outcome_label"] = outcome
            record["record_hash"] = record_hash(record)
        records.extend(buffered)
        flights.append({
            "dataset_id": config.dataset_id,
            "flight_id": flight_id,
            "scenario_id": scenario.scenario_id,
            "scenario_instance_id": scenario_instance_id,
            "difficulty_tier": flight_config.difficulty_tier,
            "scenario_family": flight_config.scenario_family,
            "severity_band": flight_config.severity_band,
            "random_seed": flight_config.seed,
            "split": flight_config.split,
            "transition_count": len(buffered),
            "recoverability_label": scenario.expected_recoverability,
            "outcome_label": outcome,
            "termination_reason": reason,
            "source_configuration_path": flight_config.scenario,
            "source_configuration_hash": source_hash,
        })

    transitions_path = raw / "transitions.jsonl"
    flights_path = raw / "flights.jsonl"
    transitions_path.write_text("".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n" for row in records), encoding="utf-8")
    flights_path.write_text("".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n" for row in flights), encoding="utf-8")
    schema = {
        "schema_version": SCHEMA_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "canonical_feature_order": list(OBSERVATION_FEATURES),
        "control_order": list(CONTROL_NAMES),
        "feature_count": 42,
        "control_count": 5,
        "required_record_fields": sorted(records[0]) if records else [],
    }
    write_json(manifests / "schema.json", schema)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": WORKFLOW_VERSION,
        "git_commit_hash": git_commit,
        "generation_configuration": config.model_dump(mode="json"),
        "generation_configuration_hash": object_hash(config.model_dump(mode="json")),
        "record_count": len(records),
        "flight_count": len(flights),
        "splits": split_flights,
        "feature_contract": {"version": "phase1-42-v1", "count": 42, "order": list(OBSERVATION_FEATURES)},
        "action_contract": {"count": 5, "order": list(CONTROL_NAMES), "bounds": [[-1,1],[-1,1],[-1,1],[0,1],[0,1]], "label": config.label_semantics},
        "units": {"time": "seconds", "altitude_and_clearance": "feet", "airspeed": "knots", "angles": "degrees", "vertical_speed": "feet_per_minute", "controls": "normalized"},
        "scenario_inventory": [{"scenario_instance_id": row["scenario_instance_id"], "scenario_id": row["scenario_id"], "tier": row["difficulty_tier"], "family": row["scenario_family"], "severity_band": row["severity_band"], "seed": row["random_seed"], "split": row["split"]} for row in flights],
        "seed_inventory": sorted({row["random_seed"] for row in flights}),
        "tier_flight_counts": {str(tier): sum(row["difficulty_tier"] == tier for row in flights) for tier in range(6)},
        "family_flight_counts": {family: sum(row["scenario_family"] == family for row in flights) for family in sorted({row["scenario_family"] for row in flights})},
        "raw_records_unfiltered": True,
        "sealed_policy_test_loaded": False,
        "declared_synthetic_corruption": config.synthetic_corruption.model_dump() if config.synthetic_corruption else None,
    }
    write_json(manifests / "generation_manifest.json", manifest)
    checksums = {
        str(path.relative_to(output)).replace("\\", "/"): file_hash(path)
        for path in (
            transitions_path, flights_path, manifests / "schema.json",
            manifests / "generation_manifest.json",
        )
    }
    write_json(manifests / "file_checksums.json", checksums)
    return manifest
