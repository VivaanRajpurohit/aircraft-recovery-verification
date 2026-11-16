"""Deterministic, resumable generation for the activated v2.1 candidate freeze."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

from aircraft_recovery.common.observations import OBSERVATION_FEATURES, observation_to_vector
from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config, load_yaml
from aircraft_recovery.controllers import DeterministicFallbackController, FallbackConfig
from aircraft_recovery.data_quality.candidate_generation import _controls, _git_hash, _outcome
from aircraft_recovery.data_quality.common import (
    CONTROL_NAMES, SCHEMA_VERSION, WORKFLOW_VERSION, file_hash, object_hash,
    record_hash, write_json,
)
from aircraft_recovery.data_quality.validation import VALIDATOR_VERSION, validate_candidate_dataset
from aircraft_recovery.research.holdout_audit import resolved_v2_1_schedule
from aircraft_recovery.research.metric_contracts import inside_recovery_envelope, safety_violations
from aircraft_recovery.research.pre_generation import expand_seed_registry
from aircraft_recovery.simulator import SimpleAircraftSimulator


DATASET_ID = "final_policy_candidate_v2_1"
ACTIVE_FREEZE_HASH = "c0b831ede8a5d75027dcce70ec1c116fc5ec58d479cf500939f2075e3d16d0dc"
SPLITS = ("train", "development", "challenge", "ood_stress", "final_sealed_test")
ENGINE_FAMILIES = {
    "isolated_engine_loss_mild_state", "isolated_engine_loss_moderate_state",
    "engine_plus_aileron", "severe_engine_plus_aileron",
    "right_engine_rudder_crosswind", "boundary_engine_aileron",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cycle(values: list[Any], index: int, default: Any = None) -> Any:
    return values[index % len(values)] if values else default


def _active_registry(root: Path) -> dict[str, Any]:
    registry = json.loads((root / "artifacts/research_stage/final_generation_freeze_registry.json").read_text(encoding="utf-8"))
    declared = registry.pop("registry_hash_excluding_this_field")
    if object_hash(registry) != declared:
        raise ValueError("active freeze registry hash mismatch")
    active = registry.get("active_generation_freeze") or {}
    if active.get("freeze_id") != "final_generation_freeze_v2_1" or active.get("freeze_hash") != ACTIVE_FREEZE_HASH:
        raise ValueError("v2.1 is not the sole expected active freeze")
    if not registry.get("generation_authorized"):
        raise PermissionError("candidate generation is not authorized by registry")
    return {**registry, "registry_hash": declared}


def build_final_flight_registry(root: Path) -> list[dict[str, Any]]:
    """Expand the frozen matrix and seeds into a deterministic flight registry."""
    _active_registry(root)
    design = load_yaml(root / "configs/training/final_policy_dataset_design_v1.yaml")
    tiers = load_yaml(root / "configs/research/final_tiers_v1.yaml")
    schedule, _ = resolved_v2_1_schedule(root)
    seed_registry = json.loads((root / "configs/research/final_seed_registry_v1.json").read_text(encoding="utf-8"))
    seeds = expand_seed_registry(seed_registry)
    seed_index = {split: 0 for split in SPLITS}
    specs: list[dict[str, Any]] = []
    common = tiers["common"]
    severity_band = {0: "none", 1: "mild", 2: "moderate_isolated", 3: "moderate_concurrent", 4: "severe_in_domain", 5: "boundary_in_domain"}
    for matrix_row in design["scenario_matrix"]:
        tier = int(matrix_row["tier"])
        family = str(matrix_row["family"])
        semantics = schedule["family_parameter_semantics"][family]
        parameters = set(semantics["parameters"])
        tier_schedule = schedule["tiers"][tier]
        tier_bounds = tiers["tiers"][tier]
        for split, count in zip(SPLITS, matrix_row["split_allocation"], strict=True):
            for local_index in range(int(count)):
                seed = seeds[split][seed_index[split]]
                seed_index[split] += 1
                fraction = float(_cycle(schedule["initial_condition_grid"][split], local_index))
                def ranged(name: str) -> float:
                    low, high = map(float, tier_bounds.get(name, common[name]))
                    return low + fraction * (high - low)
                component = _cycle(tier_schedule.get("component_severity", {}).get(split, []), local_index)
                activation = _cycle(tier_schedule.get("activation_seconds", {}).get(split, []), local_index)
                delay = _cycle(tier_schedule.get("actuator_delay_seconds", {}).get(split, []), local_index)
                turbulence = _cycle(tier_schedule.get("turbulence_intensity", {}).get(split, []), local_index, 0.0)
                crosswind = _cycle(tier_schedule.get("crosswind_maximum_kts", {}).get(split, []), local_index)
                bias_kind = None
                bias_value = None
                if "sensor_bias_maximums" in parameters:
                    names = sorted(tier_schedule["sensor_bias_maximums"])
                    bias_kind = names[local_index % len(names)]
                    bias_values = tier_schedule["sensor_bias_maximums"][bias_kind][split]
                    bias_value = _cycle(bias_values, local_index // len(names))
                specs.append({
                    "registry_index": len(specs), "split": split, "seed": seed,
                    "tier": tier, "family": family, "severity_band": severity_band[tier],
                    "scenario_instance_id": f"{split}:tier{tier}:{family}:{seed}",
                    "duration_seconds": 26.0, "initial_grid_fraction": fraction,
                    "initial_state": {
                        "airspeed_kts": ranged("initial_airspeed_kts"),
                        "roll_deg": ranged("initial_bank_deg"),
                        "pitch_deg": ranged("initial_pitch_deg"),
                        "altitude_ft": ranged("altitude_ft"),
                        "yaw_deg": 0.0, "heading_deg": 250.0,
                        "latitude_deg": 37.0, "longitude_deg": -120.0,
                    },
                    "terrain_clearance_ft": ranged("terrain_clearance_ft"),
                    "component_severity": float(component) if "component_severity" in parameters else None,
                    "activation_seconds": float(activation) if "activation_seconds" in parameters else None,
                    "actuator_delay_seconds": float(delay) if "actuator_delay_seconds" in parameters else None,
                    "sensor_bias_kind": bias_kind,
                    "sensor_bias_value": float(bias_value) if bias_value is not None else None,
                    "turbulence_intensity": float(turbulence),
                    "crosswind_maximum_kts": float(crosswind) if crosswind is not None else None,
                    "engine_side": None,
                    "parameter_semantics": semantics,
                    "gaussian_sensor_noise_enabled": False,
                })
    if any(seed_index[split] != len(seeds[split]) for split in SPLITS):
        raise ValueError(f"seed allocation mismatch: {seed_index}")
    engine_specs = [spec for spec in specs if spec["family"] in ENGINE_FAMILIES and spec["family"] != "right_engine_rudder_crosswind"]
    forced_right = [spec for spec in specs if spec["family"] == "right_engine_rudder_crosswind"]
    left_target = 152
    right_other_target = 152 - len(forced_right)
    for index, spec in enumerate(engine_specs):
        before = index * left_target // len(engine_specs)
        after = (index + 1) * left_target // len(engine_specs)
        spec["engine_side"] = "left" if after > before else "right"
    for spec in forced_right:
        spec["engine_side"] = "right"
    sides = Counter(spec["engine_side"] for spec in specs if spec["engine_side"])
    if sides != {"left": left_target, "right": right_other_target + len(forced_right)}:
        raise ValueError(f"engine side allocation mismatch: {sides}")
    _assert_registry_counts(specs, design)
    for spec in specs:
        spec["canonical_signature"] = _signature(spec)
        spec["flight_spec_hash"] = object_hash(spec)
    return specs


def _assert_registry_counts(specs: list[dict[str, Any]], design: dict[str, Any]) -> None:
    if len(specs) != 1152:
        raise ValueError(f"flight count mismatch: {len(specs)}")
    if Counter(spec["split"] for spec in specs) != Counter({key: int(value) for key, value in design["split_counts"].items()}):
        raise ValueError("split allocation mismatch")
    if Counter(spec["tier"] for spec in specs) != Counter({tier: 192 for tier in range(6)}):
        raise ValueError("tier allocation mismatch")
    expected = {row["family"]: int(row["count"]) for row in design["scenario_matrix"]}
    if Counter(spec["family"] for spec in specs) != Counter(expected):
        raise ValueError("family allocation mismatch")
    all_seeds = [spec["seed"] for spec in specs]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("duplicate seed in final flight registry")


def _signature(spec: dict[str, Any]) -> str:
    value = [
        spec["tier"], spec["family"], spec["component_severity"],
        spec["activation_seconds"], spec["actuator_delay_seconds"],
        [spec["sensor_bias_kind"], spec["sensor_bias_value"]],
        spec["turbulence_intensity"], spec["crosswind_maximum_kts"],
        spec["initial_grid_fraction"], spec["engine_side"],
    ]
    return object_hash(value)


def _failure(
    failure_id: str, failure_type: str, subsystem: str, severity: float,
    activation: float, *, parameters: dict[str, Any] | None = None,
    observable: bool = True, detected: bool = True,
) -> dict[str, Any]:
    return {
        "failure_id": failure_id, "failure_type": failure_type,
        "affected_subsystem": subsystem, "severity": severity,
        "activation": "time", "activation_time_seconds": activation,
        "permanent": True, "ramp_duration_seconds": 0.0,
        "observable": observable, "detected": detected,
        "parameters": parameters or {},
    }


def _failures(spec: dict[str, Any]) -> list[dict[str, Any]]:
    family = spec["family"]
    activation = spec["activation_seconds"]
    severity = spec["component_severity"]
    rows: list[dict[str, Any]] = []
    def engine() -> None:
        side = spec["engine_side"]
        rows.append(_failure(f"{side}_engine", f"{side}_engine_failure", f"{side}_engine", 1.0, activation))
    def aileron() -> None:
        rows.append(_failure("aileron_loss", "reduced_aileron_effectiveness", "aileron", severity, activation))
    def partial() -> None:
        rows.append(_failure("partial_thrust", "partial_thrust_loss", "engines", severity, activation, parameters={"engine": "both"}))
    def delay() -> None:
        rows.append(_failure("actuator_delay", "actuator_delay", "flight_controls", 1.0, activation, parameters={"maximum_delay_seconds": spec["actuator_delay_seconds"]}))
    def sensor() -> None:
        kind = spec["sensor_bias_kind"].replace("_kts", "").replace("_ft", "").replace("_deg", "") + "_bias"
        subsystem = spec["sensor_bias_kind"].split("_")[0] + "_sensor"
        rows.append(_failure("sensor_bias", kind, subsystem, 1.0, activation, parameters={"maximum_bias": spec["sensor_bias_value"]}, observable=False, detected=False))
    def turbulence() -> None:
        rows.append(_failure("turbulence", "turbulence_increase", "environment", spec["turbulence_intensity"], activation))
    if family == "nominal":
        return rows
    if family in {"isolated_aileron_mild", "isolated_aileron_moderate", "boundary_aileron"}:
        aileron()
    elif family == "isolated_partial_thrust_mild":
        partial()
    elif family in {"isolated_engine_loss_mild_state", "isolated_engine_loss_moderate_state"}:
        engine()
    elif family == "deterministic_sensor_bias":
        sensor()
    elif family == "isolated_actuator_delay":
        delay()
    elif family in {"engine_plus_aileron", "severe_engine_plus_aileron", "boundary_engine_aileron"}:
        engine(); aileron()
    elif family == "delay_plus_partial_thrust":
        delay(); partial()
    elif family == "aileron_plus_turbulence":
        aileron(); turbulence()
    elif family in {"sensor_bias_plus_actuator", "severe_sensor_bias_plus_delay", "boundary_sensor_delay"}:
        sensor(); delay()
    elif family == "severe_delay_turbulence_partial_thrust":
        delay(); turbulence(); partial()
    elif family == "right_engine_rudder_crosswind":
        engine()
        rows.append(_failure("rudder_loss", "reduced_rudder_effectiveness", "rudder", severity, activation))
        rows.append(_failure("crosswind", "crosswind_increase", "environment", 1.0, activation, parameters={"maximum_wind_kts": spec["crosswind_maximum_kts"]}))
    elif family == "boundary_delay_turbulence":
        delay(); turbulence()
    else:
        raise ValueError(f"unsupported frozen family: {family}")
    return rows


def _scenario(root: Path, spec: dict[str, Any]) -> ScenarioConfig:
    payload = load_yaml(root / "configs/scenarios/phase2_nominal.yaml")
    payload.update({
        "scenario_id": f"final_{spec['family']}",
        "description": f"Frozen v2.1 generated family {spec['family']}",
        "seed": spec["seed"], "duration_seconds": spec["duration_seconds"],
        "expected_recoverability": "unknown", "initial_state": spec["initial_state"],
        "failures": _failures(spec),
    })
    payload["environment"] = {
        **payload["environment"], "terrain_clearance_ft": spec["terrain_clearance_ft"],
        "turbulence_intensity": spec["turbulence_intensity"] if spec["family"] == "nominal" else 0.0,
    }
    return ScenarioConfig.model_validate(payload)


def _metric_sample(state: dict[str, float], clearance: float, aircraft: AircraftConfig) -> dict[str, float]:
    return {
        "stall_margin_kts": state["airspeed_kts"] - aircraft.stall_speed_kts,
        "overspeed_margin_kts": aircraft.overspeed_kts - state["airspeed_kts"],
        "bank_deg": state["roll_deg"], "pitch_deg": state["pitch_deg"],
        "vertical_speed_fpm": state["vertical_speed_fpm"],
        "terrain_clearance_ft": clearance,
        "angle_of_attack_deg": max(-20.0, min(40.0, state["pitch_deg"] - state["vertical_speed_fpm"] / 1000.0)),
        "load_factor_g": max(-3.0, min(9.0, 1.0 / max(math.cos(math.radians(state["roll_deg"])), 0.12))),
    }


def _vertical_bin(value: float) -> str:
    bounds = [(-2500.0, -750.0), (-750.0, -150.0), (-150.0, 150.0), (150.0, 750.0), (750.0, 2500.0)]
    names = ["[-2500,-750)", "[-750,-150)", "[-150,150)", "[150,750)", "[750,2500]"]
    for index, (low, high) in enumerate(bounds):
        if low <= value < high or (index == len(bounds) - 1 and value == high):
            return names[index]
    return "outside_frozen_bins"


def _generate_flight(
    root: Path, spec: dict[str, Any], aircraft: AircraftConfig,
    fallback_config: FallbackConfig, metrics: dict[str, Any], git_commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenario = _scenario(root, spec)
    simulator = SimpleAircraftSimulator(aircraft, scenario)
    expert = DeterministicFallbackController(fallback_config, scenario.airports)
    observation = simulator.reset(spec["seed"])
    expert.reset()
    flight_id = f"{DATASET_ID}:{spec['scenario_instance_id']}"
    records: list[dict[str, Any]] = []
    step = 0
    warmup_vs: float | None = None
    while not simulator.is_terminal():
        truth_before = simulator.audit_snapshot()
        if warmup_vs is None and observation.timestamp_seconds >= 2.0 - 1e-9:
            warmup_vs = float(truth_before["true_state"]["vertical_speed_fpm"])
        proposed = expert.act(observation)
        next_observation = simulator.step(proposed)
        truth_after = simulator.audit_snapshot()
        metric_sample = _metric_sample(truth_before["true_state"], observation.environment.terrain_clearance_ft, aircraft)
        violations = safety_violations(metric_sample, metrics["safety"])
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "dataset_id": DATASET_ID,
            "record_id": f"{flight_id}:step:{step:06d}", "flight_id": flight_id,
            "scenario_id": scenario.scenario_id, "scenario_instance_id": spec["scenario_instance_id"],
            "difficulty_tier": spec["tier"], "scenario_family": spec["family"],
            "severity_band": spec["severity_band"], "random_seed": spec["seed"],
            "split": spec["split"], "timestep_index": step,
            "simulation_timestamp_seconds": float(observation.timestamp_seconds),
            "state_after_timestamp_seconds": float(next_observation.timestamp_seconds),
            "physics_timestep_seconds": aircraft.time_step_seconds,
            "aircraft_state_before": truth_before["true_state"],
            "ordered_observation_42": observation_to_vector(observation).astype(float).tolist(),
            "observation_feature_names": list(OBSERVATION_FEATURES),
            "expert_proposed_controls": _controls(proposed), "label_semantics": "expert_proposed_action",
            "monitor_status": None, "monitor_modified_controls": None,
            "executed_controls": [float(truth_after["applied_control_commands"][name]) for name in CONTROL_NAMES],
            "actuator_state": observation.control_health.model_dump(mode="json"),
            "engine_state": observation.engine_health.model_dump(mode="json"),
            "active_failures": truth_before["ground_truth_failures"],
            "controller_known_failures": list(observation.active_failures),
            "ground_truth_failures": truth_before["ground_truth_failures"],
            "ground_truth_failures_after": truth_after["ground_truth_failures"],
            "failure_schedule": [failure.model_dump(mode="json") for failure in scenario.failures],
            "actuator_effectiveness": {
                "elevator": observation.control_health.elevator_effectiveness,
                "aileron": observation.control_health.aileron_effectiveness,
                "rudder": observation.control_health.rudder_effectiveness,
            },
            "actuator_delay_seconds": observation.control_health.actuator_delay_ms / 1000.0,
            "sensor_error_failures": [item for item in truth_before["ground_truth_failures"] if item.get("failure_type") in {"airspeed_bias", "altitude_bias", "heading_bias"}],
            "disturbance_values": observation.environment.model_dump(mode="json"),
            "disturbance_realization_id": f"{spec['scenario_instance_id']}:seed:{spec['seed']}",
            "aircraft_state_after": truth_after["true_state"],
            "termination_status": {"terminal": False, "reason": None},
            "recoverability_label": scenario.expected_recoverability, "outcome_label": None,
            "recovery_envelope_inside": inside_recovery_envelope(metric_sample, metrics["recovery"]),
            "safety_envelope_inside": not any(violations.values()), "safety_violations": violations,
            "source_configuration_path": "configs/scenarios/phase2_nominal.yaml",
            "source_configuration_hash": file_hash(root / "configs/scenarios/phase2_nominal.yaml"),
            "effective_scenario_configuration_hash": object_hash(scenario.model_dump(mode="json")),
            "simulator_configuration_hash": file_hash(root / "configs/aircraft/simple_twin.yaml"),
            "expert_configuration_hash": file_hash(root / "configs/research/final_fallback_v1.yaml"),
            "git_commit_hash": git_commit, "generator_version": "final-candidate-v2.1-batched",
            "simulator_version": "simple-aircraft-simulator-v1", "declared_synthetic_corruption": None,
            "generation_freeze_hash": ACTIVE_FREEZE_HASH, "flight_spec_hash": spec["flight_spec_hash"],
            "canonical_scenario_signature": spec["canonical_signature"],
            "gaussian_sensor_noise_enabled": False,
        }
        record["record_hash"] = record_hash(record)
        records.append(record)
        observation = next_observation
        step += 1
    reason = simulator.termination_reason()
    outcome = _outcome(reason, False)
    records[-1]["termination_status"] = {"terminal": True, "reason": reason}
    warmup_vs = 0.0 if warmup_vs is None else warmup_vs
    vertical_bin = _vertical_bin(warmup_vs)
    for record in records:
        record["outcome_label"] = outcome
        record["warmup_duration_seconds"] = 2.0
        record["measured_prefailure_vertical_speed_fpm"] = warmup_vs
        record["warmup_vertical_speed_bin"] = vertical_bin
        record["record_hash"] = record_hash(record)
    flight = {
        "dataset_id": DATASET_ID, "flight_id": flight_id,
        "scenario_id": scenario.scenario_id, "scenario_instance_id": spec["scenario_instance_id"],
        "difficulty_tier": spec["tier"], "scenario_family": spec["family"],
        "severity_band": spec["severity_band"], "random_seed": spec["seed"],
        "split": spec["split"], "transition_count": len(records),
        "recoverability_label": scenario.expected_recoverability, "outcome_label": outcome,
        "termination_reason": reason, "source_configuration_path": "configs/scenarios/phase2_nominal.yaml",
        "source_configuration_hash": records[0]["source_configuration_hash"],
        "flight_spec_hash": spec["flight_spec_hash"], "canonical_scenario_signature": spec["canonical_signature"],
        "component_severity": spec["component_severity"], "activation_seconds": spec["activation_seconds"],
        "actuator_delay_seconds": spec["actuator_delay_seconds"], "sensor_bias_kind": spec["sensor_bias_kind"],
        "sensor_bias_value": spec["sensor_bias_value"], "turbulence_intensity": spec["turbulence_intensity"],
        "crosswind_maximum_kts": spec["crosswind_maximum_kts"], "engine_side": spec["engine_side"],
        "initial_grid_fraction": spec["initial_grid_fraction"], "initial_state": spec["initial_state"],
        "terrain_clearance_ft": spec["terrain_clearance_ft"],
        "measured_prefailure_vertical_speed_fpm": warmup_vs, "warmup_vertical_speed_bin": vertical_bin,
    }
    flight["flight_hash"] = object_hash(flight)
    return records, flight


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "workflow_version": "final-candidate-v2.1-batched",
        "canonical_feature_order": list(OBSERVATION_FEATURES), "control_order": list(CONTROL_NAMES),
        "feature_count": 42, "control_count": 5,
    }


def _write_batch(
    root: Path, batch_dir: Path, specs: list[dict[str, Any]], batch_index: int,
    aircraft: AircraftConfig, fallback: FallbackConfig, metrics: dict[str, Any], git_commit: str,
) -> dict[str, Any]:
    if batch_dir.exists():
        manifest_path = batch_dir / "manifests/batch_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"incomplete existing batch: {batch_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["flight_spec_hashes"] != [spec["flight_spec_hash"] for spec in specs]:
            raise ValueError(f"batch resume spec mismatch: {batch_index}")
        report = validate_candidate_dataset(batch_dir, suppress_outcomes_for_splits={"final_sealed_test"})
        if report["record_status_counts"]["FAIL"] or report["record_status_counts"]["REVIEW_REQUIRED"]:
            raise ValueError(f"existing batch validation failed: {batch_index}")
        return manifest
    for name in ("raw", "manifests", "validation", "review"):
        (batch_dir / name).mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    flights: list[dict[str, Any]] = []
    started = _utc()
    for spec in specs:
        flight_records, flight = _generate_flight(root, spec, aircraft, fallback, metrics, git_commit)
        records.extend(flight_records)
        flights.append(flight)
    _write_jsonl(batch_dir / "raw/transitions.jsonl", records)
    _write_jsonl(batch_dir / "raw/flights.jsonl", flights)
    write_json(batch_dir / "manifests/schema.json", _schema())
    split_map = {split: [row["flight_id"] for row in flights if row["split"] == split] for split in SPLITS}
    generation_manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_id": DATASET_ID,
        "record_count": len(records), "flight_count": len(flights), "splits": split_map,
        "batch_index": batch_index, "sealed_outcome_summary_prohibited": True,
    }
    write_json(batch_dir / "manifests/generation_manifest.json", generation_manifest)
    paths = [batch_dir / "raw/transitions.jsonl", batch_dir / "raw/flights.jsonl", batch_dir / "manifests/schema.json", batch_dir / "manifests/generation_manifest.json"]
    write_json(batch_dir / "manifests/file_checksums.json", {path.relative_to(batch_dir).as_posix(): file_hash(path) for path in paths})
    report = validate_candidate_dataset(batch_dir, suppress_outcomes_for_splits={"final_sealed_test"})
    if report["record_status_counts"]["FAIL"] or report["record_status_counts"]["REVIEW_REQUIRED"]:
        raise ValueError(f"batch {batch_index} validation non-PASS: {report['record_status_counts']}")
    manifest = {
        "batch_index": batch_index, "status": "PASS", "started_utc": started,
        "completed_utc": _utc(), "flight_count": len(flights), "record_count": len(records),
        "flight_spec_hashes": [spec["flight_spec_hash"] for spec in specs],
        "record_status_counts": report["record_status_counts"],
        "validator_report_hash": report["validator_report_hash"],
        "file_hashes": {path.relative_to(batch_dir).as_posix(): file_hash(path) for path in paths},
    }
    write_json(batch_dir / "manifests/batch_manifest.json", manifest)
    return manifest


def _consolidate(root: Path, dataset: Path, specs: list[dict[str, Any]], batch_dirs: list[Path], started: str) -> dict[str, Any]:
    raw = dataset / "raw"; manifests = dataset / "manifests"
    raw.mkdir(exist_ok=False); manifests.mkdir(exist_ok=False)
    (dataset / "validation").mkdir(exist_ok=False); (dataset / "review").mkdir(exist_ok=False)
    transitions = raw / "transitions.jsonl"; flights_path = raw / "flights.jsonl"
    with transitions.open("wb") as output:
        for batch in batch_dirs:
            with (batch / "raw/transitions.jsonl").open("rb") as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
        output.flush()
    with flights_path.open("wb") as output:
        for batch in batch_dirs:
            with (batch / "raw/flights.jsonl").open("rb") as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
        output.flush()
    flights = [json.loads(line) for line in flights_path.read_text(encoding="utf-8").splitlines() if line]
    record_count = sum(int(row["transition_count"]) for row in flights)
    write_json(manifests / "schema.json", _schema())
    generation_manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_id": DATASET_ID,
        "generator_version": "final-candidate-v2.1-batched", "git_commit_hash": _git_hash(),
        "generation_freeze_id": "final_generation_freeze_v2_1",
        "generation_freeze_hash": ACTIVE_FREEZE_HASH,
        "record_count": record_count, "flight_count": len(flights),
        "splits": {split: [row["flight_id"] for row in flights if row["split"] == split] for split in SPLITS},
        "feature_contract": {"version": "phase1-42-v1", "count": 42, "order": list(OBSERVATION_FEATURES)},
        "action_contract": {"count": 5, "order": list(CONTROL_NAMES), "label": "expert_proposed_action"},
        "scenario_inventory": [{"scenario_instance_id": row["scenario_instance_id"], "scenario_id": row["scenario_id"], "tier": row["difficulty_tier"], "family": row["scenario_family"], "seed": row["random_seed"], "split": row["split"]} for row in flights],
        "seed_inventory": sorted(row["random_seed"] for row in flights),
        "tier_flight_counts": dict(Counter(str(row["difficulty_tier"]) for row in flights)),
        "family_flight_counts": dict(Counter(row["scenario_family"] for row in flights)),
        "raw_records_unfiltered": True, "gaussian_sensor_noise_enabled": False,
        "sealed_outcome_analysis_authorized": False,
        "generation_started_utc": started, "generation_completed_utc": _utc(),
        "batch_count": len(batch_dirs), "batch_size": 48,
        "flight_registry_hash": object_hash(specs),
    }
    write_json(manifests / "generation_manifest.json", generation_manifest)
    paths = [transitions, flights_path, manifests / "schema.json", manifests / "generation_manifest.json"]
    write_json(manifests / "file_checksums.json", {path.relative_to(dataset).as_posix(): file_hash(path) for path in paths})
    return generation_manifest


def generate_final_candidate_dataset(root: Path, batch_size: int = 48) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    dataset = root / "datasets/candidate_policy_data" / DATASET_ID
    if (dataset / "manifests/candidate_dataset_manifest.json").exists():
        raise FileExistsError(f"completed candidate is immutable: {dataset}")
    dataset.mkdir(parents=True, exist_ok=True)
    started_path = dataset / "generation_started.json"
    if started_path.exists():
        started = json.loads(started_path.read_text(encoding="utf-8"))["generation_started_utc"]
    else:
        started = _utc(); write_json(started_path, {"generation_started_utc": started, "freeze_hash": ACTIVE_FREEZE_HASH})
    specs = build_final_flight_registry(root)
    registry_path = dataset / "frozen_flight_registry.json"
    if registry_path.exists():
        if json.loads(registry_path.read_text(encoding="utf-8")) != specs:
            raise ValueError("immutable flight registry changed during resume")
    else:
        write_json(registry_path, specs)
    aircraft = load_config(root / "configs/aircraft/simple_twin.yaml", AircraftConfig)
    fallback = load_config(root / "configs/research/final_fallback_v1.yaml", FallbackConfig)
    metrics = load_yaml(root / "configs/research/final_metrics_v1.yaml")
    git_commit = _git_hash()
    batches_root = dataset / ".batches"; batches_root.mkdir(exist_ok=True)
    batch_dirs: list[Path] = []
    manifests: list[dict[str, Any]] = []
    for start in range(0, len(specs), batch_size):
        index = start // batch_size
        batch_dir = batches_root / f"batch_{index:03d}"
        batch_specs = specs[start:start + batch_size]
        manifest = _write_batch(root, batch_dir, batch_specs, index, aircraft, fallback, metrics, git_commit)
        batch_dirs.append(batch_dir); manifests.append(manifest)
        print(json.dumps({"batch": index, "batches_total": math.ceil(len(specs) / batch_size), "flight_count": manifest["flight_count"], "record_count": manifest["record_count"], "status": manifest["status"]}), flush=True)
    if not (dataset / "raw").exists():
        generation_manifest = _consolidate(root, dataset, specs, batch_dirs, started)
    else:
        generation_manifest = json.loads((dataset / "manifests/generation_manifest.json").read_text(encoding="utf-8"))
    return {"dataset": str(dataset), "generation_manifest": generation_manifest, "batch_manifests": manifests}
