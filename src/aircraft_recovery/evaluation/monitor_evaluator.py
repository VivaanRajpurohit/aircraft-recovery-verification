"""Three-controller Phase 4 closed-loop monitor evaluation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import (
    DeterministicFallbackController,
    FallbackConfig,
    MonitoredNeuralController,
    NeuralBaselineController,
)
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.safety import SafetyConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator


class MonitorEvaluationScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    category: str
    seed_offsets: list[int] = Field(default_factory=lambda: [0])


class MonitorEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluation_name: str
    checkpoint: str
    safety_config: str
    aircraft_config: str
    fallback_config: str
    output_directory: str
    device: str = "auto"
    scenarios: list[MonitorEvaluationScenario]


def _unique_directory(base: Path) -> Path:
    if not base.exists():
        return base
    index = 2
    while (candidate := base.with_name(f"{base.name}_run{index:03d}")).exists():
        index += 1
    return candidate


def _metrics(run_directory: Path) -> dict[str, Any]:
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_directory / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    decisions = [row.get("monitor_decision") for row in rows if row.get("monitor_decision")]
    statuses = Counter(item["status"] for item in decisions)
    violations = sum(bool(item["violated_properties"]) for item in decisions)
    return {
        "termination_reason": summary["termination_reason"],
        "recovery_success": summary["recovery_result"] == "successful",
        "stabilization_success": summary["minimum_stall_margin_kts"] >= 0.0 and summary["maximum_absolute_roll_deg"] <= 60.0,
        "minimum_stall_margin_kts": summary["minimum_stall_margin_kts"],
        "maximum_angle_of_attack_deg": summary["maximum_angle_of_attack_deg"],
        "maximum_absolute_roll_deg": summary["maximum_absolute_roll_deg"],
        "minimum_terrain_clearance_ft": summary["minimum_terrain_clearance_ft"],
        "accepted_actions": statuses.get("accept", 0),
        "modified_actions": statuses.get("modify", 0),
        "rejected_actions": statuses.get("reject_fallback", 0),
        "unknown_fallback_actions": statuses.get("unknown_fallback", 0),
        "fallback_activations": sum(item["fallback_activated"] for item in decisions),
        "solver_timeouts": sum(item["timeout"] for item in decisions),
        "solver_unknown": sum(item["unknown"] for item in decisions),
        "safety_violation_steps": violations,
        "safety_violation_duration_seconds": violations * 0.1,
        "monitor_latency_ms_mean": float(np.mean([item["monitor_duration_ms"] for item in decisions])) if decisions else 0.0,
        "solver_latency_ms_mean": float(np.mean([item["solver_duration_ms"] for item in decisions])) if decisions else 0.0,
        "total_loop_latency_ms_mean": summary["controller_latency_ms_mean"],
        "steps": summary["steps"],
    }


def evaluate_monitor(config: MonitorEvaluationConfig) -> dict[str, Any]:
    """Evaluate expert, unmonitored neural, and monitored neural on paired seeds."""
    aircraft = load_config(config.aircraft_config, AircraftConfig)
    fallback_config = load_config(config.fallback_config, FallbackConfig)
    safety_config = load_config(config.safety_config, SafetyConfig)
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for entry in config.scenarios:
        base = load_config(entry.scenario, ScenarioConfig)
        for offset in entry.seed_offsets:
            scenario = ScenarioConfig.model_validate({**base.model_dump(), "seed": base.seed + offset})
            record: dict[str, Any] = {
                "scenario_id": scenario.scenario_id,
                "category": entry.category,
                "seed": scenario.seed,
                "expected_recoverability": scenario.expected_recoverability,
            }
            controllers = {
                "expert": DeterministicFallbackController(fallback_config, scenario.airports),
                "neural": NeuralBaselineController(config.checkpoint, config.device),
                "monitored": MonitoredNeuralController(
                    config.checkpoint, config.device, fallback_config,
                    scenario.airports, safety_config,
                ),
            }
            for name, controller in controllers.items():
                directory = _unique_directory(
                    output / "runs" / f"{scenario.scenario_id}_{scenario.seed}_{name}"
                )
                result = ScenarioRunner(
                    SimpleAircraftSimulator(aircraft, scenario), controller
                ).run(scenario.seed, directory, scenario.scenario_id)
                record[name] = _metrics(result.run_directory)
            runs.append(record)
    aggregate = {
        controller: {
            "run_count": len(runs),
            "recovery_success_rate": float(np.mean([run[controller]["recovery_success"] for run in runs])),
            "stabilization_success_rate": float(np.mean([run[controller]["stabilization_success"] for run in runs])),
            "minimum_stall_margin_kts_mean": float(np.mean([run[controller]["minimum_stall_margin_kts"] for run in runs])),
            "maximum_absolute_roll_deg_mean": float(np.mean([run[controller]["maximum_absolute_roll_deg"] for run in runs])),
            "total_loop_latency_ms_mean": float(np.mean([run[controller]["total_loop_latency_ms_mean"] for run in runs])),
        }
        for controller in ("expert", "neural", "monitored")
    }
    monitored = aggregate["monitored"]
    monitored.update({
        "accepted_actions": sum(run["monitored"]["accepted_actions"] for run in runs),
        "modified_actions": sum(run["monitored"]["modified_actions"] for run in runs),
        "rejected_actions": sum(run["monitored"]["rejected_actions"] for run in runs),
        "fallback_activations": sum(run["monitored"]["fallback_activations"] for run in runs),
        "solver_timeouts": sum(run["monitored"]["solver_timeouts"] for run in runs),
        "solver_unknown": sum(run["monitored"]["solver_unknown"] for run in runs),
        "monitor_latency_ms_mean": float(np.mean([run["monitored"]["monitor_latency_ms_mean"] for run in runs])),
        "solver_latency_ms_mean": float(np.mean([run["monitored"]["solver_latency_ms_mean"] for run in runs])),
    })
    report = {
        "evaluation_name": config.evaluation_name,
        "paired_runs": runs,
        "aggregate": aggregate,
        "claim_boundary": "Runtime checks and bounded Z3 properties do not prove recovery or real-world safety.",
    }
    (output / "monitor_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
