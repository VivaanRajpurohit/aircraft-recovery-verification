"""Offline metrics and own-state closed-loop expert/neural comparisons."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import (
    DeterministicFallbackController,
    FallbackConfig,
    NeuralBaselineController,
)
from aircraft_recovery.data.dataset import DemonstrationDataset, MODE_NAMES
from aircraft_recovery.data.preprocessing import ObservationPreprocessor, PreprocessingStatistics
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.training.checkpointing import load_checkpoint
from aircraft_recovery.training.models import ImitationPolicyNetwork, PolicyArchitectureConfig
from aircraft_recovery.training.trainer import resolve_device


class EvaluationScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    category: str
    seed_offsets: list[int] = Field(default_factory=lambda: [0])


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluation_name: str
    dataset_directory: str
    split_manifest: str
    checkpoint: str
    output_directory: str
    aircraft_config: str
    fallback_config: str
    device: str = "auto"
    offline_splits: list[str] = Field(default_factory=lambda: ["test", "ood"])
    disagreement_threshold: float = Field(default=0.1, gt=0.0)
    scenarios: list[EvaluationScenario]


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(MODE_NAMES):
        true_positive = int(np.sum((labels == index) & (predictions == index)))
        false_positive = int(np.sum((labels != index) & (predictions == index)))
        false_negative = int(np.sum((labels == index) & (predictions != index)))
        result[name] = {
            "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
            "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
            "support": int(np.sum(labels == index)),
        }
    return result


def evaluate_offline(
    checkpoint_path: str,
    dataset_directory: str,
    episode_ids: set[str],
    device_name: str,
    disagreement_threshold: float,
) -> dict[str, Any]:
    """Evaluate imitation against held-out expert labels only."""
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = ImitationPolicyNetwork(
        PolicyArchitectureConfig.model_validate(checkpoint["architecture_configuration"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocessor = ObservationPreprocessor(
        PreprocessingStatistics.from_dict(checkpoint["normalization_statistics"])
    )
    dataset = DemonstrationDataset(dataset_directory, episode_ids)
    observations = preprocessor.transform(dataset.observations)
    predictions: list[np.ndarray] = []
    modes: list[np.ndarray] = []
    latencies_ms: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch_size = 1024
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            tensor = torch.from_numpy(observations[start:start + batch_size]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            output = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = (perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed / max(1, len(tensor)))
            predictions.append(output["controls"].cpu().numpy())
            modes.append(torch.argmax(output["mode_logits"], dim=-1).cpu().numpy())
    predicted_controls = np.concatenate(predictions)
    predicted_modes = np.concatenate(modes)
    error = predicted_controls - dataset.controls
    disagreement = (
        np.max(np.abs(error), axis=1) > disagreement_threshold
    ) | (predicted_modes != dataset.modes)
    smoothness_values = []
    for index in range(1, len(predicted_controls)):
        if dataset.episode_ids[index] == dataset.episode_ids[index - 1]:
            smoothness_values.append(float(np.mean(np.abs(predicted_controls[index] - predicted_controls[index - 1]))))
    saturated = np.any(np.abs(predicted_controls[:, :3]) >= 0.99, axis=1) | np.any(
        (predicted_controls[:, 3:] <= 0.01) | (predicted_controls[:, 3:] >= 0.99), axis=1
    )
    return {
        "transitions": len(dataset),
        "episodes": len(set(dataset.episode_ids.tolist())),
        "control_mae": float(np.mean(np.abs(error))),
        "control_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mode_accuracy": float(np.mean(predicted_modes == dataset.modes)),
        "per_mode": _classification_metrics(dataset.modes, predicted_modes),
        "expert_neural_disagreement_rate": float(np.mean(disagreement)),
        "command_smoothness_mean_absolute_delta": float(np.mean(smoothness_values)) if smoothness_values else 0.0,
        "command_saturation_frequency": float(np.mean(saturated)),
        "inference_latency_ms_per_transition_mean": float(np.mean(latencies_ms)),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }


def _closed_loop_metrics(run_directory: Path) -> dict[str, Any]:
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_directory / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    commands = np.asarray([[row["final_action"]["control_commands"][name] for name in (
        "elevator", "aileron", "rudder", "throttle_left", "throttle_right"
    )] for row in rows], dtype=np.float32)
    saturation = np.any(np.abs(commands[:, :3]) >= 0.99, axis=1) | np.any(
        (commands[:, 3:] <= 0.01) | (commands[:, 3:] >= 0.99), axis=1
    )
    smoothness = np.abs(np.diff(commands, axis=0)).mean() if len(commands) > 1 else 0.0
    selected = [row["final_action"]["navigation_commands"]["selected_airport"] for row in rows]
    return {
        "termination_reason": summary["termination_reason"],
        "recovery_success": summary["recovery_result"] == "successful",
        "stabilization_success": summary["minimum_stall_margin_kts"] >= 0.0 and summary["maximum_absolute_roll_deg"] <= 60.0,
        "diversion_selected": any(value is not None for value in selected),
        "successful_diversion": summary["termination_reason"] in {"diversion_target_reached", "approach_completed"},
        "minimum_stall_margin_kts": summary["minimum_stall_margin_kts"],
        "maximum_absolute_roll_deg": summary["maximum_absolute_roll_deg"],
        "maximum_angle_of_attack_deg": summary["maximum_angle_of_attack_deg"],
        "minimum_terrain_clearance_ft": summary["minimum_terrain_clearance_ft"],
        "inference_latency_ms_mean": summary["controller_latency_ms_mean"],
        "command_smoothness_mean_absolute_delta": float(smoothness),
        "command_saturation_frequency": float(np.mean(saturation)),
    }


def _unique_run_directory(base: Path) -> Path:
    """Preserve prior evaluations by adding a deterministic numeric suffix."""
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = base.with_name(f"{base.name}_run{index:03d}")
        if not candidate.exists():
            return candidate
        index += 1


def evaluate_policy(config: EvaluationConfig) -> dict[str, Any]:
    """Run offline and paired own-state closed-loop evaluation."""
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(config.split_manifest).read_text(encoding="utf-8"))
    offline: dict[str, Any] = {}
    for split in config.offline_splits:
        if manifest.get(split):
            offline[split] = evaluate_offline(
                config.checkpoint, config.dataset_directory, set(manifest[split]),
                config.device, config.disagreement_threshold,
            )
    aircraft = load_config(config.aircraft_config, AircraftConfig)
    fallback_config = load_config(config.fallback_config, FallbackConfig)
    paired_runs: list[dict[str, Any]] = []
    for evaluation_scenario in config.scenarios:
        base_scenario = load_config(evaluation_scenario.scenario, ScenarioConfig)
        for seed_offset in evaluation_scenario.seed_offsets:
            scenario = ScenarioConfig.model_validate({
                **base_scenario.model_dump(), "seed": base_scenario.seed + seed_offset
            })
            pair: dict[str, Any] = {
                "scenario_id": scenario.scenario_id,
                "category": evaluation_scenario.category,
                "seed": scenario.seed,
            }
            for controller_name in ("expert", "neural"):
                simulator = SimpleAircraftSimulator(aircraft, scenario)
                controller = (
                    DeterministicFallbackController(fallback_config, scenario.airports)
                    if controller_name == "expert" else
                    NeuralBaselineController(config.checkpoint, config.device)
                )
                run_directory = _unique_run_directory(
                    output / "closed_loop" / f"{scenario.scenario_id}_{scenario.seed}_{controller_name}"
                )
                result = ScenarioRunner(simulator, controller).run(
                    scenario.seed, run_directory, scenario.scenario_id
                )
                pair[controller_name] = _closed_loop_metrics(result.run_directory)
            paired_runs.append(pair)
    summary = {
        "evaluation_name": config.evaluation_name,
        "offline": offline,
        "paired_closed_loop": paired_runs,
        "paired_seed_match": all(item["expert"] and item["neural"] for item in paired_runs),
        "termination_reason_distributions": {
            name: dict(Counter(item[name]["termination_reason"] for item in paired_runs))
            for name in ("expert", "neural")
        },
        "closed_loop_aggregate": {
            name: {
                "recovery_success_rate": float(np.mean([item[name]["recovery_success"] for item in paired_runs])),
                "stabilization_success_rate": float(np.mean([item[name]["stabilization_success"] for item in paired_runs])),
                "diversion_selection_rate": float(np.mean([item[name]["diversion_selected"] for item in paired_runs])),
                "successful_diversion_rate": float(np.mean([item[name]["successful_diversion"] for item in paired_runs])),
                "inference_latency_ms_mean": float(np.mean([item[name]["inference_latency_ms_mean"] for item in paired_runs])),
            } for name in ("expert", "neural")
        },
        "warning": "Offline imitation error does not establish safe or stable closed-loop behavior.",
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
