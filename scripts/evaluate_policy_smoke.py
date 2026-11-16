"""Evaluate a smoke checkpoint without opening the sealed held-out test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import DeterministicFallbackController, FallbackConfig, NeuralBaselineController
from aircraft_recovery.data.dataset import DemonstrationDataset
from aircraft_recovery.data.demonstrations import DemonstrationConfig
from aircraft_recovery.evaluation.policy_evaluator import CONTROL_NAMES, _closed_loop_metrics, evaluate_offline
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.models import (
    ControlCommands,
    ControllerInput,
    ControllerOutput,
    NavigationCommands,
    SafetyDecision,
    StabilizationTargets,
)
from aircraft_recovery.simulator import SimpleAircraftSimulator


class ConstantController:
    """Simple neutral-surface, half-throttle comparison baseline."""

    name = "constant_neutral_half_throttle"

    def reset(self) -> None:
        return None

    def act(self, observation: ControllerInput) -> ControllerOutput:
        state = observation.aircraft_state
        return ControllerOutput(
            timestamp_seconds=observation.timestamp_seconds,
            control_commands=ControlCommands(
                elevator=0.0, aileron=0.0, rudder=0.0,
                throttle_left=0.5, throttle_right=0.5,
            ),
            stabilization_targets=StabilizationTargets(
                target_pitch_deg=state.pitch_deg,
                target_roll_deg=state.roll_deg,
                target_airspeed_kts=state.airspeed_kts,
                target_altitude_ft=state.altitude_ft,
            ),
            navigation_commands=NavigationCommands(target_heading_deg=state.heading_deg),
            emergency_mode="normal",
            safety_decision=SafetyDecision(
                status="not_checked", ai_action_accepted=True, fallback_activated=False,
                explanation="Constant comparison baseline; no safety monitor invoked.",
            ),
            confidence=0.0,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _control_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    error = predictions - targets
    saturated = np.any(np.abs(predictions[:, :3]) >= 0.99, axis=1) | np.any(
        (predictions[:, 3:] <= 0.01) | (predictions[:, 3:] >= 0.99), axis=1
    )
    return {
        "control_mae": float(np.mean(np.abs(error))),
        "control_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "per_control": {
            name: {
                "mae": float(np.mean(np.abs(error[:, index]))),
                "rmse": float(np.sqrt(np.mean(np.square(error[:, index])))),
            }
            for index, name in enumerate(CONTROL_NAMES)
        },
        "command_saturation_frequency": float(np.mean(saturated)),
    }


def _unique_directory(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while path.with_name(f"{path.name}_run{index:03d}").exists():
        index += 1
    return path.with_name(f"{path.name}_run{index:03d}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/training/policy_smoke_demonstrations.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/policy_smoke_v1/best.pt")
    parser.add_argument("--existing-checkpoint", default="checkpoints/phase3_quick/best.pt")
    parser.add_argument("--output", default="results/policy_smoke_v1")
    args = parser.parse_args()

    generation = load_config(args.dataset_config, DemonstrationConfig)
    dataset_directory = Path(generation.output_directory)
    manifest = json.loads((dataset_directory / "split_manifest.json").read_text(encoding="utf-8"))
    validation_ids = set(manifest["validation"])
    validation = DemonstrationDataset(dataset_directory, validation_ids)
    checkpoint = Path(args.checkpoint)
    existing_checkpoint = Path(args.existing_checkpoint)
    offline = {
        "smoke_checkpoint": evaluate_offline(
            str(checkpoint), str(dataset_directory), validation_ids, "cpu", 0.1,
        ),
        "existing_checkpoint": evaluate_offline(
            str(existing_checkpoint), str(dataset_directory), validation_ids, "cpu", 0.1,
        ),
        "expert_labels": _control_metrics(validation.controls.copy(), validation.controls),
        "constant_baseline": _control_metrics(
            np.tile(np.asarray([0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float32), (len(validation), 1)),
            validation.controls,
        ),
    }

    source_by_category = {item.category: item.scenario for item in generation.scenarios}
    aircraft = load_config(generation.aircraft_config, AircraftConfig)
    fallback_config = load_config(generation.fallback_config, FallbackConfig)
    output = Path(args.output)
    closed_loop: list[dict[str, object]] = []
    for episode_id in sorted(validation_ids):
        category = manifest["episode_categories"][episode_id]
        seed = int(manifest["episode_seeds"][episode_id])
        base_scenario = load_config(source_by_category[category], ScenarioConfig)
        scenario = ScenarioConfig.model_validate({**base_scenario.model_dump(), "seed": seed})
        item: dict[str, object] = {"flight_id": episode_id, "category": category, "seed": seed}
        controllers = {
            "expert": DeterministicFallbackController(fallback_config, scenario.airports),
            "smoke_checkpoint": NeuralBaselineController(str(checkpoint), "cpu"),
            "existing_checkpoint": NeuralBaselineController(str(existing_checkpoint), "cpu"),
            "constant_baseline": ConstantController(),
        }
        for name, controller in controllers.items():
            run_directory = _unique_directory(output / "closed_loop" / f"{episode_id}_{name}")
            started = perf_counter()
            result = ScenarioRunner(SimpleAircraftSimulator(aircraft, scenario), controller).run(
                seed, run_directory, scenario.scenario_id,
            )
            metrics = _closed_loop_metrics(result.run_directory)
            metrics["evaluation_wall_seconds"] = perf_counter() - started
            item[name] = metrics
        closed_loop.append(item)

    aggregate = {
        name: {
            "recovery_success_rate": float(np.mean([row[name]["recovery_success"] for row in closed_loop])),
            "safety_envelope_violation_rate": float(np.mean([
                row[name]["safety_envelope_violation"] for row in closed_loop
            ])),
            "inference_latency_ms_mean": float(np.mean([
                row[name]["inference_latency_ms_mean"] for row in closed_loop
            ])),
        }
        for name in ("expert", "smoke_checkpoint", "existing_checkpoint", "constant_baseline")
    }
    archive = np.load(dataset_directory / "transitions.npz", allow_pickle=False)
    history = json.loads((checkpoint.parent / "training_history.json").read_text(encoding="utf-8"))
    result = {
        "evaluation_policy": "Validation only; final test flights remained sealed and were not used for selection.",
        "dataset": {
            "transitions": int(len(archive["observations"])),
            "episodes": len(set(archive["episode_ids"].astype(str))),
            "split_flight_ids": {name: manifest[name] for name in ("train", "validation", "test")},
            "split_seeds": {
                name: [manifest["episode_seeds"][episode] for episode in manifest[name]]
                for name in ("train", "validation", "test")
            },
            "test_reporting_groups": manifest["test_reporting_groups"],
            "test_status": "sealed_not_evaluated",
            "input_dimension": 42,
            "target_dimension": 5,
            "feature_minimums": archive["observations"].min(axis=0).astype(float).tolist(),
            "feature_maximums": archive["observations"].max(axis=0).astype(float).tolist(),
            "target_minimums": archive["controls"].min(axis=0).astype(float).tolist(),
            "target_maximums": archive["controls"].max(axis=0).astype(float).tolist(),
        },
        "training_history": history,
        "offline_validation": offline,
        "closed_loop_validation": closed_loop,
        "closed_loop_aggregate": aggregate,
        "checkpoint_sha256": _sha256(checkpoint),
        "existing_checkpoint_sha256": _sha256(existing_checkpoint),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

