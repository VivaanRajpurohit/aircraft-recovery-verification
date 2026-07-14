"""Controlled ReLU-versus-SiLU training and evaluation experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aircraft_recovery.config import load_config
from aircraft_recovery.data.hashing import canonical_hash
from aircraft_recovery.evaluation.policy_evaluator import EvaluationConfig, evaluate_policy
from aircraft_recovery.training.trainer import TrainingConfig, train_policy


class ActivationComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    comparison_name: str
    base_training_config: str
    evaluation_config: str
    output_directory: str
    activations: list[Literal["relu", "silu"]] = Field(default_factory=lambda: ["relu", "silu"], min_length=2)


def run_activation_comparison(
    config: ActivationComparisonConfig,
    device_override: str | None = None,
) -> dict[str, object]:
    """Train/evaluate activations while holding all material settings constant."""
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    base_training = load_config(config.base_training_config, TrainingConfig)
    base_evaluation = load_config(config.evaluation_config, EvaluationConfig)
    results: dict[str, dict[str, object]] = {}
    controlled_signatures: list[str] = []
    for activation in config.activations:
        training = base_training.model_copy(deep=True)
        training.experiment_name = f"{config.comparison_name}_{activation}"
        training.output_directory = str(output / "checkpoints" / activation)
        training.architecture.activation = activation
        if device_override:
            training.device = device_override
        signature = training.model_dump(mode="json")
        for field in ("experiment_name", "output_directory"):
            signature.pop(field)
        signature["architecture"] = dict(signature["architecture"])
        signature["architecture"].pop("activation")
        controlled_signatures.append(canonical_hash(signature))
        training_result = train_policy(training)

        evaluation = base_evaluation.model_copy(deep=True)
        evaluation.evaluation_name = f"{config.comparison_name}_{activation}"
        evaluation.checkpoint = training_result["best_checkpoint"]
        evaluation.output_directory = str(output / "evaluation" / activation)
        if device_override:
            evaluation.device = device_override
        evaluation_result = evaluate_policy(evaluation)
        offline = evaluation_result["offline"].get("test", {})
        closed = evaluation_result["closed_loop_aggregate"]["neural"]
        results[activation] = {
            "parameter_count": training_result["parameter_count"],
            "best_epoch": training_result["best_epoch"],
            "final_training_loss": training_result["final_train_losses"]["total"],
            "final_validation_loss": training_result["final_validation_losses"]["total"],
            "offline_control_mae": offline.get("control_mae"),
            "offline_control_rmse": offline.get("control_rmse"),
            "emergency_mode_accuracy": offline.get("mode_accuracy"),
            "closed_loop_stabilization_rate": closed["stabilization_success_rate"],
            "closed_loop_recovery_rate": closed["recovery_success_rate"],
            "closed_loop_inference_latency_ms": closed["inference_latency_ms_mean"],
            "checkpoint": training_result["best_checkpoint"],
        }
    controlled = len(set(controlled_signatures)) == 1
    if not controlled:
        raise RuntimeError("Activation comparison settings differ beyond activation")
    best = min(results, key=lambda name: float(results[name]["final_validation_loss"]))
    report: dict[str, object] = {
        "comparison_name": config.comparison_name,
        "controlled_settings_identical": controlled,
        "results": results,
        "lowest_validation_loss_activation": best,
        "interpretation": (
            "The lowest validation loss identifies only this controlled run's result; "
            "activation choice does not establish safety."
        ),
    }
    (output / "activation_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

