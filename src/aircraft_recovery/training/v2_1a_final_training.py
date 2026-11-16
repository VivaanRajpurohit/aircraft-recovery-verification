"""Frozen v2.1A five-seed final training.

Only TRAIN enters optimization and only DEVELOPMENT enters the frozen
checkpoint/early-stopping rule. This stage exposes no Challenge, OOD-stress,
or final-sealed-test record access surface.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.common import file_hash, object_hash
from aircraft_recovery.data_quality.final_approved import APPROVED_ID
from aircraft_recovery.training.v2_1a_architecture_screen import (
    AMENDMENT_LOGICAL_HASH,
    APPROVED_DATASET_LOGICAL_HASH,
    DEVELOPMENT_COUNT,
    DEVELOPMENT_FLIGHT_COUNT,
    FORBIDDEN_SPLITS,
    INPUT_DIMENSION,
    NORMALIZATION_LOGICAL_HASH,
    OUTPUT_DIMENSION,
    TARGET_ORDER,
    TRAIN_COUNT,
    TRAIN_FLIGHT_COUNT,
    ArchitectureScreenDataSource,
    V21AControlPolicyNetwork,
    _checkpoint,
    _environment,
    _git_revision,
    _normalize,
    _read_split,
    _save_checkpoint,
    _set_seed,
    _write_json,
    assert_access_policy,
)
from aircraft_recovery.training.v2_1a_optimizer_selection import (
    WINNING_ARCHITECTURE,
    optimizer_selection_preflight,
)


FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH = (
    "288b96bbeed75d4ce19793e28198be25430f8dd260604fb654276c405467917a"
)
FINAL_TRAINING_SEEDS = (17001, 17002, 17003, 17004, 17005)
FINAL_TRAINING_CONTROLS: dict[str, Any] = {
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "batch_size": 512,
    "weight_decay": 0.0001,
    "objective": "mean_squared_error",
    "maximum_epochs": 50,
    "early_stopping_patience": 8,
    "improvement_tolerance": 1e-10,
    "gradient_norm_clip": 1.0,
    "checkpoint_selection": "minimum_development_mse",
}
SATURATION_THRESHOLDS = {
    "surface_absolute_threshold": 0.99,
    "throttle_low_threshold": 0.01,
    "throttle_high_threshold": 0.99,
}
POST_TRAINING_MODEL_RULE = "absent_stop_before_challenge"


def _canonical_hash(document: dict[str, Any], field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return object_hash(payload)


def _expected_stage_config() -> dict[str, Any]:
    return {
        "protocol_id": "final_five_seed_training_v2_1a",
        "final_training_configuration_logical_sha256": FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH,
        "dataset_id": APPROVED_ID,
        "dataset_logical_sha256": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "amendment_logical_sha256": AMENDMENT_LOGICAL_HASH,
        "architecture": {
            "architecture_id": "moderate",
            "input_dimension": INPUT_DIMENSION,
            "hidden_dimensions": [256, 256, 128],
            "activation": "silu",
            "output_dimension": OUTPUT_DIMENSION,
            "parameter_count": 110341,
            "target_order": list(TARGET_ORDER),
        },
        "training": FINAL_TRAINING_CONTROLS,
        "seeds": list(FINAL_TRAINING_SEEDS),
        "fit_split": "train",
        "selection_split": "development",
        "forbidden_splits": ["challenge", "ood_stress", "final_sealed_test"],
        "saturation": SATURATION_THRESHOLDS,
        "summary_standard_deviation": "sample_ddof_1",
        "post_training_representative_model_rule": POST_TRAINING_MODEL_RULE,
    }


def five_seed_training_preflight(
    root: str | Path, *, verify_large_file: bool = True
) -> dict[str, Any]:
    root = Path(root).resolve()
    base = optimizer_selection_preflight(root, verify_large_file=verify_large_file)
    assert_access_policy()
    freeze_path = root / "artifacts/research_stage/final_training_configuration_v2_1a.json"
    stage_path = root / "configs/training/final_five_seed_training_v2_1a.yaml"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    declared = freeze.get("freeze_logical_sha256")
    if declared != FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH:
        raise ValueError("final-training configuration identity mismatch")
    if _canonical_hash(freeze, "freeze_logical_sha256") != declared:
        raise ValueError("final-training configuration canonical hash mismatch")
    expected_freeze = _expected_stage_config()
    if freeze.get("architecture") != expected_freeze["architecture"]:
        raise ValueError("frozen final architecture mismatch")
    if freeze.get("optimization") != FINAL_TRAINING_CONTROLS:
        raise ValueError("frozen final optimization controls mismatch")
    if tuple(freeze.get("final_training_seeds", ())) != FINAL_TRAINING_SEEDS:
        raise ValueError("frozen final seed registry mismatch")
    stage = load_yaml(stage_path)
    for key, value in expected_freeze.items():
        if stage.get(key) != value:
            raise ValueError(f"five-seed stage config mismatch: {key}")
    model = V21AControlPolicyNetwork(WINNING_ARCHITECTURE)
    if model.parameter_count != 110341 or model.output_dimension != 5:
        raise ValueError("final architecture implementation mismatch")
    return {
        "status": "PASS",
        "protocol_id": "final_five_seed_training_v2_1a",
        "architecture": expected_freeze["architecture"],
        "training": FINAL_TRAINING_CONTROLS,
        "seeds": list(FINAL_TRAINING_SEEDS),
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
        "final_training_configuration_logical_hash": declared,
        "final_training_configuration_file_sha256": file_hash(freeze_path),
        "stage_configuration_file_sha256": file_hash(stage_path),
        "post_training_representative_model_rule": POST_TRAINING_MODEL_RULE,
        "base_preflight": base,
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
    }


def _development_metrics_with_saturation(
    model: V21AControlPolicyNetwork,
    observations: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    device = observations.device
    absolute_error = torch.zeros(OUTPUT_DIMENSION, dtype=torch.float64, device=device)
    squared_error = torch.zeros(OUTPUT_DIMENSION, dtype=torch.float64, device=device)
    saturation_count = torch.zeros(OUTPUT_DIMENSION, dtype=torch.int64, device=device)
    any_saturation_count = 0
    count = 0
    with torch.inference_mode():
        for start in range(0, observations.shape[0], batch_size):
            prediction = model(observations[start:start + batch_size])
            error = prediction - targets[start:start + batch_size]
            absolute_error += torch.abs(error).double().sum(dim=0)
            squared_error += torch.square(error).double().sum(dim=0)
            saturated = torch.cat(
                (
                    torch.abs(prediction[:, :3]) >= SATURATION_THRESHOLDS["surface_absolute_threshold"],
                    (prediction[:, 3:] <= SATURATION_THRESHOLDS["throttle_low_threshold"])
                    | (prediction[:, 3:] >= SATURATION_THRESHOLDS["throttle_high_threshold"]),
                ),
                dim=1,
            )
            saturation_count += saturated.sum(dim=0)
            any_saturation_count += int(saturated.any(dim=1).sum().cpu())
            count += error.shape[0]
    per_mae = (absolute_error / count).cpu().numpy()
    per_rmse = torch.sqrt(squared_error / count).cpu().numpy()
    mse = float((squared_error.sum() / (count * OUTPUT_DIMENSION)).cpu())
    return {
        "mse": mse,
        "mae": float(per_mae.mean()),
        "rmse": float(np.sqrt(mse)),
        "per_output": {
            name: {"mae": float(per_mae[index]), "rmse": float(per_rmse[index])}
            for index, name in enumerate(TARGET_ORDER)
        },
        "saturation": {
            "thresholds": SATURATION_THRESHOLDS,
            "any_output_count": any_saturation_count,
            "any_output_frequency": float(any_saturation_count / count),
            "per_output": {
                name: {
                    "count": int(saturation_count[index].cpu()),
                    "frequency": float(saturation_count[index].cpu() / count),
                }
                for index, name in enumerate(TARGET_ORDER)
            },
        },
        "transition_count": count,
    }


def _train_seed(
    *,
    root: Path,
    seed: int,
    train_observations: torch.Tensor,
    train_targets: torch.Tensor,
    development_observations: torch.Tensor,
    development_targets: torch.Tensor,
    normalization: dict[str, Any],
    preflight: dict[str, Any],
    source_hash: str,
    stage_config_hash: str,
) -> dict[str, Any]:
    _set_seed(seed)
    device = train_observations.device
    model = V21AControlPolicyNetwork(WINNING_ARCHITECTURE).to(device)
    if model.parameter_count != 110341 or model.output_dimension != 5:
        raise ValueError("final architecture implementation mismatch")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINAL_TRAINING_CONTROLS["learning_rate"],
        weight_decay=FINAL_TRAINING_CONTROLS["weight_decay"],
    )
    checkpoint_dir = (
        root / "checkpoints/final_policy_v1/moderate/final_training_v2_1a"
        / f"seed_{seed}"
    )
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite final-training output: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "protocol_id": "final_five_seed_training_v2_1a",
        "seed": seed,
        "architecture": _expected_stage_config()["architecture"],
        "training": FINAL_TRAINING_CONTROLS,
        "fit_split": "train",
        "selection_split": "development",
        "forbidden_splits": ["challenge", "ood_stress", "final_sealed_test"],
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
        "final_training_configuration_logical_hash": FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH,
        "final_training_configuration_file_sha256": preflight["final_training_configuration_file_sha256"],
        "source_file_sha256": source_hash,
        "stage_configuration_file_sha256": stage_config_hash,
        "source_git_revision": _git_revision(root),
        "post_training_representative_model_rule": POST_TRAINING_MODEL_RULE,
    }
    run_config["configuration_logical_hash"] = object_hash(run_config)
    run_config_path = checkpoint_dir / "run_configuration.json"
    _write_json(run_config_path, run_config)
    environment = _environment(device)
    history: list[dict[str, Any]] = []
    best_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    batch_size = int(FINAL_TRAINING_CONTROLS["batch_size"])
    generator = torch.Generator(device=device.type).manual_seed(seed)
    started = datetime.now(timezone.utc)
    started_timer = perf_counter()

    for epoch in range(1, int(FINAL_TRAINING_CONTROLS["maximum_epochs"]) + 1):
        model.train()
        permutation = torch.randperm(
            train_observations.shape[0], generator=generator, device=device
        )
        sum_squared_error = 0.0
        element_count = 0
        for start in range(0, permutation.shape[0], batch_size):
            indices = permutation[start:start + batch_size]
            observations = train_observations.index_select(0, indices)
            targets = train_targets.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(observations)
            loss = torch.mean(torch.square(predictions - targets))
            loss.backward()
            clip_grad_norm_(
                model.parameters(), float(FINAL_TRAINING_CONTROLS["gradient_norm_clip"])
            )
            optimizer.step()
            sum_squared_error += float(loss.detach().cpu()) * targets.numel()
            element_count += targets.numel()
        development = _development_metrics_with_saturation(
            model, development_observations, development_targets, batch_size
        )
        improved = development["mse"] < best_mse - float(
            FINAL_TRAINING_CONTROLS["improvement_tolerance"]
        )
        history.append({
            "epoch": epoch,
            "train_mse": sum_squared_error / element_count,
            "development": development,
            "checkpoint_improved": improved,
        })
        checkpoint_value = _checkpoint(
            model=model,
            optimizer=optimizer,
            candidate=WINNING_ARCHITECTURE,
            epoch=epoch,
            metrics=development,
            run_config=run_config,
            normalization=normalization,
            environment=environment,
        )
        if improved:
            best_mse = development["mse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(checkpoint_dir / "best.pt", checkpoint_value)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(
            FINAL_TRAINING_CONTROLS["early_stopping_patience"]
        ):
            break

    _save_checkpoint(checkpoint_dir / "last.pt", checkpoint_value)
    best_path = checkpoint_dir / "best.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    verified_metrics = _development_metrics_with_saturation(
        model, development_observations, development_targets, batch_size
    )
    if abs(verified_metrics["mse"] - best_mse) > 1e-12:
        raise ValueError("best checkpoint development MSE failed reload verification")
    history_path = checkpoint_dir / "training_history.json"
    _write_json(history_path, history)
    ended = datetime.now(timezone.utc)
    summary = {
        "seed": seed,
        "epochs_completed": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "development_selection_loss_name": "mean_squared_error",
        "development_selection_loss": best_mse,
        "development_metrics": verified_metrics,
        "checkpoint_path": str(best_path.relative_to(root)).replace("\\", "/"),
        "checkpoint_sha256": file_hash(best_path),
        "last_checkpoint_path": str((checkpoint_dir / "last.pt").relative_to(root)).replace("\\", "/"),
        "last_checkpoint_sha256": file_hash(checkpoint_dir / "last.pt"),
        "training_history_path": str(history_path.relative_to(root)).replace("\\", "/"),
        "training_history_sha256": file_hash(history_path),
        "run_configuration_path": str(run_config_path.relative_to(root)).replace("\\", "/"),
        "run_configuration_logical_hash": run_config["configuration_logical_hash"],
        "run_configuration_file_sha256": file_hash(run_config_path),
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
        "final_training_configuration_logical_hash": FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH,
        "final_training_configuration_file_sha256": preflight["final_training_configuration_file_sha256"],
        "stage_configuration_file_sha256": stage_config_hash,
        "source_file_sha256": source_hash,
        "source_git_revision": run_config["source_git_revision"],
        "training_start_utc": started.isoformat(),
        "training_end_utc": ended.isoformat(),
        "duration_seconds": perf_counter() - started_timer,
        "environment": environment,
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
    }
    _write_json(checkpoint_dir / "training_summary.json", summary)
    return summary


def descriptive_summary(values: Iterable[float | int]) -> dict[str, float]:
    numeric = [float(value) for value in values]
    if len(numeric) < 2:
        raise ValueError("five-seed descriptive summary requires at least two values")
    return {
        "mean": mean(numeric),
        "sample_standard_deviation": stdev(numeric),
        "median": median(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def _across_seed_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        metric: descriptive_summary(run["development_metrics"][metric] for run in runs)
        for metric in ("mse", "mae", "rmse")
    }
    result["per_output"] = {
        name: {
            metric: descriptive_summary(
                run["development_metrics"]["per_output"][name][metric] for run in runs
            )
            for metric in ("mae", "rmse")
        }
        for name in TARGET_ORDER
    }
    result["epochs_completed"] = descriptive_summary(run["epochs_completed"] for run in runs)
    result["runtime_seconds"] = descriptive_summary(run["duration_seconds"] for run in runs)
    return result


def run_five_seed_training(
    root: str | Path, *, verify_large_file: bool = True
) -> dict[str, Any]:
    root = Path(root).resolve()
    results_dir = root / "results/final_five_seed_training_v2_1a"
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite five-seed results: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    preflight = five_seed_training_preflight(root, verify_large_file=verify_large_file)
    _write_json(results_dir / "preflight.json", preflight)

    approved = root / "datasets/approved_policy_data" / APPROVED_ID
    normalization_path = approved / "preprocessing/train_only_normalization.json"
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    source = ArchitectureScreenDataSource(approved)
    train = _read_split(
        source.training_records(), expected_split="train",
        expected_count=TRAIN_COUNT, expected_flights=TRAIN_FLIGHT_COUNT,
    )
    development = _read_split(
        source.development_records(), expected_split="development",
        expected_count=DEVELOPMENT_COUNT, expected_flights=DEVELOPMENT_FLIGHT_COUNT,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_observations = torch.from_numpy(_normalize(train.observations, normalization)).to(device)
    train_targets = torch.from_numpy(train.controls).to(device)
    development_observations = torch.from_numpy(
        _normalize(development.observations, normalization)
    ).to(device)
    development_targets = torch.from_numpy(development.controls).to(device)
    source_path = root / "src/aircraft_recovery/training/v2_1a_final_training.py"
    stage_path = root / "configs/training/final_five_seed_training_v2_1a.yaml"
    source_hash = file_hash(source_path)
    stage_hash = file_hash(stage_path)

    runs = [
        _train_seed(
            root=root,
            seed=seed,
            train_observations=train_observations,
            train_targets=train_targets,
            development_observations=development_observations,
            development_targets=development_targets,
            normalization=normalization,
            preflight=preflight,
            source_hash=source_hash,
            stage_config_hash=stage_hash,
        )
        for seed in FINAL_TRAINING_SEEDS
    ]
    result = {
        "result_version": "final-five-seed-training-v2.1a-v1",
        "status": "COMPLETE",
        "seeds": list(FINAL_TRAINING_SEEDS),
        "individual_seed_runs": runs,
        "across_seed_development_summary": _across_seed_summary(runs),
        "standard_deviation_definition": "sample_ddof_1",
        "representative_model_rule_status": POST_TRAINING_MODEL_RULE,
        "representative_model_selected": False,
        "next_stage_status": "blocked_pending_prospective_representative_model_rule",
        "data": {
            "train_flights": train.flight_count,
            "train_transitions": train.transition_count,
            "train_record_id_order_sha256": train.record_id_order_sha256,
            "development_flights": development.flight_count,
            "development_transitions": development.transition_count,
            "development_record_id_order_sha256": development.record_id_order_sha256,
        },
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "normalization_file_sha256": file_hash(normalization_path),
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
        "final_training_configuration_logical_hash": FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH,
        "final_training_configuration_file_sha256": preflight["final_training_configuration_file_sha256"],
        "stage_configuration_file_sha256": stage_hash,
        "source_file_sha256": source_hash,
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
    }
    _write_json(results_dir / "five_seed_training_summary.json", result)
    return result


def verify_five_seed_checkpoints(root: str | Path) -> dict[str, Any]:
    """Independently reload every best checkpoint and reproduce DEVELOPMENT metrics."""
    root = Path(root).resolve()
    summary_path = root / "results/final_five_seed_training_v2_1a/five_seed_training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    approved = root / "datasets/approved_policy_data" / APPROVED_ID
    normalization = json.loads(
        (approved / "preprocessing/train_only_normalization.json").read_text(encoding="utf-8")
    )
    source = ArchitectureScreenDataSource(approved)
    development = _read_split(
        source.development_records(), expected_split="development",
        expected_count=DEVELOPMENT_COUNT, expected_flights=DEVELOPMENT_FLIGHT_COUNT,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observations = torch.from_numpy(_normalize(development.observations, normalization)).to(device)
    targets = torch.from_numpy(development.controls).to(device)
    verified = []
    for run in summary["individual_seed_runs"]:
        path = root / run["checkpoint_path"]
        if file_hash(path) != run["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch for seed {run['seed']}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        architecture = checkpoint["architecture"]
        if (
            architecture["input_dimension"] != 42
            or architecture["output_dimension"] != 5
            or architecture["hidden_dimensions"] != [256, 256, 128]
        ):
            raise ValueError(f"checkpoint architecture mismatch for seed {run['seed']}")
        model = V21AControlPolicyNetwork(WINNING_ARCHITECTURE).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if model.parameter_count != 110341:
            raise ValueError(f"checkpoint parameter mismatch for seed {run['seed']}")
        metrics = _development_metrics_with_saturation(model, observations, targets, 512)
        for name in ("mse", "mae", "rmse"):
            if abs(metrics[name] - run["development_metrics"][name]) > 1e-12:
                raise ValueError(f"development {name} mismatch for seed {run['seed']}")
        verified.append({
            "seed": run["seed"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "input_dimension": 42,
            "output_dimension": 5,
            "parameter_count": 110341,
            "development_metrics_reproduced": True,
        })
    result = {
        "status": "PASS",
        "summary_sha256": file_hash(summary_path),
        "verified_checkpoints": verified,
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
    }
    _write_json(
        root / "results/final_five_seed_training_v2_1a/checkpoint_verification.json",
        result,
    )
    return result
