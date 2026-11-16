"""Frozen v2.1A optimizer selection for the selected five-control network.

Optimization can read only TRAIN. Checkpoint and hyperparameter selection can
read only DEVELOPMENT. The challenge, OOD-stress, and sealed splits have no
access surface in this stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.common import file_hash, object_hash
from aircraft_recovery.data_quality.final_approved import APPROVED_ID
from aircraft_recovery.training.v2_1a_architecture_screen import (
    AMENDMENT_LOGICAL_HASH,
    APPROVED_DATASET_LOGICAL_HASH,
    ARCHITECTURE_CANDIDATES,
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
    ArchitectureCandidate,
    V21AControlPolicyNetwork,
    _checkpoint,
    _development_metrics,
    _environment,
    _git_revision,
    _normalize,
    _read_split,
    _save_checkpoint,
    _set_seed,
    _write_json,
    architecture_screen_preflight,
    assert_access_policy,
)


ARCHITECTURE_SCREEN_RUNTIME_HASH = "281699cb38c4c9594d8e24a356573a2df5e5e404a1a1011e5a06fbc3d4898726"
ARCHITECTURE_SCREEN_PROVENANCE_HASH = "6836e0e1111479e6aea165065f8442985172ff1b81817674c1212937de298a4d"
OPTIMIZER_SELECTION_SEED = 17001
WINNING_ARCHITECTURE_ID = "moderate"
WINNING_ARCHITECTURE = next(
    item for item in ARCHITECTURE_CANDIDATES if item.architecture_id == WINNING_ARCHITECTURE_ID
)
SELECTION_CONTROLS: dict[str, Any] = {
    "optimizer": "AdamW",
    "objective": "mean_squared_error",
    "checkpoint_selection": "minimum_development_mse",
    "weight_decay": 0.0001,
    "maximum_epochs": 50,
    "patience": 8,
    "improvement_tolerance": 1e-10,
    "gradient_norm_clip": 1.0,
}


@dataclass(frozen=True)
class OptimizerGridRun:
    run_id: str
    learning_rate: float
    batch_size: int


OPTIMIZER_GRID_RUNS = (
    OptimizerGridRun("adamw_lr_0p0003_bs_256", 0.0003, 256),
    OptimizerGridRun("adamw_lr_0p0003_bs_512", 0.0003, 512),
    OptimizerGridRun("adamw_lr_0p001_bs_256", 0.001, 256),
    OptimizerGridRun("adamw_lr_0p001_bs_512", 0.001, 512),
)


def _expected_config() -> dict[str, Any]:
    return {
        "protocol_id": "final_optimizer_selection_v2_1a",
        "dataset_id": APPROVED_ID,
        "dataset_logical_sha256": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "amendment_logical_sha256": AMENDMENT_LOGICAL_HASH,
        "architecture_screen_runtime_sha256": ARCHITECTURE_SCREEN_RUNTIME_HASH,
        "architecture_screen_provenance_sha256": ARCHITECTURE_SCREEN_PROVENANCE_HASH,
        "architecture": {
            "architecture_id": WINNING_ARCHITECTURE_ID,
            "input_dimension": INPUT_DIMENSION,
            "hidden_dimensions": list(WINNING_ARCHITECTURE.hidden_dimensions),
            "activation": WINNING_ARCHITECTURE.activation,
            "output_dimension": OUTPUT_DIMENSION,
            "parameter_count": WINNING_ARCHITECTURE.expected_parameter_count,
            "target_order": list(TARGET_ORDER),
        },
        "fit_split": "train",
        "selection_split": "development",
        "forbidden_splits": ["challenge", "ood_stress", "final_sealed_test"],
        "optimizer_selection_seed": OPTIMIZER_SELECTION_SEED,
        "optimizer": "AdamW",
        "objective": "mean_squared_error",
        "checkpoint_selection": "minimum_development_mse",
        "weight_decay": 0.0001,
        "learning_rates": [0.0003, 0.001],
        "batch_sizes": [256, 512],
        "maximum_epochs": 50,
        "patience": 8,
        "improvement_tolerance": 1e-10,
        "gradient_norm_clip": 1.0,
    }


def optimizer_selection_preflight(
    root: str | Path, *, verify_large_file: bool = True
) -> dict[str, Any]:
    root = Path(root).resolve()
    base = architecture_screen_preflight(root, verify_large_file=verify_large_file)
    assert_access_policy()
    config_path = root / "configs/training/final_optimizer_selection_v2_1a.yaml"
    runtime_path = root / "results/final_architecture_screen_v2_1a/architecture_screen_summary.json"
    provenance_path = root / "artifacts/research_stage/final_architecture_screen_v2_1a.json"
    config = load_yaml(config_path)
    expected = _expected_config()
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen optimizer-selection config mismatch: {key}")
    if file_hash(runtime_path) != ARCHITECTURE_SCREEN_RUNTIME_HASH:
        raise ValueError("architecture-screen runtime hash mismatch")
    if file_hash(provenance_path) != ARCHITECTURE_SCREEN_PROVENANCE_HASH:
        raise ValueError("architecture-screen provenance hash mismatch")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("status") != "COMPLETE" or runtime.get("winner_architecture_id") != WINNING_ARCHITECTURE_ID:
        raise ValueError("architecture-screen winner identity mismatch")
    winner = next(
        row for row in runtime["architectures"] if row["architecture_id"] == WINNING_ARCHITECTURE_ID
    )
    if (
        winner["hidden_dimensions"] != list(WINNING_ARCHITECTURE.hidden_dimensions)
        or winner["activation"] != WINNING_ARCHITECTURE.activation
        or winner["parameter_count"] != WINNING_ARCHITECTURE.expected_parameter_count
    ):
        raise ValueError("architecture-screen winning contract mismatch")
    model = V21AControlPolicyNetwork(WINNING_ARCHITECTURE)
    if model.parameter_count != 110341 or model.output_dimension != 5:
        raise ValueError("selected architecture implementation mismatch")
    return {
        "status": "PASS",
        "protocol_id": "final_optimizer_selection_v2_1a",
        "architecture": expected["architecture"],
        "optimizer": "AdamW",
        "learning_rates": expected["learning_rates"],
        "batch_sizes": expected["batch_sizes"],
        "grid_run_count": len(OPTIMIZER_GRID_RUNS),
        "maximum_epochs": SELECTION_CONTROLS["maximum_epochs"],
        "patience": SELECTION_CONTROLS["patience"],
        "gradient_norm_clip": SELECTION_CONTROLS["gradient_norm_clip"],
        "development_checkpoint_selection_metric": "mean_squared_error",
        "optimizer_selection_seed": OPTIMIZER_SELECTION_SEED,
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
        "architecture_screen_runtime_sha256": ARCHITECTURE_SCREEN_RUNTIME_HASH,
        "architecture_screen_provenance_sha256": ARCHITECTURE_SCREEN_PROVENANCE_HASH,
        "optimizer_selection_configuration_sha256": file_hash(config_path),
        "base_preflight": base,
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
    }


def _train_grid_run(
    *,
    root: Path,
    grid_run: OptimizerGridRun,
    train_observations: torch.Tensor,
    train_targets: torch.Tensor,
    development_observations: torch.Tensor,
    development_targets: torch.Tensor,
    normalization: dict[str, Any],
    preflight: dict[str, Any],
    source_hash: str,
    config_file_hash: str,
) -> dict[str, Any]:
    _set_seed(OPTIMIZER_SELECTION_SEED)
    device = train_observations.device
    candidate: ArchitectureCandidate = WINNING_ARCHITECTURE
    model = V21AControlPolicyNetwork(candidate).to(device)
    if model.parameter_count != 110341 or model.output_dimension != 5:
        raise ValueError("selected architecture implementation mismatch")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=grid_run.learning_rate,
        weight_decay=SELECTION_CONTROLS["weight_decay"],
    )
    checkpoint_dir = (
        root / "checkpoints/final_policy_v1/moderate/optimizer_grid"
        / grid_run.run_id / f"seed_{OPTIMIZER_SELECTION_SEED}"
    )
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite optimizer-grid output: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training = dict(SELECTION_CONTROLS)
    training.update({"learning_rate": grid_run.learning_rate, "batch_size": grid_run.batch_size})
    run_config = {
        "protocol_id": "final_optimizer_selection_v2_1a",
        "run_id": grid_run.run_id,
        "architecture_id": candidate.architecture_id,
        "hidden_dimensions": list(candidate.hidden_dimensions),
        "activation": candidate.activation,
        "parameter_count": model.parameter_count,
        "input_dimension": INPUT_DIMENSION,
        "output_dimension": OUTPUT_DIMENSION,
        "target_order": list(TARGET_ORDER),
        "seed": OPTIMIZER_SELECTION_SEED,
        "fit_split": "train",
        "selection_split": "development",
        "forbidden_splits": sorted(FORBIDDEN_SPLITS),
        "training": training,
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
        "architecture_screen_runtime_sha256": ARCHITECTURE_SCREEN_RUNTIME_HASH,
        "architecture_screen_provenance_sha256": ARCHITECTURE_SCREEN_PROVENANCE_HASH,
        "source_file_sha256": source_hash,
        "optimizer_selection_configuration_sha256": config_file_hash,
        "source_git_revision": _git_revision(root),
    }
    run_config["configuration_logical_hash"] = object_hash(run_config)
    run_config_path = checkpoint_dir / "run_configuration.json"
    _write_json(run_config_path, run_config)
    environment = _environment(device)
    history: list[dict[str, Any]] = []
    best_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    generator = torch.Generator(device=device.type).manual_seed(OPTIMIZER_SELECTION_SEED)
    started = datetime.now(timezone.utc)
    started_timer = perf_counter()

    for epoch in range(1, int(SELECTION_CONTROLS["maximum_epochs"]) + 1):
        model.train()
        permutation = torch.randperm(
            train_observations.shape[0], generator=generator, device=device
        )
        sum_squared_error = 0.0
        element_count = 0
        for start in range(0, permutation.shape[0], grid_run.batch_size):
            indices = permutation[start:start + grid_run.batch_size]
            observations = train_observations.index_select(0, indices)
            targets = train_targets.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(observations)
            loss = torch.mean(torch.square(predictions - targets))
            loss.backward()
            clip_grad_norm_(model.parameters(), float(SELECTION_CONTROLS["gradient_norm_clip"]))
            optimizer.step()
            sum_squared_error += float(loss.detach().cpu()) * targets.numel()
            element_count += targets.numel()
        development = _development_metrics(
            model, development_observations, development_targets, grid_run.batch_size
        )
        improved = development["mse"] < best_mse - float(SELECTION_CONTROLS["improvement_tolerance"])
        history.append({
            "epoch": epoch,
            "train_mse": sum_squared_error / element_count,
            "development": development,
            "checkpoint_improved": improved,
        })
        checkpoint_value = _checkpoint(
            model=model,
            optimizer=optimizer,
            candidate=candidate,
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
        if epochs_without_improvement >= int(SELECTION_CONTROLS["patience"]):
            break

    _save_checkpoint(checkpoint_dir / "last.pt", checkpoint_value)
    best_path = checkpoint_dir / "best.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    verified_metrics = _development_metrics(
        model, development_observations, development_targets, grid_run.batch_size
    )
    if abs(verified_metrics["mse"] - best_mse) > 1e-12:
        raise ValueError("best checkpoint development MSE failed reload verification")
    history_path = checkpoint_dir / "training_history.json"
    _write_json(history_path, history)
    ended = datetime.now(timezone.utc)
    summary = {
        "run_id": grid_run.run_id,
        "optimizer": "AdamW",
        "learning_rate": grid_run.learning_rate,
        "batch_size": grid_run.batch_size,
        "architecture_id": candidate.architecture_id,
        "hidden_dimensions": list(candidate.hidden_dimensions),
        "activation": candidate.activation,
        "parameter_count": model.parameter_count,
        "seed": OPTIMIZER_SELECTION_SEED,
        "epochs_completed": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "development_selection_loss_name": "mean_squared_error",
        "development_selection_loss": best_mse,
        "development_metrics": verified_metrics,
        "checkpoint_path": str(best_path.relative_to(root)).replace("\\", "/"),
        "checkpoint_sha256": file_hash(best_path),
        "last_checkpoint_sha256": file_hash(checkpoint_dir / "last.pt"),
        "training_history_path": str(history_path.relative_to(root)).replace("\\", "/"),
        "training_history_sha256": file_hash(history_path),
        "run_configuration_path": str(run_config_path.relative_to(root)).replace("\\", "/"),
        "run_configuration_logical_hash": run_config["configuration_logical_hash"],
        "run_configuration_file_sha256": file_hash(run_config_path),
        "optimizer_selection_configuration_sha256": config_file_hash,
        "source_file_sha256": source_hash,
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
        "architecture_screen_runtime_sha256": ARCHITECTURE_SCREEN_RUNTIME_HASH,
        "training_start_utc": started.isoformat(),
        "training_end_utc": ended.isoformat(),
        "duration_seconds": perf_counter() - started_timer,
        "environment": environment,
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
        "preflight_status": preflight["status"],
    }
    _write_json(checkpoint_dir / "training_summary.json", summary)
    return summary


def run_optimizer_selection(
    root: str | Path, *, verify_large_file: bool = True
) -> dict[str, Any]:
    root = Path(root).resolve()
    results_dir = root / "results/final_optimizer_selection_v2_1a"
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite optimizer-selection results: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    preflight = optimizer_selection_preflight(root, verify_large_file=verify_large_file)
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
    source_path = root / "src/aircraft_recovery/training/v2_1a_optimizer_selection.py"
    config_path = root / "configs/training/final_optimizer_selection_v2_1a.yaml"
    source_hash = file_hash(source_path)
    config_file_hash = file_hash(config_path)

    summaries = [
        _train_grid_run(
            root=root,
            grid_run=grid_run,
            train_observations=train_observations,
            train_targets=train_targets,
            development_observations=development_observations,
            development_targets=development_targets,
            normalization=normalization,
            preflight=preflight,
            source_hash=source_hash,
            config_file_hash=config_file_hash,
        )
        for grid_run in OPTIMIZER_GRID_RUNS
    ]
    winner = min(summaries, key=lambda row: row["development_selection_loss"])
    result = {
        "result_version": "final-optimizer-selection-v2.1a-v1",
        "status": "COMPLETE",
        "selection_rule": "minimum_development_mse",
        "winner_run_id": winner["run_id"],
        "winner_checkpoint_path": winner["checkpoint_path"],
        "architecture": _expected_config()["architecture"],
        "runs": summaries,
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
        "architecture_screen_runtime_sha256": ARCHITECTURE_SCREEN_RUNTIME_HASH,
        "architecture_screen_provenance_sha256": ARCHITECTURE_SCREEN_PROVENANCE_HASH,
        "optimizer_selection_configuration_sha256": config_file_hash,
        "source_file_sha256": source_hash,
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
    }
    _write_json(results_dir / "optimizer_selection_summary.json", result)
    return result
