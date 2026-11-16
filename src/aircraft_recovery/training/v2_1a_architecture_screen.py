"""Advanced v2.1A five-control architecture screening.

Only the approved train split can enter optimization. Only the approved
development split can enter checkpoint and architecture selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.common import CONTROL_NAMES, file_hash, object_hash
from aircraft_recovery.data_quality.final_approved import (
    APPROVED_ID,
    assert_role_split,
    iter_approved_records,
    validate_approved_manifest_contract,
    validate_normalization_contract,
)


AMENDMENT_ID = "final_training_protocol_amendment_v2_1a"
AMENDMENT_LOGICAL_HASH = "4ecb3942f366ac862990b11b353a06a37b1f0e261004981e9ab5f1359cc92f1f"
APPROVED_DATASET_LOGICAL_HASH = "bc69cbcb5bee7189c091dba51159f8938bdd661b3793007f45c7f9eb2897b43a"
APPROVED_TRANSITIONS_FILE_HASH = "187374b333f99a2df1891dbb06dad5de7a1eadeb4bc9f1ed5282b4094f64d2e1"
NORMALIZATION_LOGICAL_HASH = "005ccca1bfc32214482e89e552761d73dbeefbce651e7a7abee60037ad978cb7"
NORMALIZATION_FILE_HASH = "3fa7538fabe52658ce10ed1606ae8ad0696c85ec5e517e691ba42e696a38f59f"
MONITOR_FILE_HASH = "427d37c1b0c15d8659cab811876bd05b148fb627bba949f86709369f46846281"
FALLBACK_FILE_HASH = "67d08f11c510d5b68e3fc8148589fd8e1a5bf270dc7b9dd637c1d7f1b6562412"
SEED_REGISTRY_FILE_HASH = "98fa364f3168412ccec9017d24a2ed8217f35f40294d30d997d0e30135165319"

INPUT_DIMENSION = 42
OUTPUT_DIMENSION = 5
TARGET_ORDER = tuple(CONTROL_NAMES)
TRAIN_COUNT = 149760
DEVELOPMENT_COUNT = 37440
TRAIN_FLIGHT_COUNT = 576
DEVELOPMENT_FLIGHT_COUNT = 144
ARCHITECTURE_SCREEN_SEED = 17001
FINAL_TRAINING_SEEDS = (17001, 17002, 17003, 17004, 17005)
FORBIDDEN_SPLITS = frozenset({"challenge", "ood_stress", "final_sealed_test"})

SCREEN_CONTROLS: dict[str, Any] = {
    "optimizer": "AdamW",
    "objective": "mean_squared_error",
    "checkpoint_selection": "minimum_development_mse",
    "learning_rate": 0.001,
    "batch_size": 256,
    "weight_decay": 0.0001,
    "maximum_epochs": 50,
    "patience": 8,
    "improvement_tolerance": 1e-10,
    "gradient_norm_clip": 1.0,
}
OPTIMIZER_GRID = {"learning_rates": (0.0003, 0.001), "batch_sizes": (256, 512)}


@dataclass(frozen=True)
class ArchitectureCandidate:
    architecture_id: str
    hidden_dimensions: tuple[int, ...]
    activation: str
    expected_parameter_count: int


ARCHITECTURE_CANDIDATES = (
    ArchitectureCandidate("small_reference", (128, 64), "relu", 14085),
    ArchitectureCandidate("current_reference", (256, 128), "relu", 44549),
    ArchitectureCandidate("moderate", (256, 256, 128), "silu", 110341),
)


class V21AControlPolicyNetwork(nn.Module):
    """Feed-forward policy with exactly five bounded control outputs."""

    def __init__(self, candidate: ArchitectureCandidate) -> None:
        super().__init__()
        activation_type: type[nn.Module]
        if candidate.activation == "relu":
            activation_type = nn.ReLU
        elif candidate.activation == "silu":
            activation_type = nn.SiLU
        else:
            raise ValueError(f"unsupported activation: {candidate.activation}")
        layers: list[nn.Module] = []
        previous = INPUT_DIMENSION
        for width in candidate.hidden_dimensions:
            if width <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.extend((nn.Linear(previous, width), activation_type()))
            previous = width
        self.candidate = candidate
        self.trunk = nn.Sequential(*layers)
        self.control_head = nn.Linear(previous, OUTPUT_DIMENSION)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raw = self.control_head(self.trunk(observations))
        return torch.cat((torch.tanh(raw[..., :3]), torch.sigmoid(raw[..., 3:])), dim=-1)

    @property
    def output_dimension(self) -> int:
        return self.control_head.out_features

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


RecordLoader = Callable[..., Iterable[dict[str, Any]]]


class ArchitectureScreenDataSource:
    """The only record-access surface exposed to v2.1A screening."""

    def __init__(self, dataset: str | Path, loader: RecordLoader = iter_approved_records) -> None:
        self.dataset = Path(dataset)
        self._loader = loader

    def training_records(self) -> Iterator[dict[str, Any]]:
        yield from self._loader(self.dataset, split="train", role="training")

    def development_records(self) -> Iterator[dict[str, Any]]:
        yield from self._loader(self.dataset, split="development", role="model_selection")


@dataclass(frozen=True)
class SplitArrays:
    observations: np.ndarray
    controls: np.ndarray
    transition_count: int
    flight_count: int
    record_id_order_sha256: str


def assert_access_policy() -> None:
    assert_role_split("training", "train")
    assert_role_split("model_selection", "development")
    for role in ("training", "model_selection"):
        for split in FORBIDDEN_SPLITS:
            try:
                assert_role_split(role, split)
            except PermissionError:
                continue
            raise AssertionError(f"{role} unexpectedly gained access to {split}")


def validate_supervised_record(record: dict[str, Any], expected_split: str) -> None:
    if record.get("split") != expected_split:
        raise ValueError(f"unexpected split in {expected_split} loader: {record.get('split')}")
    if tuple(record.get("observation_feature_names", ())) != tuple(OBSERVATION_FEATURES):
        raise ValueError("ordered 42-feature contract mismatch")
    observation = np.asarray(record.get("ordered_observation_42"), dtype=np.float32)
    target = np.asarray(record.get("expert_proposed_controls"), dtype=np.float32)
    if observation.shape != (INPUT_DIMENSION,):
        raise ValueError(f"supervised input dimension must be {INPUT_DIMENSION}")
    if target.shape != (OUTPUT_DIMENSION,):
        raise ValueError(f"supervised target dimension must be {OUTPUT_DIMENSION}")
    if record.get("label_semantics") != "expert_proposed_action":
        raise ValueError("v2.1A target must be the expert proposed action")
    if not np.isfinite(observation).all() or not np.isfinite(target).all():
        raise ValueError("non-finite supervised record")
    if np.any(target[:3] < -1.0) or np.any(target[:3] > 1.0):
        raise ValueError("surface-control target outside [-1, 1]")
    if np.any(target[3:] < 0.0) or np.any(target[3:] > 1.0):
        raise ValueError("throttle target outside [0, 1]")


def _read_split(
    records: Iterable[dict[str, Any]],
    *,
    expected_split: str,
    expected_count: int,
    expected_flights: int,
) -> SplitArrays:
    observations = np.empty((expected_count, INPUT_DIMENSION), dtype=np.float32)
    controls = np.empty((expected_count, OUTPUT_DIMENSION), dtype=np.float32)
    flights: set[str] = set()
    record_ids = hashlib.sha256()
    count = 0
    for count, record in enumerate(records, start=1):
        if count > expected_count:
            raise ValueError(f"{expected_split} exceeds frozen transition count")
        validate_supervised_record(record, expected_split)
        observations[count - 1] = record["ordered_observation_42"]
        controls[count - 1] = record["expert_proposed_controls"]
        flights.add(str(record["flight_id"]))
        record_ids.update(str(record["record_id"]).encode("utf-8") + b"\n")
    if count != expected_count or len(flights) != expected_flights:
        raise ValueError(
            f"{expected_split} count mismatch: {len(flights)} flights/{count} transitions"
        )
    return SplitArrays(observations, controls, count, len(flights), record_ids.hexdigest())


def _verify_amendment(amendment: dict[str, Any]) -> None:
    declared = amendment.get("amendment_logical_hash")
    payload = {key: value for key, value in amendment.items() if key != "amendment_logical_hash"}
    if amendment.get("amendment_id") != AMENDMENT_ID or declared != AMENDMENT_LOGICAL_HASH:
        raise ValueError("v2.1A amendment identity mismatch")
    if object_hash(payload) != AMENDMENT_LOGICAL_HASH:
        raise ValueError("v2.1A amendment canonical hash mismatch")


def _candidate_summary() -> list[dict[str, Any]]:
    result = []
    for candidate in ARCHITECTURE_CANDIDATES:
        model = V21AControlPolicyNetwork(candidate)
        if model.parameter_count != candidate.expected_parameter_count:
            raise ValueError(f"parameter count mismatch for {candidate.architecture_id}")
        result.append({
            "architecture_id": candidate.architecture_id,
            "hidden_dimensions": list(candidate.hidden_dimensions),
            "activation": candidate.activation,
            "parameter_count": model.parameter_count,
        })
    return result


def architecture_screen_preflight(root: str | Path, *, verify_large_file: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    approved = root / "datasets/approved_policy_data" / APPROVED_ID
    manifest_path = approved / "manifests/approved_manifest.json"
    normalization_path = approved / "preprocessing/train_only_normalization.json"
    transitions_path = approved / "raw/transitions.jsonl"
    amendment_path = root / "artifacts/research_stage/final_training_protocol_amendment_v2_1a.json"
    config_path = root / "configs/training/final_architecture_screen_v2_1a.yaml"
    monitor_path = root / "configs/research/final_monitor_v1.yaml"
    fallback_path = root / "configs/research/final_fallback_v1.yaml"
    seed_path = root / "configs/research/final_seed_registry_v1.json"

    assert_access_policy()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    config = load_yaml(config_path)
    validate_approved_manifest_contract(manifest)
    validate_normalization_contract(normalization)
    _verify_amendment(amendment)

    expected_hashes = {
        "approved_dataset_logical_sha256": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "normalization_file_sha256": NORMALIZATION_FILE_HASH,
        "monitor_file_sha256": MONITOR_FILE_HASH,
        "fallback_file_sha256": FALLBACK_FILE_HASH,
        "seed_registry_file_sha256": SEED_REGISTRY_FILE_HASH,
        "amendment_logical_sha256": AMENDMENT_LOGICAL_HASH,
    }
    observed_hashes = {
        "approved_dataset_logical_sha256": manifest["approved_dataset_logical_hash"],
        "normalization_logical_sha256": normalization["normalization_artifact_hash"],
        "normalization_file_sha256": file_hash(normalization_path),
        "monitor_file_sha256": file_hash(monitor_path),
        "fallback_file_sha256": file_hash(fallback_path),
        "seed_registry_file_sha256": file_hash(seed_path),
        "amendment_logical_sha256": amendment["amendment_logical_hash"],
    }
    if verify_large_file:
        expected_hashes["approved_transitions_file_sha256"] = APPROVED_TRANSITIONS_FILE_HASH
        observed_hashes["approved_transitions_file_sha256"] = file_hash(transitions_path)
    mismatches = [name for name, value in expected_hashes.items() if observed_hashes.get(name) != value]
    if mismatches:
        raise ValueError(f"v2.1A frozen hash mismatch: {mismatches}")

    expected_config = {
        "input_dimension": INPUT_DIMENSION,
        "output_dimension": OUTPUT_DIMENSION,
        "target_order": list(TARGET_ORDER),
        "fit_split": "train",
        "selection_split": "development",
        "forbidden_splits": ["challenge", "ood_stress", "final_sealed_test"],
        "architecture_screen_seed": ARCHITECTURE_SCREEN_SEED,
    }
    config_errors = [key for key, value in expected_config.items() if config.get(key) != value]
    if config.get("training") != {
        **SCREEN_CONTROLS,
        "device": "auto",
        "num_workers": 0,
    }:
        config_errors.append("training")
    if config_errors:
        raise ValueError(f"v2.1A architecture-screen config mismatch: {config_errors}")

    source = ArchitectureScreenDataSource(approved)
    first_train = next(source.training_records(), None)
    if first_train is None:
        raise ValueError("approved train split is empty")
    validate_supervised_record(first_train, "train")
    return {
        "report_version": "v2.1a-architecture-screen-preflight-v1",
        "status": "PASS",
        "expected_hashes": expected_hashes,
        "observed_hashes": observed_hashes,
        "configuration_file_sha256": file_hash(config_path),
        "amendment_file_sha256": file_hash(amendment_path),
        "architectures": _candidate_summary(),
        "target_order": list(TARGET_ORDER),
        "optimization_objective": "mean_squared_error",
        "checkpoint_selection": "minimum_development_mse",
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
    }


def _normalize(observations: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    means = np.asarray(normalization["means"], dtype=np.float32)
    standard_deviations = np.asarray(normalization["standard_deviations"], dtype=np.float32)
    if means.shape != (INPUT_DIMENSION,) or standard_deviations.shape != (INPUT_DIMENSION,):
        raise ValueError("normalization dimension mismatch")
    transformed = (observations - means) / standard_deviations
    limit = float(normalization["clip_standard_deviations"])
    return np.clip(transformed, -limit, limit).astype(np.float32, copy=False)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": True,
    }


def _development_metrics(
    model: V21AControlPolicyNetwork,
    observations: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    absolute_error = torch.zeros(OUTPUT_DIMENSION, dtype=torch.float64, device=observations.device)
    squared_error = torch.zeros(OUTPUT_DIMENSION, dtype=torch.float64, device=observations.device)
    count = 0
    with torch.inference_mode():
        for start in range(0, observations.shape[0], batch_size):
            prediction = model(observations[start:start + batch_size])
            error = prediction - targets[start:start + batch_size]
            absolute_error += torch.abs(error).double().sum(dim=0)
            squared_error += torch.square(error).double().sum(dim=0)
            count += error.shape[0]
    per_mae = (absolute_error / count).cpu().numpy()
    per_rmse = torch.sqrt(squared_error / count).cpu().numpy()
    mse = float((squared_error.sum() / (count * OUTPUT_DIMENSION)).cpu())
    mae = float(per_mae.mean())
    rmse = float(np.sqrt(float(squared_error.sum().cpu()) / (count * OUTPUT_DIMENSION)))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "per_output": {
            name: {"mae": float(per_mae[index]), "rmse": float(per_rmse[index])}
            for index, name in enumerate(TARGET_ORDER)
        },
        "transition_count": count,
    }


def _checkpoint(
    *,
    model: V21AControlPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    candidate: ArchitectureCandidate,
    epoch: int,
    metrics: dict[str, Any],
    run_config: dict[str, Any],
    normalization: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_format_version": "final-policy-v2.1a-control-only-v1",
        "amendment_id": AMENDMENT_ID,
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": {
            "architecture_id": candidate.architecture_id,
            "input_dimension": INPUT_DIMENSION,
            "hidden_dimensions": list(candidate.hidden_dimensions),
            "activation": candidate.activation,
            "output_dimension": OUTPUT_DIMENSION,
            "target_order": list(TARGET_ORDER),
        },
        "parameter_count": model.parameter_count,
        "epoch": epoch,
        "development_metrics": metrics,
        "training_configuration": run_config,
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "normalization": normalization,
        "environment": environment,
    }


def _save_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _train_candidate(
    *,
    root: Path,
    candidate: ArchitectureCandidate,
    train_observations: torch.Tensor,
    train_targets: torch.Tensor,
    development_observations: torch.Tensor,
    development_targets: torch.Tensor,
    normalization: dict[str, Any],
    preflight: dict[str, Any],
    source_hash: str,
    config_file_hash: str,
) -> dict[str, Any]:
    seed = ARCHITECTURE_SCREEN_SEED
    _set_seed(seed)
    device = train_observations.device
    model = V21AControlPolicyNetwork(candidate).to(device)
    if model.parameter_count != candidate.expected_parameter_count or model.output_dimension != OUTPUT_DIMENSION:
        raise ValueError(f"implementation contract mismatch for {candidate.architecture_id}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SCREEN_CONTROLS["learning_rate"],
        weight_decay=SCREEN_CONTROLS["weight_decay"],
    )
    checkpoint_dir = root / "checkpoints/final_policy_v1" / candidate.architecture_id / f"seed_{seed}"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite architecture output: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "protocol_id": "final_architecture_screen_v2_1a",
        "amendment_id": AMENDMENT_ID,
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
        "architecture_id": candidate.architecture_id,
        "hidden_dimensions": list(candidate.hidden_dimensions),
        "activation": candidate.activation,
        "parameter_count": model.parameter_count,
        "input_dimension": INPUT_DIMENSION,
        "output_dimension": OUTPUT_DIMENSION,
        "target_order": list(TARGET_ORDER),
        "seed": seed,
        "fit_split": "train",
        "selection_split": "development",
        "forbidden_splits": sorted(FORBIDDEN_SPLITS),
        "training": SCREEN_CONTROLS,
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "source_file_sha256": source_hash,
        "screen_configuration_file_sha256": config_file_hash,
        "source_git_revision": _git_revision(root),
    }
    run_config["configuration_logical_hash"] = object_hash(run_config)
    run_config_path = checkpoint_dir / "run_configuration.json"
    _write_json(run_config_path, run_config)
    run_config_file_hash = file_hash(run_config_path)
    environment = _environment(device)
    history: list[dict[str, Any]] = []
    best_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    batch_size = int(SCREEN_CONTROLS["batch_size"])
    generator = torch.Generator(device=device.type).manual_seed(seed)
    started = datetime.now(timezone.utc)
    started_timer = perf_counter()

    for epoch in range(1, int(SCREEN_CONTROLS["maximum_epochs"]) + 1):
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
            clip_grad_norm_(model.parameters(), float(SCREEN_CONTROLS["gradient_norm_clip"]))
            optimizer.step()
            sum_squared_error += float(loss.detach().cpu()) * targets.numel()
            element_count += targets.numel()
        train_mse = sum_squared_error / element_count
        development = _development_metrics(
            model, development_observations, development_targets, batch_size
        )
        improved = development["mse"] < best_mse - float(SCREEN_CONTROLS["improvement_tolerance"])
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
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
        if epochs_without_improvement >= int(SCREEN_CONTROLS["patience"]):
            break

    _save_checkpoint(checkpoint_dir / "last.pt", checkpoint_value)
    best_path = checkpoint_dir / "best.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    verified_best_metrics = _development_metrics(
        model, development_observations, development_targets, batch_size
    )
    if abs(verified_best_metrics["mse"] - best_mse) > 1e-12:
        raise ValueError("best checkpoint development MSE failed reload verification")
    history_path = checkpoint_dir / "training_history.json"
    _write_json(history_path, history)
    ended = datetime.now(timezone.utc)
    summary = {
        "architecture_id": candidate.architecture_id,
        "hidden_dimensions": list(candidate.hidden_dimensions),
        "activation": candidate.activation,
        "parameter_count": model.parameter_count,
        "seed": seed,
        "epochs_completed": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "development_selection_loss_name": "mean_squared_error",
        "development_selection_loss": best_mse,
        "development_metrics": verified_best_metrics,
        "checkpoint_path": str(best_path.relative_to(root)).replace("\\", "/"),
        "checkpoint_sha256": file_hash(best_path),
        "last_checkpoint_sha256": file_hash(checkpoint_dir / "last.pt"),
        "training_history_path": str(history_path.relative_to(root)).replace("\\", "/"),
        "training_history_sha256": file_hash(history_path),
        "run_configuration_path": str(run_config_path.relative_to(root)).replace("\\", "/"),
        "run_configuration_logical_hash": run_config["configuration_logical_hash"],
        "run_configuration_file_sha256": run_config_file_hash,
        "screen_configuration_file_sha256": config_file_hash,
        "source_file_sha256": source_hash,
        "approved_dataset_logical_hash": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_hash": NORMALIZATION_LOGICAL_HASH,
        "amendment_logical_hash": AMENDMENT_LOGICAL_HASH,
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


def run_architecture_screen(root: str | Path, *, verify_large_file: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    results_dir = root / "results/final_architecture_screen_v2_1a"
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite architecture-screen results: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    preflight = architecture_screen_preflight(root, verify_large_file=verify_large_file)
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
    source_path = root / "src/aircraft_recovery/training/v2_1a_architecture_screen.py"
    config_path = root / "configs/training/final_architecture_screen_v2_1a.yaml"
    source_hash = file_hash(source_path)
    config_file_hash = file_hash(config_path)

    summaries = [
        _train_candidate(
            root=root,
            candidate=candidate,
            train_observations=train_observations,
            train_targets=train_targets,
            development_observations=development_observations,
            development_targets=development_targets,
            normalization=normalization,
            preflight=preflight,
            source_hash=source_hash,
            config_file_hash=config_file_hash,
        )
        for candidate in ARCHITECTURE_CANDIDATES
    ]
    winner = min(
        summaries,
        key=lambda row: (row["development_selection_loss"], row["parameter_count"]),
    )
    screen = {
        "result_version": "final-architecture-screen-v2.1a-v1",
        "status": "COMPLETE",
        "selection_rule": "minimum_development_mse_then_lower_parameter_count_exact_tie",
        "winner_architecture_id": winner["architecture_id"],
        "winner_checkpoint_path": winner["checkpoint_path"],
        "architectures": summaries,
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
        "amendment_file_sha256": preflight["amendment_file_sha256"],
        "screen_configuration_file_sha256": config_file_hash,
        "source_file_sha256": source_hash,
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
    }
    _write_json(results_dir / "architecture_screen_summary.json", screen)
    return screen
