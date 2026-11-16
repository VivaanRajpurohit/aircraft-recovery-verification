"""Fail-closed preflight for the frozen advanced v2.1 architecture screen.

This module intentionally exposes only train and development readers. It will
not permit training when the approved export cannot supply every label required
by the frozen multi-head objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch
from torch.nn import functional as F

from aircraft_recovery.data.dataset import MODE_NAMES
from aircraft_recovery.data_quality.common import file_hash
from aircraft_recovery.data_quality.final_approved import (
    APPROVED_ID,
    assert_role_split,
    iter_approved_records,
    validate_approved_manifest_contract,
    validate_normalization_contract,
)
from aircraft_recovery.training.models import (
    ImitationPolicyNetwork,
    PolicyArchitectureConfig,
    TARGET_SCALES,
)


APPROVED_DATASET_LOGICAL_HASH = (
    "bc69cbcb5bee7189c091dba51159f8938bdd661b3793007f45c7f9eb2897b43a"
)
NORMALIZATION_LOGICAL_HASH = (
    "005ccca1bfc32214482e89e552761d73dbeefbce651e7a7abee60037ad978cb7"
)
APPROVED_TRANSITIONS_FILE_HASH = (
    "187374b333f99a2df1891dbb06dad5de7a1eadeb4bc9f1ed5282b4094f64d2e1"
)
NORMALIZATION_FILE_HASH = (
    "3fa7538fabe52658ce10ed1606ae8ad0696c85ec5e517e691ba42e696a38f59f"
)
MONITOR_FILE_HASH = (
    "427d37c1b0c15d8659cab811876bd05b148fb627bba949f86709369f46846281"
)
FALLBACK_FILE_HASH = (
    "67d08f11c510d5b68e3fc8148589fd8e1a5bf270dc7b9dd637c1d7f1b6562412"
)
SEED_REGISTRY_FILE_HASH = (
    "98fa364f3168412ccec9017d24a2ed8217f35f40294d30d997d0e30135165319"
)

ARCHITECTURE_SCREEN_ALLOWED_SPLITS = frozenset({"train", "development"})
ARCHITECTURE_SCREEN_FORBIDDEN_SPLITS = frozenset(
    {"challenge", "ood_stress", "final_sealed_test"}
)
ARCHITECTURE_SCREEN_SEED = 17001
FINAL_TRAINING_SEEDS = (17001, 17002, 17003, 17004, 17005)
OPTIMIZER_GRID = {
    "learning_rates": (0.0003, 0.001),
    "batch_sizes": (256, 512),
}
SCREEN_TRAINING_CONTROLS = {
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "batch_size": 256,
    "weight_decay": 0.0001,
    "maximum_epochs": 50,
    "patience": 8,
    "improvement_tolerance": 1e-10,
    "gradient_norm_clip": 1.0,
}
LOSS_WEIGHTS = {"controls": 1.0, "emergency_mode": 0.25, "stabilization_targets": 0.1}

REQUIRED_MULTI_HEAD_LABEL_FIELDS = (
    "expert_emergency_mode",
    "expert_stabilization_targets",
)


class ArchitectureScreenBlocked(RuntimeError):
    """Raised before optimization when frozen protocol requirements are unmet."""

    def __init__(self, issues: Iterable[str], report: dict[str, Any]) -> None:
        self.issues = tuple(issues)
        self.report = report
        super().__init__("advanced v2.1 architecture screen blocked: " + "; ".join(self.issues))


@dataclass(frozen=True)
class ArchitectureCandidate:
    architecture_id: str
    hidden_dimensions: tuple[int, ...]
    activation: str

    def model_config(self) -> PolicyArchitectureConfig:
        return PolicyArchitectureConfig(
            input_dim=42,
            hidden_dimensions=list(self.hidden_dimensions),
            activation=self.activation,
            predict_stabilization_targets=True,
            emergency_mode_count=5,
        )

    @property
    def parameter_count(self) -> int:
        return ImitationPolicyNetwork(self.model_config()).parameter_count


ARCHITECTURE_CANDIDATES = (
    ArchitectureCandidate("small_reference", (128, 64), "relu"),
    ArchitectureCandidate("current_reference", (256, 128), "relu"),
    ArchitectureCandidate("moderate", (256, 256, 128), "silu"),
)


RecordLoader = Callable[..., Iterable[dict[str, Any]]]


class ArchitectureScreenDataSource:
    """The complete data surface available to architecture screening."""

    def __init__(self, dataset: str | Path, loader: RecordLoader = iter_approved_records) -> None:
        self.dataset = Path(dataset)
        self._loader = loader

    def training_records(self) -> Iterator[dict[str, Any]]:
        yield from self._loader(self.dataset, split="train", role="training")

    def development_records(self) -> Iterator[dict[str, Any]]:
        yield from self._loader(
            self.dataset, split="development", role="model_selection"
        )


def assert_architecture_screen_access_policy() -> None:
    """Prove the two allowed requests and rejection of every held-out request."""
    assert_role_split("training", "train")
    assert_role_split("model_selection", "development")
    for role in ("training", "model_selection"):
        for split in ARCHITECTURE_SCREEN_FORBIDDEN_SPLITS:
            try:
                assert_role_split(role, split)
            except PermissionError:
                continue
            raise AssertionError(f"{role} unexpectedly gained access to {split}")


def frozen_protocol_summary() -> dict[str, Any]:
    return {
        "architecture_candidates": [
            {
                "architecture_id": item.architecture_id,
                "hidden_dimensions": list(item.hidden_dimensions),
                "activation": item.activation,
                "parameter_count": item.parameter_count,
            }
            for item in ARCHITECTURE_CANDIDATES
        ],
        "architecture_screen_seed": ARCHITECTURE_SCREEN_SEED,
        "optimizer_grid": {
            "learning_rates": list(OPTIMIZER_GRID["learning_rates"]),
            "batch_sizes": list(OPTIMIZER_GRID["batch_sizes"]),
        },
        "screen_training_controls": SCREEN_TRAINING_CONTROLS,
        "loss_weights": LOSS_WEIGHTS,
        "final_training_seeds": list(FINAL_TRAINING_SEEDS),
        "fit_split": "train",
        "selection_split": "development",
        "forbidden_splits": sorted(ARCHITECTURE_SCREEN_FORBIDDEN_SPLITS),
    }


def multi_head_imitation_loss(
    predictions: dict[str, torch.Tensor],
    controls: torch.Tensor,
    modes: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Implement the frozen MSE/CE/MSE objective without legacy Huber loss."""
    control_loss = F.mse_loss(predictions["controls"], controls)
    mode_loss = F.cross_entropy(predictions["mode_logits"], modes)
    scales = torch.as_tensor(TARGET_SCALES, dtype=targets.dtype, device=targets.device)
    target_loss = F.mse_loss(predictions["targets_normalized"], targets / scales)
    total = (
        LOSS_WEIGHTS["controls"] * control_loss
        + LOSS_WEIGHTS["emergency_mode"] * mode_loss
        + LOSS_WEIGHTS["stabilization_targets"] * target_loss
    )
    return total, {
        "controls_mse": control_loss,
        "mode_cross_entropy": mode_loss,
        "targets_normalized_mse": target_loss,
    }


def validate_multi_head_label_contract(record: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_MULTI_HEAD_LABEL_FIELDS if name not in record]
    if missing:
        raise ValueError(f"approved record is missing frozen multi-head labels: {missing}")
    mode = record["expert_emergency_mode"]
    if mode not in MODE_NAMES:
        raise ValueError(f"invalid expert_emergency_mode: {mode!r}")
    targets = record["expert_stabilization_targets"]
    if not isinstance(targets, list) or len(targets) != 4:
        raise ValueError("expert_stabilization_targets must be a four-value list")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def architecture_screen_preflight(root: str | Path, *, verify_large_file: bool = True) -> dict[str, Any]:
    """Verify frozen inputs and fail before optimization on any ambiguity."""
    root = Path(root).resolve()
    approved = root / "datasets" / "approved_policy_data" / APPROVED_ID
    manifest_path = approved / "manifests" / "approved_manifest.json"
    normalization_path = approved / "preprocessing" / "train_only_normalization.json"
    transitions_path = approved / "raw" / "transitions.jsonl"
    monitor_path = root / "configs" / "research" / "final_monitor_v1.yaml"
    fallback_path = root / "configs" / "research" / "final_fallback_v1.yaml"
    seed_registry_path = root / "configs" / "research" / "final_seed_registry_v1.json"

    assert_architecture_screen_access_policy()
    manifest = _load_json(manifest_path)
    normalization = _load_json(normalization_path)
    validate_approved_manifest_contract(manifest)
    validate_normalization_contract(normalization)

    observed_hashes = {
        "dataset_logical_sha256": manifest.get("approved_dataset_logical_hash"),
        "normalization_logical_sha256": normalization.get("normalization_artifact_hash"),
        "normalization_file_sha256": file_hash(normalization_path),
        "monitor_file_sha256": file_hash(monitor_path),
        "fallback_file_sha256": file_hash(fallback_path),
        "seed_registry_file_sha256": file_hash(seed_registry_path),
    }
    if verify_large_file:
        observed_hashes["approved_transitions_file_sha256"] = file_hash(transitions_path)

    expected_hashes = {
        "dataset_logical_sha256": APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "normalization_file_sha256": NORMALIZATION_FILE_HASH,
        "monitor_file_sha256": MONITOR_FILE_HASH,
        "fallback_file_sha256": FALLBACK_FILE_HASH,
        "seed_registry_file_sha256": SEED_REGISTRY_FILE_HASH,
    }
    if verify_large_file:
        expected_hashes["approved_transitions_file_sha256"] = APPROVED_TRANSITIONS_FILE_HASH

    issues = [
        f"hash mismatch for {name}: {observed_hashes.get(name)} != {expected}"
        for name, expected in expected_hashes.items()
        if observed_hashes.get(name) != expected
    ]

    source = ArchitectureScreenDataSource(approved)
    first_train_record = next(source.training_records(), None)
    if first_train_record is None:
        issues.append("approved train split is empty")
        observed_fields: list[str] = []
    else:
        observed_fields = sorted(first_train_record)
        try:
            validate_multi_head_label_contract(first_train_record)
        except ValueError as exc:
            issues.append(str(exc))

    issues.append(
        "the frozen protocol does not state an unambiguous per-epoch checkpoint-selection metric"
    )
    issues.append(
        "the frozen final architecture rule requires challenge/closed-loop veto evidence, "
        "which is outside the currently authorized development-only stage"
    )

    report = {
        "report_version": "v2.1-architecture-screen-preflight-v1",
        "status": "BLOCKED" if issues else "PASS",
        "dataset_id": APPROVED_ID,
        "expected_hashes": expected_hashes,
        "observed_hashes": observed_hashes,
        "protocol": frozen_protocol_summary(),
        "required_multi_head_label_fields": list(REQUIRED_MULTI_HEAD_LABEL_FIELDS),
        "observed_approved_record_fields": observed_fields,
        "checkpoint_selection_rule": "unresolved_in_frozen_protocol",
        "issues": issues,
        "optimization_started": False,
        "challenge_opened": False,
        "ood_opened": False,
        "sealed_opened": False,
    }
    if issues:
        raise ArchitectureScreenBlocked(issues, report)
    return report

