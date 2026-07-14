"""Metadata-rich checkpoint save/load and compatibility validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data.preprocessing import PreprocessingStatistics
from aircraft_recovery.training.models import OUTPUT_SCALING_RULES, ImitationPolicyNetwork, PolicyArchitectureConfig


CHECKPOINT_FORMAT_VERSION = 1
OBSERVATION_VECTOR_VERSION = "phase1-42-v1"


def build_checkpoint(
    model: ImitationPolicyNetwork,
    preprocessing: PreprocessingStatistics,
    training_configuration: dict[str, Any],
    dataset_identifier: str,
    split_manifest_identifier: str,
    software_versions: dict[str, Any],
    validation_metrics: dict[str, float],
    epoch: int,
    optimizer_state_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "architecture_configuration": model.config.model_dump(mode="json"),
        "parameter_count": model.parameter_count,
        "observation_vector_version": OBSERVATION_VECTOR_VERSION,
        "observation_feature_names": list(OBSERVATION_FEATURES),
        "normalization_statistics": preprocessing.to_dict(),
        "output_scaling_rules": OUTPUT_SCALING_RULES,
        "training_configuration": training_configuration,
        "dataset_identifier": dataset_identifier,
        "split_manifest_identifier": split_manifest_identifier,
        "software_versions": software_versions,
        "validation_metrics": validation_metrics,
        "epoch": epoch,
    }


def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as error:
        raise ValueError(f"Unable to load checkpoint {checkpoint_path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    required = {
        "checkpoint_format_version", "model_state_dict", "architecture_configuration",
        "parameter_count", "observation_vector_version", "observation_feature_names",
        "normalization_statistics", "output_scaling_rules", "training_configuration",
        "dataset_identifier", "split_manifest_identifier", "software_versions",
        "validation_metrics", "epoch",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint missing required fields: {sorted(missing)}")
    if checkpoint["checkpoint_format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version")
    if checkpoint["observation_vector_version"] != OBSERVATION_VECTOR_VERSION:
        raise ValueError("Checkpoint observation-vector version mismatch")
    if checkpoint["observation_feature_names"] != list(OBSERVATION_FEATURES):
        raise ValueError("Checkpoint observation feature names mismatch")
    architecture = PolicyArchitectureConfig.model_validate(checkpoint["architecture_configuration"])
    model = ImitationPolicyNetwork(architecture)
    if checkpoint["parameter_count"] != model.parameter_count:
        raise ValueError("Checkpoint parameter count does not match architecture")
    PreprocessingStatistics.from_dict(checkpoint["normalization_statistics"])
    return checkpoint

