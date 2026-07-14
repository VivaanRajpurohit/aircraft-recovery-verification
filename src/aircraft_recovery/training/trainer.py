"""Reproducible supervised imitation trainer for CPU and CUDA."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import random
import sys
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from aircraft_recovery.data.dataset import DemonstrationDataset, MODE_NAMES
from aircraft_recovery.data.hashing import canonical_hash
from aircraft_recovery.data.preprocessing import ObservationPreprocessor, PreprocessingStatistics
from aircraft_recovery.data.splitting import validate_no_episode_overlap
from aircraft_recovery.training.checkpointing import build_checkpoint, load_checkpoint, save_checkpoint
from aircraft_recovery.training.losses import LossWeights, imitation_loss
from aircraft_recovery.training.models import ImitationPolicyNetwork, PolicyArchitectureConfig


class LossConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    controls: float = Field(ge=0.0)
    emergency_mode: float = Field(ge=0.0)
    stabilization_targets: float = Field(ge=0.0)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_name: str
    dataset_directory: str
    split_manifest: str
    output_directory: str
    seed: int = Field(ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    architecture: PolicyArchitectureConfig
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    epochs: int = Field(gt=0)
    weight_decay: float = Field(ge=0.0)
    gradient_clip_norm: float = Field(gt=0.0)
    early_stopping_patience: int = Field(gt=0)
    checkpoint_every_epochs: int = Field(gt=0)
    clip_standard_deviations: float = Field(gt=0.0)
    balance_emergency_modes: bool = True
    automatic_mixed_precision: bool = False
    num_workers: int = Field(default=0, ge=0)
    resume_checkpoint: str | None = None
    losses: LossConfiguration


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(
    dataset: DemonstrationDataset,
    batch_size: int,
    seed: int,
    workers: int,
    balanced: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = not balanced
    if balanced:
        counts = Counter(int(mode) for mode in dataset.modes)
        weights = torch.tensor([1.0 / counts[int(mode)] for mode in dataset.modes], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=workers, generator=generator,
    )


def _run_epoch(
    model: ImitationPolicyNetwork,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    weights: LossWeights,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter[str]()
    batches = 0
    for batch in loader:
        observations = batch["observation"].to(device)
        controls = batch["controls"].to(device)
        targets = batch["targets"].to(device)
        modes = batch["mode"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(observations)
                total, components = imitation_loss(predictions, controls, modes, targets, weights)
            if optimizer is not None:
                if scaler is not None:
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
        totals["total"] += float(total.detach().cpu())
        for name, value in components.items():
            totals[name] += float(value.detach().cpu())
        batches += 1
    if batches == 0:
        raise ValueError("Dataset split contains no transitions")
    return {name: value / batches for name, value in totals.items()}


def train_policy(config: TrainingConfig) -> dict[str, Any]:
    """Train, early-stop, and save a self-describing best checkpoint."""
    started_at = datetime.now(timezone.utc)
    set_reproducible_seed(config.seed)
    device = resolve_device(config.device)
    if config.automatic_mixed_precision and device.type != "cuda":
        raise ValueError("Automatic mixed precision is enabled only for CUDA training")
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(config.split_manifest).read_text(encoding="utf-8"))
    validate_no_episode_overlap(manifest)
    train_dataset = DemonstrationDataset(config.dataset_directory, set(manifest["train"]))
    validation_dataset = DemonstrationDataset(config.dataset_directory, set(manifest["validation"]))
    dataset_metadata = train_dataset.metadata

    model = ImitationPolicyNetwork(config.architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_epoch = 1
    if config.resume_checkpoint:
        resumed = load_checkpoint(config.resume_checkpoint, map_location=device)
        if resumed["architecture_configuration"] != config.architecture.model_dump(mode="json"):
            raise ValueError("Resume checkpoint architecture differs from training configuration")
        if resumed["split_manifest_identifier"] != manifest["identifier"]:
            raise ValueError("Resume checkpoint split manifest differs from current manifest")
        model.load_state_dict(resumed["model_state_dict"])
        if resumed.get("optimizer_state_dict"):
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
        preprocessor = ObservationPreprocessor(
            PreprocessingStatistics.from_dict(resumed["normalization_statistics"])
        )
        start_epoch = int(resumed["epoch"]) + 1
    else:
        preprocessor = ObservationPreprocessor()
        preprocessor.fit(train_dataset.observations, config.clip_standard_deviations)
    train_dataset.observations = preprocessor.transform(train_dataset.observations)
    validation_dataset.observations = preprocessor.transform(validation_dataset.observations)

    train_loader = _loader(
        train_dataset, config.batch_size, config.seed, config.num_workers,
        config.balance_emergency_modes,
    )
    validation_loader = _loader(
        validation_dataset, config.batch_size, config.seed + 1, config.num_workers, False
    )
    loss_weights = LossWeights(**config.losses.model_dump())
    scaler = torch.amp.GradScaler("cuda") if config.automatic_mixed_precision else None
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    software = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, device, loss_weights, optimizer,
            config.gradient_clip_norm, config.automatic_mixed_precision, scaler,
        )
        validation_metrics = _run_epoch(
            model, validation_loader, device, loss_weights, None,
            config.gradient_clip_norm, False,
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        checkpoint = build_checkpoint(
            model=model,
            preprocessing=preprocessor.statistics,
            training_configuration=config.model_dump(mode="json"),
            dataset_identifier=dataset_metadata["dataset_identifier"],
            split_manifest_identifier=manifest["identifier"],
            software_versions=software,
            validation_metrics=validation_metrics,
            epoch=epoch,
            optimizer_state_dict=optimizer.state_dict(),
        )
        if epoch % config.checkpoint_every_epochs == 0:
            save_checkpoint(output / f"epoch_{epoch:04d}.pt", checkpoint)
        if validation_metrics["total"] < best_loss - 1e-10:
            best_loss = validation_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output / "best.pt", checkpoint)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    save_checkpoint(output / "last.pt", checkpoint)
    ended_at = datetime.now(timezone.utc)
    original_modes = train_dataset.mode_counts()
    present_modes = [name for name in MODE_NAMES if original_modes.get(name, 0) > 0]
    effective_distribution = (
        {name: 1.0 / len(present_modes) for name in present_modes}
        if config.balance_emergency_modes else
        {name: count / len(train_dataset) for name, count in original_modes.items()}
    )
    result = {
        "experiment_name": config.experiment_name,
        "device": str(device),
        "parameter_count": model.parameter_count,
        "architecture": config.architecture.model_dump(mode="json"),
        "dataset_identifier": dataset_metadata["dataset_identifier"],
        "split_manifest_identifier": manifest["identifier"],
        "configuration_hash": canonical_hash(config.model_dump(mode="json")),
        "training_start_utc": started_at.isoformat(),
        "training_end_utc": ended_at.isoformat(),
        "best_epoch": best_epoch,
        "epochs_completed": history[-1]["epoch"],
        "best_validation_loss": best_loss,
        "final_train_losses": history[-1]["train"],
        "final_validation_losses": history[-1]["validation"],
        "original_training_mode_distribution": original_modes,
        "effective_weighted_sampling_distribution": effective_distribution,
        "weighted_sampler_replacement": config.balance_emergency_modes,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "software": software,
        "best_checkpoint": str(output / "best.pt"),
    }
    (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
