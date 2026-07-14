"""Weighted supervised imitation objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from aircraft_recovery.training.models import TARGET_SCALES


@dataclass(frozen=True)
class LossWeights:
    controls: float = 1.0
    emergency_mode: float = 0.25
    stabilization_targets: float = 0.1


def imitation_loss(
    predictions: dict[str, torch.Tensor],
    controls: torch.Tensor,
    modes: torch.Tensor,
    targets: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Huber control/target losses and emergency-mode cross entropy."""
    control_loss = functional.huber_loss(predictions["controls"], controls)
    mode_loss = functional.cross_entropy(predictions["mode_logits"], modes)
    target_loss = torch.zeros((), device=controls.device)
    if "targets_normalized" in predictions:
        scales = torch.tensor(TARGET_SCALES, device=targets.device, dtype=targets.dtype)
        target_loss = functional.huber_loss(predictions["targets_normalized"], targets / scales)
    total = (
        weights.controls * control_loss
        + weights.emergency_mode * mode_loss
        + weights.stabilization_targets * target_loss
    )
    return total, {"control": control_loss, "mode": mode_loss, "targets": target_loss}

