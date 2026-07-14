"""Sound interval bounds for the existing checkpoint-backed ReLU policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from torch import nn

from aircraft_recovery.data.preprocessing import PreprocessingStatistics
from aircraft_recovery.training.checkpointing import load_checkpoint
from aircraft_recovery.training.models import ImitationPolicyNetwork, PolicyArchitectureConfig


@dataclass(frozen=True)
class Interval:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.shape != upper.shape:
            raise ValueError("Interval endpoints must have matching shapes")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Interval endpoints must be finite")
        if np.any(lower > upper):
            raise ValueError("Every interval lower bound must be <= its upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class PolicyBounds:
    controls: Interval
    mode_logits: Interval
    targets_normalized: Interval | None
    checkpoint_sha256: str | None
    method: str = "sound_interval_bound_propagation"


def _outward(interval: Interval) -> Interval:
    """Round away from the represented interval using adjacent float64 values."""
    return Interval(
        np.nextafter(interval.lower, -np.inf),
        np.nextafter(interval.upper, np.inf),
    )


def affine_bounds(interval: Interval, layer: nn.Linear) -> Interval:
    weights = layer.weight.detach().cpu().numpy().astype(np.float64)
    bias = layer.bias.detach().cpu().numpy().astype(np.float64)
    positive = np.maximum(weights, 0.0)
    negative = np.minimum(weights, 0.0)
    lower = positive @ interval.lower + negative @ interval.upper + bias
    upper = positive @ interval.upper + negative @ interval.lower + bias
    return _outward(Interval(lower, upper))


def _monotone(interval: Interval, function) -> Interval:
    return _outward(Interval(function(interval.lower), function(interval.upper)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _bounded_outward(interval: Interval, minimum: float, maximum: float) -> Interval:
    widened = _outward(interval)
    return Interval(np.maximum(widened.lower, minimum), np.minimum(widened.upper, maximum))


def normalize_observation_bounds(raw: Interval, statistics: PreprocessingStatistics) -> Interval:
    if raw.lower.shape != (42,):
        raise ValueError("Raw observation bounds must have shape (42,)")
    means = np.asarray(statistics.means, dtype=np.float64)
    deviations = np.asarray(statistics.standard_deviations, dtype=np.float64)
    normalized = np.asarray(statistics.normalized, dtype=bool)
    if np.any(deviations <= 0.0):
        raise ValueError("Normalization deviations must be positive")
    lower = raw.lower.copy()
    upper = raw.upper.copy()
    lower[normalized] = (lower[normalized] - means[normalized]) / deviations[normalized]
    upper[normalized] = (upper[normalized] - means[normalized]) / deviations[normalized]
    clip = statistics.clip_standard_deviations
    lower[normalized] = np.clip(lower[normalized], -clip, clip)
    upper[normalized] = np.clip(upper[normalized], -clip, clip)
    return _outward(Interval(lower, upper))


def bound_policy_model(
    model: ImitationPolicyNetwork,
    normalized_observations: Interval,
    *,
    checkpoint_sha256: str | None = None,
) -> PolicyBounds:
    """Propagate an input box through the exact policy parameters using IBP."""
    if normalized_observations.lower.shape != (model.config.input_dim,):
        raise ValueError(f"Policy input bounds must have shape ({model.config.input_dim},)")
    if model.config.activation != "relu":
        raise ValueError("The current sound backend supports ReLU checkpoints only")

    hidden = normalized_observations
    for layer in model.trunk:
        if isinstance(layer, nn.Linear):
            hidden = affine_bounds(hidden, layer)
        elif isinstance(layer, nn.ReLU):
            hidden = _outward(Interval(np.maximum(hidden.lower, 0.0), np.maximum(hidden.upper, 0.0)))
        else:
            raise TypeError(f"Unsupported policy layer for sound IBP: {type(layer).__name__}")

    raw_controls = affine_bounds(hidden, model.control_head)
    surfaces = _bounded_outward(Interval(
        np.tanh(raw_controls.lower[:3]), np.tanh(raw_controls.upper[:3])
    ), -1.0, 1.0)
    throttles = _bounded_outward(Interval(
        _sigmoid(raw_controls.lower[3:]), _sigmoid(raw_controls.upper[3:])
    ), 0.0, 1.0)
    controls = Interval(
        np.concatenate((surfaces.lower, throttles.lower)),
        np.concatenate((surfaces.upper, throttles.upper)),
    )
    mode_logits = affine_bounds(hidden, model.mode_head)

    targets = None
    if model.target_head is not None:
        raw_targets = affine_bounds(hidden, model.target_head)
        target_angles = _bounded_outward(Interval(
            np.tanh(raw_targets.lower[:2]), np.tanh(raw_targets.upper[:2])
        ), -1.0, 1.0)
        target_positive = _bounded_outward(Interval(
            _sigmoid(raw_targets.lower[2:]), _sigmoid(raw_targets.upper[2:])
        ), 0.0, 1.0)
        targets = Interval(
            np.concatenate((target_angles.lower, target_positive.lower)),
            np.concatenate((target_angles.upper, target_positive.upper)),
        )
    return PolicyBounds(controls, mode_logits, targets, checkpoint_sha256)


def bound_checkpoint(checkpoint_path: str | Path, raw_observations: Interval) -> PolicyBounds:
    path = Path(checkpoint_path)
    checkpoint = load_checkpoint(path, map_location="cpu")
    architecture = PolicyArchitectureConfig.model_validate(checkpoint["architecture_configuration"])
    model = ImitationPolicyNetwork(architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    statistics = PreprocessingStatistics.from_dict(checkpoint["normalization_statistics"])
    normalized = normalize_observation_bounds(raw_observations, statistics)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return bound_policy_model(model, normalized, checkpoint_sha256=digest)
