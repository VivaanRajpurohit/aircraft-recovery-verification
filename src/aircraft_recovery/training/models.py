"""Configurable bounded MLP imitation policy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
from torch import nn


TARGET_SCALES = (30.0, 60.0, 250.0, 60000.0)
OUTPUT_SCALING_RULES = {
    "elevator_aileron_rudder": "tanh -> [-1, 1]",
    "throttle_left_right": "sigmoid -> [0, 1]",
    "target_pitch_deg": "tanh * 30",
    "target_roll_deg": "tanh * 60",
    "target_airspeed_kts": "sigmoid * 250",
    "target_altitude_ft": "sigmoid * 60000",
}


class PolicyArchitectureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_dim: int = 42
    hidden_dimensions: list[int] = Field(default_factory=lambda: [256, 128], min_length=1)
    activation: Literal["silu", "relu"] = "relu"
    predict_stabilization_targets: bool = True
    emergency_mode_count: int = 5

    @model_validator(mode="after")
    def enforce_interface_dimensions(self) -> "PolicyArchitectureConfig":
        if self.input_dim != 42:
            raise ValueError("Phase 3 policy input_dim must remain exactly 42")
        if self.emergency_mode_count != 5:
            raise ValueError("Existing output schema has exactly five public emergency modes")
        if any(value <= 0 for value in self.hidden_dimensions):
            raise ValueError("Hidden dimensions must be positive")
        return self


class ImitationPolicyNetwork(nn.Module):
    """Shared MLP trunk with bounded controls, targets, and mode logits."""

    def __init__(self, config: PolicyArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        activation_type: type[nn.Module] = nn.SiLU if config.activation == "silu" else nn.ReLU
        layers: list[nn.Module] = []
        previous = config.input_dim
        for width in config.hidden_dimensions:
            layers.extend([nn.Linear(previous, width), activation_type()])
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.control_head = nn.Linear(previous, 5)
        self.mode_head = nn.Linear(previous, config.emergency_mode_count)
        self.target_head = nn.Linear(previous, 4) if config.predict_stabilization_targets else None

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(observations)
        raw_controls = self.control_head(hidden)
        controls = torch.cat((torch.tanh(raw_controls[..., :3]), torch.sigmoid(raw_controls[..., 3:])), dim=-1)
        result = {"controls": controls, "mode_logits": self.mode_head(hidden)}
        if self.target_head is not None:
            raw_targets = self.target_head(hidden)
            result["targets_normalized"] = torch.cat((
                torch.tanh(raw_targets[..., :2]),
                torch.sigmoid(raw_targets[..., 2:]),
            ), dim=-1)
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
