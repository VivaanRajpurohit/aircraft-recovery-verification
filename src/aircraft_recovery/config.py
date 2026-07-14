"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aircraft_recovery.failures.models import FailureDefinition

T = TypeVar("T", bound=BaseModel)


class AircraftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_step_seconds: float = Field(gt=0.0, le=1.0)
    controller_frequency_hz: float = Field(gt=0.0, le=100.0)
    stall_speed_kts: float = Field(gt=0.0)
    overspeed_kts: float = Field(gt=0.0)
    cruise_speed_kts: float = Field(gt=0.0)
    max_thrust_accel_kts_s: float = Field(gt=0.0)
    drag_coefficient_per_s: float = Field(gt=0.0)
    pitch_rate_deg_s: float = Field(gt=0.0)
    roll_rate_deg_s: float = Field(gt=0.0)
    yaw_rate_deg_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def control_rate_matches_step(self) -> "AircraftConfig":
        """Phase 1 evaluates one fresh command for every dynamics step."""
        if abs(self.controller_frequency_hz * self.time_step_seconds - 1.0) > 1e-9:
            raise ValueError(
                "Phase 1 requires controller_frequency_hz == 1 / time_step_seconds"
            )
        return self


class InitialStateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    altitude_ft: float
    airspeed_kts: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    heading_deg: float = Field(ge=0.0, lt=360.0)
    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)


class SyntheticAirportConfig(BaseModel):
    """Rich internal airport model projected into the unchanged input schema."""

    model_config = ConfigDict(extra="forbid")
    airport_id: str
    distance_nm: float = Field(ge=0.0)
    bearing_deg: float = Field(ge=0.0, lt=360.0)
    runway_heading_deg: float = Field(ge=0.0, lt=360.0)
    runway_length_ft: float = Field(gt=0.0)
    terrain_clearance_ft: float
    within_glide_range: bool = False
    latitude_deg: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    elevation_ft: float = 0.0
    runway_width_ft: float = Field(default=100.0, gt=0.0)
    terrain_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    available: bool = True
    surface_suitable: bool = True
    minimum_control_authority: float = Field(default=0.0, ge=0.0, le=1.0)


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str = Field(min_length=1)
    description: str
    seed: int = Field(ge=0)
    duration_seconds: float = Field(gt=0.0)
    recoverability: Literal["recoverable", "marginal", "unrecoverable", "unknown"] = "unknown"
    expected_recoverability: Literal["recoverable", "marginal", "unrecoverable", "unknown"] | None = None
    initial_state: InitialStateConfig
    airports: list[SyntheticAirportConfig]
    environment: dict[str, Any]
    failures: list[FailureDefinition] = Field(default_factory=list)
    termination: dict[str, Any] = Field(default_factory=dict)
    diversion_requested: bool = False

    @model_validator(mode="after")
    def normalize_recoverability(self) -> "ScenarioConfig":
        if self.expected_recoverability is None:
            self.expected_recoverability = self.recoverability
        else:
            self.recoverability = self.expected_recoverability
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with useful errors."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return data


def load_config(path: str | Path, model: type[T]) -> T:
    """Load and validate YAML as the requested Pydantic model."""
    return model.model_validate(load_yaml(path))
