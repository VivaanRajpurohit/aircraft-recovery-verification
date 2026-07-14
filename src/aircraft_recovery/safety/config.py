"""Validated configuration for bounded Phase 4 safety analysis."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CheckedStateBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum_airspeed_kts: float
    maximum_airspeed_kts: float
    minimum_pitch_deg: float
    maximum_pitch_deg: float
    maximum_absolute_roll_deg: float = Field(gt=0.0)
    minimum_angle_of_attack_deg: float
    maximum_angle_of_attack_deg: float
    minimum_load_factor_g: float
    maximum_load_factor_g: float
    minimum_terrain_clearance_ft: float


class SafetyThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stall_speed_kts: float = Field(gt=0.0)
    stall_margin_kts: float = Field(ge=0.0)
    overspeed_kts: float = Field(gt=0.0)
    overspeed_margin_kts: float = Field(ge=0.0)
    maximum_absolute_bank_deg: float = Field(gt=0.0)
    minimum_pitch_deg: float
    maximum_pitch_deg: float
    maximum_angle_of_attack_deg: float
    minimum_load_factor_g: float
    maximum_load_factor_g: float
    minimum_terrain_clearance_ft: float
    maximum_control_rate_per_second: float = Field(gt=0.0)


class BoundedDynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_step_seconds: float = Field(gt=0.0)
    horizon_steps: int = Field(default=1, ge=1, le=5)
    cruise_speed_kts: float
    thrust_gain_kts_per_second: float = Field(gt=0.0)
    drag_gain_per_second: float = Field(ge=0.0)
    pitch_rate_deg_per_second: float = Field(gt=0.0)
    roll_rate_deg_per_second: float = Field(gt=0.0)
    angle_of_attack_gain: float = Field(gt=0.0)
    load_factor_gain: float = Field(ge=0.0)
    maximum_speed_disturbance_kts: float = Field(ge=0.0)
    maximum_pitch_disturbance_deg: float = Field(ge=0.0)
    maximum_roll_disturbance_deg: float = Field(ge=0.0)
    maximum_aoa_disturbance_deg: float = Field(ge=0.0)
    maximum_load_disturbance_g: float = Field(ge=0.0)
    maximum_terrain_loss_ft: float = Field(ge=0.0)


class MonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solver_timeout_ms: int = Field(gt=0)
    projection_blend_factors: list[float] = Field(min_length=1)
    reject_invalid_navigation: bool = True
    reject_nonfinite_input: bool = True


class SafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    configuration_name: str
    thresholds: SafetyThresholds
    dynamics: BoundedDynamicsConfig
    checked_state_bounds: CheckedStateBounds
    monitor: MonitorConfig
    assumptions: list[str]
    limitations: list[str]

