"""Configuration and runtime records for deterministic failure injection."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FailureType(str, Enum):
    LEFT_ENGINE_FAILURE = "left_engine_failure"
    RIGHT_ENGINE_FAILURE = "right_engine_failure"
    PARTIAL_THRUST_LOSS = "partial_thrust_loss"
    ASYMMETRIC_THRUST = "asymmetric_thrust"
    DELAYED_THROTTLE_RESPONSE = "delayed_throttle_response"
    REDUCED_ELEVATOR = "reduced_elevator_effectiveness"
    REDUCED_AILERON = "reduced_aileron_effectiveness"
    REDUCED_RUDDER = "reduced_rudder_effectiveness"
    STUCK_ELEVATOR = "stuck_elevator"
    STUCK_AILERON = "stuck_aileron"
    STUCK_RUDDER = "stuck_rudder"
    ACTUATOR_DELAY = "actuator_delay"
    CONTROL_RATE_LIMIT = "control_command_rate_limitation"
    AIRSPEED_BIAS = "airspeed_bias"
    ALTITUDE_BIAS = "altitude_bias"
    HEADING_BIAS = "heading_bias"
    FROZEN_SENSOR = "frozen_sensor_value"
    GAUSSIAN_NOISE = "gaussian_measurement_noise"
    TEMPORARY_MISSING_SENSOR = "temporary_missing_sensor_data"
    NAVIGATION_INVALID = "navigation_data_invalidation"
    TURBULENCE_INCREASE = "turbulence_increase"
    CROSSWIND_INCREASE = "crosswind_increase"
    HEADWIND_CHANGE = "headwind_or_tailwind_change"
    REDUCED_VISIBILITY = "reduced_visibility"


class StateCondition(BaseModel):
    """Simple safe predicate evaluated against named true-state values."""

    model_config = ConfigDict(extra="forbid")
    field: str
    operator: Literal["lt", "le", "gt", "ge", "eq"]
    value: float


class FailureDefinition(BaseModel):
    """One uniquely identified, scheduled simulator failure."""

    model_config = ConfigDict(extra="forbid")
    failure_id: str = Field(min_length=1)
    failure_type: FailureType
    affected_subsystem: str = Field(min_length=1)
    severity: float = Field(ge=0.0, le=1.0)
    activation: Literal["start", "time", "condition"] = "start"
    activation_time_seconds: float | None = Field(default=None, ge=0.0)
    condition: StateCondition | None = None
    duration_seconds: float | None = Field(default=None, gt=0.0)
    permanent: bool = True
    ramp_duration_seconds: float = Field(default=0.0, ge=0.0)
    observable: bool = True
    detected: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_schedule(self) -> "FailureDefinition":
        if self.activation == "time" and self.activation_time_seconds is None:
            raise ValueError("time activation requires activation_time_seconds")
        if self.activation == "condition" and self.condition is None:
            raise ValueError("condition activation requires condition")
        if not self.permanent and self.duration_seconds is None:
            raise ValueError("non-permanent failure requires duration_seconds")
        return self


class ActiveFailure(BaseModel):
    """Ground-truth runtime failure state."""

    failure_id: str
    failure_type: FailureType
    affected_subsystem: str
    intensity: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    observable: bool
    detected: bool
    activation_time_seconds: float
    parameters: dict[str, Any]

