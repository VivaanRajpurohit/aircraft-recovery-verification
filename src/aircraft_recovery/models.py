"""Validated JSON boundary models for controllers and scenarios."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
UnitControl = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
Throttle = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
HealthFraction = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and non-finite numbers."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MissionPhase(str, Enum):
    """Supported high-level mission phases."""

    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    DIVERSION = "diversion"


class AircraftState(StrictModel):
    altitude_ft: Annotated[float, Field(ge=-1000.0, le=60000.0, allow_inf_nan=False)]
    airspeed_kts: Annotated[float, Field(ge=0.0, le=600.0, allow_inf_nan=False)]
    pitch_deg: Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)]
    roll_deg: Annotated[float, Field(ge=-180.0, le=180.0, allow_inf_nan=False)]
    yaw_deg: Annotated[float, Field(ge=-180.0, le=180.0, allow_inf_nan=False)]
    vertical_speed_fpm: Annotated[float, Field(ge=-20000.0, le=20000.0, allow_inf_nan=False)]
    heading_deg: Annotated[float, Field(ge=0.0, lt=360.0, allow_inf_nan=False)]


class FlightEnvelope(StrictModel):
    angle_of_attack_deg: Annotated[float, Field(ge=-20.0, le=40.0, allow_inf_nan=False)]
    load_factor_g: Annotated[float, Field(ge=-3.0, le=9.0, allow_inf_nan=False)]
    stall_margin_kts: FiniteFloat
    overspeed_margin_kts: FiniteFloat


class ControlHealth(StrictModel):
    elevator_effectiveness: HealthFraction = 1.0
    aileron_effectiveness: HealthFraction = 1.0
    rudder_effectiveness: HealthFraction = 1.0
    elevator_stuck: bool = False
    aileron_stuck: bool = False
    rudder_stuck: bool = False
    actuator_delay_ms: Annotated[float, Field(ge=0.0, le=10000.0, allow_inf_nan=False)] = 0.0


class EngineHealth(StrictModel):
    left_engine_thrust_available: HealthFraction = 1.0
    right_engine_thrust_available: HealthFraction = 1.0
    left_engine_failed: bool = False
    right_engine_failed: bool = False
    asymmetric_thrust: bool = False

    @model_validator(mode="after")
    def failures_have_no_thrust(self) -> "EngineHealth":
        if self.left_engine_failed and self.left_engine_thrust_available != 0.0:
            raise ValueError("left_engine_failed requires zero left thrust availability")
        if self.right_engine_failed and self.right_engine_thrust_available != 0.0:
            raise ValueError("right_engine_failed requires zero right thrust availability")
        return self


class Airport(StrictModel):
    airport_id: Annotated[str, Field(min_length=1, max_length=16)]
    distance_nm: Annotated[float, Field(ge=0.0, le=1000.0, allow_inf_nan=False)]
    bearing_deg: Annotated[float, Field(ge=0.0, lt=360.0, allow_inf_nan=False)]
    runway_heading_deg: Annotated[float, Field(ge=0.0, lt=360.0, allow_inf_nan=False)]
    runway_length_ft: Annotated[float, Field(gt=0.0, le=30000.0, allow_inf_nan=False)]
    terrain_clearance_ft: FiniteFloat
    within_glide_range: bool


class Navigation(StrictModel):
    latitude_deg: Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)]
    longitude_deg: Annotated[float, Field(ge=-180.0, le=180.0, allow_inf_nan=False)]
    nearest_airports: list[Airport] = Field(default_factory=list, max_length=20)
    data_valid: bool = True


class Environment(StrictModel):
    wind_speed_kts: Annotated[float, Field(ge=0.0, le=250.0, allow_inf_nan=False)]
    wind_direction_deg: Annotated[float, Field(ge=0.0, lt=360.0, allow_inf_nan=False)]
    turbulence_intensity: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    visibility_nm: Annotated[float, Field(ge=0.0, le=200.0, allow_inf_nan=False)]
    terrain_clearance_ft: FiniteFloat


class Mission(StrictModel):
    phase: MissionPhase
    emergency_declared: bool
    original_destination: Annotated[str, Field(min_length=1, max_length=16)]


class ControllerInput(StrictModel):
    """Complete serializable controller observation packet."""

    timestamp_seconds: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    aircraft_state: AircraftState
    flight_envelope: FlightEnvelope
    control_health: ControlHealth
    engine_health: EngineHealth
    navigation: Navigation
    environment: Environment
    mission: Mission
    active_failures: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)


class ControlCommands(StrictModel):
    elevator: UnitControl
    aileron: UnitControl
    rudder: UnitControl
    throttle_left: Throttle
    throttle_right: Throttle


class StabilizationTargets(StrictModel):
    target_pitch_deg: Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)]
    target_roll_deg: Annotated[float, Field(ge=-180.0, le=180.0, allow_inf_nan=False)]
    target_airspeed_kts: Annotated[float, Field(ge=0.0, le=600.0, allow_inf_nan=False)]
    target_altitude_ft: Annotated[float, Field(ge=-1000.0, le=60000.0, allow_inf_nan=False)]


class NavigationCommands(StrictModel):
    target_heading_deg: Annotated[float, Field(ge=0.0, lt=360.0, allow_inf_nan=False)]
    selected_airport: str | None = None
    selected_runway_heading_deg: Annotated[float, Field(ge=0.0, lt=360.0, allow_inf_nan=False)] | None = None
    approach_phase: Literal["none", "diversion", "approach", "landing"] = "none"


class SafetyDecision(StrictModel):
    status: Literal["not_checked", "accepted", "modified", "rejected", "fallback"]
    ai_action_accepted: bool
    fallback_activated: bool
    violated_constraints: list[str] = Field(default_factory=list)
    explanation: str


class ControllerOutput(StrictModel):
    """Complete serializable controller command packet."""

    timestamp_seconds: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    control_commands: ControlCommands
    stabilization_targets: StabilizationTargets
    navigation_commands: NavigationCommands
    emergency_mode: Literal["normal", "stabilize", "divert", "approach", "land"]
    safety_decision: SafetyDecision
    confidence: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

