"""Independent deterministic fallback recovery controller."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aircraft_recovery.config import SyntheticAirportConfig
from aircraft_recovery.controllers.emergency import EmergencyMode, EmergencyStateMachine
from aircraft_recovery.controllers.pid import PIDController
from aircraft_recovery.models import (
    ControlCommands,
    ControllerInput,
    ControllerOutput,
    NavigationCommands,
    SafetyDecision,
    StabilizationTargets,
)
from aircraft_recovery.navigation import DiversionPlan, DiversionPlanner


class PIDGains(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kp: float
    ki: float
    kd: float
    integral_limit: float = Field(gt=0.0)


class FallbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_step_seconds: float = Field(gt=0.0)
    target_airspeed_kts: float = Field(gt=0.0)
    stall_recovery_margin_kts: float = Field(gt=0.0)
    maximum_recovery_pitch_deg: float = Field(gt=0.0)
    minimum_mode_duration_seconds: float = Field(ge=0.0)
    pitch: PIDGains
    roll: PIDGains
    heading: PIDGains
    airspeed: PIDGains
    minimum_runway_ft: float = Field(gt=0.0)
    maximum_crosswind_kts: float = Field(gt=0.0)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _heading_error(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


class DeterministicFallbackController:
    """PID/rule recovery controller independent of any learned policy."""

    name = "phase2_deterministic_fallback"

    def __init__(
        self,
        config: FallbackConfig,
        airports: list[SyntheticAirportConfig],
    ) -> None:
        self.config = config
        self.airports = airports
        self.pitch_pid = PIDController(**config.pitch.model_dump())
        self.roll_pid = PIDController(**config.roll.model_dump())
        self.heading_pid = PIDController(**config.heading.model_dump())
        self.airspeed_pid = PIDController(
            **config.airspeed.model_dump(), output_min=-0.5, output_max=0.5
        )
        self.state_machine = EmergencyStateMachine(config.minimum_mode_duration_seconds)
        self.planner = DiversionPlanner(config.minimum_runway_ft, config.maximum_crosswind_kts)
        self.last_plan: DiversionPlan | None = None
        self.selected_airport: str | None = None
        self.activation_count = 0

    def reset(self) -> None:
        for controller in (self.pitch_pid, self.roll_pid, self.heading_pid, self.airspeed_pid):
            controller.reset()
        self.state_machine.reset()
        self.last_plan = None
        self.selected_airport = None
        self.activation_count = 0

    def act(self, observation: ControllerInput) -> ControllerOutput:
        self.activation_count += 1
        self.last_plan = self.planner.plan(observation, self.airports)
        selected_distance = next((
            airport.distance_nm for airport in observation.navigation.nearest_airports
            if airport.airport_id == self.last_plan.selected_airport
        ), None)
        mode = self.state_machine.update(
            observation,
            self.last_plan.selected_airport,
            selected_distance,
            no_reachable_airport=not self.last_plan.estimated_reachable,
        )
        state, envelope = observation.aircraft_state, observation.flight_envelope
        health, engines = observation.control_health, observation.engine_health
        target_pitch = 0.0
        target_speed = self.config.target_airspeed_kts
        if envelope.stall_margin_kts < self.config.stall_recovery_margin_kts:
            target_pitch = min(-5.0, state.pitch_deg - 5.0)
            target_speed += 10.0
        elif abs(state.pitch_deg) > self.config.maximum_recovery_pitch_deg:
            target_pitch = 0.0
        if mode == EmergencyMode.GLIDE:
            target_pitch = -3.0
            target_speed = max(target_speed, 1.25 * max(1.0, target_speed - envelope.stall_margin_kts))

        heading_target = self.last_plan.target_heading_deg if mode in {
            EmergencyMode.DIVERT, EmergencyMode.APPROACH, EmergencyMode.GLIDE
        } else state.heading_deg
        heading_error = _heading_error(heading_target, state.heading_deg)
        # The simulator changes ground-track heading through bank angle; rudder
        # changes yaw only. Convert navigation error into a bounded bank target.
        target_roll = _clip(0.35 * heading_error, -25.0, 25.0) if mode in {
            EmergencyMode.DIVERT, EmergencyMode.APPROACH, EmergencyMode.GLIDE
        } else 0.0
        elevator = self.pitch_pid.update(target_pitch - state.pitch_deg, self.config.time_step_seconds)
        aileron = self.roll_pid.update(target_roll - state.roll_deg, self.config.time_step_seconds)
        heading_control = self.heading_pid.update(
            heading_error, self.config.time_step_seconds
        )
        asymmetry = engines.right_engine_thrust_available - engines.left_engine_thrust_available
        rudder = _clip(heading_control - 0.45 * asymmetry, -1.0, 1.0)

        elevator = _clip(elevator / max(health.elevator_effectiveness, 0.25), -1.0, 1.0)
        aileron = _clip(aileron / max(health.aileron_effectiveness, 0.25), -1.0, 1.0)
        rudder = _clip(rudder / max(health.rudder_effectiveness, 0.25), -1.0, 1.0)
        throttle_adjustment = self.airspeed_pid.update(
            target_speed - state.airspeed_kts, self.config.time_step_seconds
        )
        throttle_base = _clip(0.55 + throttle_adjustment, 0.0, 1.0)
        throttle_left = 0.0 if engines.left_engine_failed else throttle_base
        throttle_right = 0.0 if engines.right_engine_failed else throttle_base
        if mode == EmergencyMode.GLIDE:
            throttle_left = throttle_right = 0.0

        output_mode = {
            EmergencyMode.NORMAL: "normal",
            EmergencyMode.FAILURE_DETECTED: "stabilize",
            EmergencyMode.STABILIZE: "stabilize",
            EmergencyMode.GLIDE: "divert",
            EmergencyMode.DIVERT: "divert",
            EmergencyMode.APPROACH: "approach",
            EmergencyMode.EMERGENCY_LANDING: "land",
            EmergencyMode.UNRECOVERABLE: "stabilize",
            EmergencyMode.TERMINATED: "stabilize",
        }[mode]
        navigation_active = mode != EmergencyMode.NORMAL
        self.selected_airport = self.last_plan.selected_airport if navigation_active else None
        approach_phase = "approach" if mode == EmergencyMode.APPROACH else "diversion" if mode in {
            EmergencyMode.GLIDE, EmergencyMode.DIVERT
        } else "none"
        return ControllerOutput(
            timestamp_seconds=observation.timestamp_seconds,
            control_commands=ControlCommands(
                elevator=elevator, aileron=aileron, rudder=rudder,
                throttle_left=throttle_left, throttle_right=throttle_right,
            ),
            stabilization_targets=StabilizationTargets(
                target_pitch_deg=target_pitch, target_roll_deg=target_roll,
                target_airspeed_kts=target_speed, target_altitude_ft=state.altitude_ft,
            ),
            navigation_commands=NavigationCommands(
                target_heading_deg=heading_target,
                selected_airport=self.selected_airport,
                selected_runway_heading_deg=(
                    self.last_plan.selected_runway_heading_deg if navigation_active else None
                ),
                approach_phase=approach_phase,
            ),
            emergency_mode=output_mode,
            safety_decision=SafetyDecision(
                status="fallback", ai_action_accepted=False, fallback_activated=True,
                violated_constraints=[],
                explanation=f"Deterministic fallback active in {mode.value} mode; no formal monitor is present.",
            ),
            confidence=self.last_plan.confidence,
        )
