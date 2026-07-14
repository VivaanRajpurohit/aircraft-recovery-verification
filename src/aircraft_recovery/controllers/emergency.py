"""Explicit hysteretic emergency-mode state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from aircraft_recovery.models import ControllerInput


class EmergencyMode(str, Enum):
    NORMAL = "NORMAL"
    FAILURE_DETECTED = "FAILURE_DETECTED"
    STABILIZE = "STABILIZE"
    GLIDE = "GLIDE"
    DIVERT = "DIVERT"
    APPROACH = "APPROACH"
    EMERGENCY_LANDING = "EMERGENCY_LANDING"
    UNRECOVERABLE = "UNRECOVERABLE"
    TERMINATED = "TERMINATED"


ALLOWED_TRANSITIONS: dict[EmergencyMode, set[EmergencyMode]] = {
    EmergencyMode.NORMAL: {EmergencyMode.FAILURE_DETECTED, EmergencyMode.TERMINATED},
    EmergencyMode.FAILURE_DETECTED: {EmergencyMode.STABILIZE, EmergencyMode.UNRECOVERABLE},
    EmergencyMode.STABILIZE: {EmergencyMode.GLIDE, EmergencyMode.DIVERT, EmergencyMode.UNRECOVERABLE},
    EmergencyMode.GLIDE: {EmergencyMode.DIVERT, EmergencyMode.APPROACH, EmergencyMode.UNRECOVERABLE},
    EmergencyMode.DIVERT: {EmergencyMode.APPROACH, EmergencyMode.STABILIZE, EmergencyMode.UNRECOVERABLE},
    EmergencyMode.APPROACH: {EmergencyMode.EMERGENCY_LANDING, EmergencyMode.STABILIZE, EmergencyMode.UNRECOVERABLE},
    EmergencyMode.EMERGENCY_LANDING: {EmergencyMode.TERMINATED, EmergencyMode.UNRECOVERABLE},
    EmergencyMode.UNRECOVERABLE: {EmergencyMode.TERMINATED},
    EmergencyMode.TERMINATED: set(),
}


@dataclass(frozen=True)
class ModeTransition:
    previous_mode: str
    new_mode: str
    timestamp_seconds: float
    trigger: str
    relevant_state_values: dict[str, float]
    active_failures: tuple[str, ...]
    selected_diversion_airport: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EmergencyStateMachine:
    """Deterministic emergency modes with minimum dwell-time hysteresis."""

    def __init__(self, minimum_mode_duration_seconds: float = 2.0) -> None:
        self.minimum_mode_duration_seconds = minimum_mode_duration_seconds
        self.mode = EmergencyMode.NORMAL
        self.entered_at_seconds = 0.0
        self.transitions: list[ModeTransition] = []

    def reset(self) -> None:
        self.mode = EmergencyMode.NORMAL
        self.entered_at_seconds = 0.0
        self.transitions.clear()

    def update(
        self,
        observation: ControllerInput,
        selected_airport: str | None,
        airport_distance_nm: float | None,
        no_reachable_airport: bool = False,
    ) -> EmergencyMode:
        state, envelope = observation.aircraft_state, observation.flight_envelope
        now = observation.timestamp_seconds
        elapsed = now - self.entered_at_seconds
        unstable = (
            envelope.stall_margin_kts < 12.0 or abs(state.roll_deg) > 45.0
            or abs(state.pitch_deg) > 20.0 or envelope.angle_of_attack_deg > 14.0
        )
        failures = bool(observation.active_failures) or observation.mission.emergency_declared
        engines_out = (
            observation.engine_health.left_engine_thrust_available <= 0.01
            and observation.engine_health.right_engine_thrust_available <= 0.01
        )
        target: EmergencyMode | None = None
        trigger = ""
        if self.mode == EmergencyMode.NORMAL and (failures or unstable):
            target, trigger = EmergencyMode.FAILURE_DETECTED, "failure_or_envelope_anomaly"
        elif self.mode == EmergencyMode.FAILURE_DETECTED and elapsed >= self.minimum_mode_duration_seconds:
            target, trigger = EmergencyMode.STABILIZE, "assessment_dwell_complete"
        elif self.mode == EmergencyMode.STABILIZE and no_reachable_airport and state.altitude_ft < 1200.0:
            target, trigger = EmergencyMode.UNRECOVERABLE, "no_reachable_airport_at_low_altitude"
        elif self.mode == EmergencyMode.STABILIZE and elapsed >= self.minimum_mode_duration_seconds and not unstable:
            target = EmergencyMode.GLIDE if engines_out else EmergencyMode.DIVERT
            trigger = "stable_envelope_established"
        elif self.mode == EmergencyMode.GLIDE and selected_airport and elapsed >= self.minimum_mode_duration_seconds:
            target, trigger = EmergencyMode.DIVERT, "glide_diversion_selected"
        elif self.mode == EmergencyMode.DIVERT and airport_distance_nm is not None and airport_distance_nm <= 3.0:
            target, trigger = EmergencyMode.APPROACH, "within_approach_distance"
        elif self.mode == EmergencyMode.DIVERT and unstable and elapsed >= self.minimum_mode_duration_seconds:
            target, trigger = EmergencyMode.STABILIZE, "envelope_departure_during_diversion"
        elif self.mode == EmergencyMode.APPROACH and state.altitude_ft <= 500.0:
            target, trigger = EmergencyMode.EMERGENCY_LANDING, "landing_gate_reached"
        if target is not None:
            self._transition(target, trigger, observation, selected_airport)
        return self.mode

    def terminate(self, observation: ControllerInput, trigger: str) -> None:
        if EmergencyMode.TERMINATED in ALLOWED_TRANSITIONS[self.mode]:
            self._transition(EmergencyMode.TERMINATED, trigger, observation, None)

    def _transition(
        self,
        new_mode: EmergencyMode,
        trigger: str,
        observation: ControllerInput,
        selected_airport: str | None,
    ) -> None:
        if new_mode not in ALLOWED_TRANSITIONS[self.mode]:
            raise ValueError(f"Transition {self.mode.value} -> {new_mode.value} is not allowed")
        state = observation.aircraft_state
        transition = ModeTransition(
            previous_mode=self.mode.value,
            new_mode=new_mode.value,
            timestamp_seconds=observation.timestamp_seconds,
            trigger=trigger,
            relevant_state_values={
                "altitude_ft": state.altitude_ft,
                "airspeed_kts": state.airspeed_kts,
                "pitch_deg": state.pitch_deg,
                "roll_deg": state.roll_deg,
                "vertical_speed_fpm": state.vertical_speed_fpm,
            },
            active_failures=tuple(observation.active_failures),
            selected_diversion_airport=selected_airport,
        )
        self.transitions.append(transition)
        self.mode = new_mode
        self.entered_at_seconds = observation.timestamp_seconds
