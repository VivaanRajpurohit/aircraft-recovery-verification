"""Simple deterministic controller used to make Phase 1 runnable."""

from __future__ import annotations

from aircraft_recovery.models import (
    ControlCommands,
    ControllerInput,
    ControllerOutput,
    NavigationCommands,
    SafetyDecision,
    StabilizationTargets,
)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class StabilizationController:
    """Proportional attitude/airspeed controller; not the Phase 2 fallback."""

    name = "phase1_stabilization_controller"

    def __init__(self, target_airspeed_kts: float = 140.0) -> None:
        self.target_airspeed_kts = target_airspeed_kts

    def act(self, observation: ControllerInput) -> ControllerOutput:
        state = observation.aircraft_state
        elevator = _clip(-0.04 * state.pitch_deg, -1.0, 1.0)
        aileron = _clip(-0.03 * state.roll_deg, -1.0, 1.0)
        rudder = _clip(-0.03 * state.yaw_deg, -1.0, 1.0)
        throttle = _clip(0.5 + 0.012 * (self.target_airspeed_kts - state.airspeed_kts), 0.0, 1.0)
        airport = observation.navigation.nearest_airports[0] if observation.navigation.nearest_airports else None
        return ControllerOutput(
            timestamp_seconds=observation.timestamp_seconds,
            control_commands=ControlCommands(
                elevator=elevator, aileron=aileron, rudder=rudder,
                throttle_left=throttle, throttle_right=throttle,
            ),
            stabilization_targets=StabilizationTargets(
                target_pitch_deg=0.0, target_roll_deg=0.0,
                target_airspeed_kts=self.target_airspeed_kts,
                target_altitude_ft=state.altitude_ft,
            ),
            navigation_commands=NavigationCommands(
                target_heading_deg=state.heading_deg,
                selected_airport=airport.airport_id if airport else None,
                selected_runway_heading_deg=airport.runway_heading_deg if airport else None,
                approach_phase="none",
            ),
            emergency_mode="normal",
            safety_decision=SafetyDecision(
                status="not_checked", ai_action_accepted=True, fallback_activated=False,
                violated_constraints=[],
                explanation="Phase 1 controller has no formal runtime safety monitor.",
            ),
            confidence=1.0,
        )

