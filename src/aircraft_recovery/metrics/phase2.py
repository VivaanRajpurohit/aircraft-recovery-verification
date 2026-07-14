"""Phase 2 metrics compatible with later paired-controller experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aircraft_recovery.models import ControllerInput, ControllerOutput


@dataclass
class Phase2Metrics:
    """Incrementally aggregate recovery, envelope, failure, and latency data."""

    dt_seconds: float
    first_failure_activation_seconds: float | None = None
    first_failure_detection_seconds: float | None = None
    stabilization_time_seconds: float | None = None
    diversion_selection_time_seconds: float | None = None
    outside_envelope_seconds: float = 0.0
    minimum_airspeed_kts: float = float("inf")
    minimum_stall_margin_kts: float = float("inf")
    maximum_airspeed_kts: float = float("-inf")
    maximum_angle_of_attack_deg: float = float("-inf")
    maximum_absolute_roll_deg: float = 0.0
    maximum_absolute_pitch_deg: float = 0.0
    maximum_descent_rate_fpm: float = 0.0
    minimum_terrain_clearance_ft: float = float("inf")
    fallback_activation_count: int = 0
    controller_latencies_ms: list[float] = field(default_factory=list)
    failure_ids: set[str] = field(default_factory=set)
    _fallback_active: bool = False

    def update(
        self,
        observation: ControllerInput,
        action: ControllerOutput,
        ground_truth: dict[str, Any] | None,
        controller_latency_ms: float,
    ) -> None:
        state, envelope, environment = (
            observation.aircraft_state,
            observation.flight_envelope,
            observation.environment,
        )
        self.minimum_airspeed_kts = min(self.minimum_airspeed_kts, state.airspeed_kts)
        self.minimum_stall_margin_kts = min(self.minimum_stall_margin_kts, envelope.stall_margin_kts)
        self.maximum_airspeed_kts = max(self.maximum_airspeed_kts, state.airspeed_kts)
        self.maximum_angle_of_attack_deg = max(self.maximum_angle_of_attack_deg, envelope.angle_of_attack_deg)
        self.maximum_absolute_roll_deg = max(self.maximum_absolute_roll_deg, abs(state.roll_deg))
        self.maximum_absolute_pitch_deg = max(self.maximum_absolute_pitch_deg, abs(state.pitch_deg))
        self.maximum_descent_rate_fpm = max(self.maximum_descent_rate_fpm, max(0.0, -state.vertical_speed_fpm))
        self.minimum_terrain_clearance_ft = min(self.minimum_terrain_clearance_ft, environment.terrain_clearance_ft)
        outside = (
            envelope.stall_margin_kts < 15.0 or envelope.overspeed_margin_kts < 15.0
            or abs(state.roll_deg) > 60.0 or abs(state.pitch_deg) > 25.0
            or envelope.angle_of_attack_deg > 14.0
        )
        if outside:
            self.outside_envelope_seconds += self.dt_seconds
        if action.safety_decision.fallback_activated and not self._fallback_active:
            self.fallback_activation_count += 1
        self._fallback_active = action.safety_decision.fallback_activated
        self.controller_latencies_ms.append(controller_latency_ms)
        if observation.active_failures and self.first_failure_detection_seconds is None:
            self.first_failure_detection_seconds = observation.timestamp_seconds
        if action.navigation_commands.selected_airport and self.diversion_selection_time_seconds is None:
            self.diversion_selection_time_seconds = observation.timestamp_seconds
        if ground_truth:
            failures = ground_truth.get("ground_truth_failures", [])
            for failure in failures:
                self.failure_ids.add(str(failure["failure_id"]))
                activated = float(failure["activation_time_seconds"])
                if self.first_failure_activation_seconds is None or activated < self.first_failure_activation_seconds:
                    self.first_failure_activation_seconds = activated
            stable = envelope.stall_margin_kts >= 15.0 and abs(state.roll_deg) < 10.0 and abs(state.pitch_deg) < 8.0
            if failures and stable and self.stabilization_time_seconds is None:
                self.stabilization_time_seconds = observation.timestamp_seconds

    def summary(
        self,
        termination_reason: str,
        mode_transition_count: int,
        selected_airport_feasibility: str | None,
        final_distance_nm: float | None,
    ) -> dict[str, Any]:
        activation_to_stable = None
        if self.first_failure_activation_seconds is not None and self.stabilization_time_seconds is not None:
            activation_to_stable = self.stabilization_time_seconds - self.first_failure_activation_seconds
        latency = self.controller_latencies_ms
        successful = termination_reason in {
            "stable_recovery_achieved", "diversion_target_reached", "approach_completed", "duration_complete"
        }
        return {
            "failure_activation_time_seconds": self.first_failure_activation_seconds,
            "failure_detection_time_seconds": self.first_failure_detection_seconds,
            "time_from_failure_activation_to_stabilization_seconds": activation_to_stable,
            "time_outside_envelope_seconds": self.outside_envelope_seconds,
            "minimum_airspeed_kts": self.minimum_airspeed_kts,
            "minimum_stall_margin_kts": self.minimum_stall_margin_kts,
            "maximum_airspeed_kts": self.maximum_airspeed_kts,
            "maximum_angle_of_attack_deg": self.maximum_angle_of_attack_deg,
            "maximum_absolute_roll_deg": self.maximum_absolute_roll_deg,
            "maximum_absolute_pitch_deg": self.maximum_absolute_pitch_deg,
            "maximum_descent_rate_fpm": self.maximum_descent_rate_fpm,
            "minimum_terrain_clearance_ft": self.minimum_terrain_clearance_ft,
            "mode_transition_count": mode_transition_count,
            "fallback_activation_count": self.fallback_activation_count,
            "diversion_selection_time_seconds": self.diversion_selection_time_seconds,
            "selected_airport_feasibility": selected_airport_feasibility,
            "final_distance_from_selected_airport_nm": final_distance_nm,
            "recovery_result": "successful" if successful else "unsuccessful",
            "active_failure_count": len(self.failure_ids),
            "concurrent_failure_combination": sorted(self.failure_ids),
            "controller_latency_ms_mean": sum(latency) / len(latency) if latency else None,
            "controller_latency_ms_max": max(latency) if latency else None,
        }
