"""Conservative one-step runtime safety monitor and action projection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any

from aircraft_recovery.models import ControlCommands, ControllerInput
from aircraft_recovery.safety.config import SafetyConfig
from aircraft_recovery.verification.z3_model import SafetySolverBackend, SolverCheck, Z3SafetySolver


COMMAND_NAMES = ("elevator", "aileron", "rudder", "throttle_left", "throttle_right")


@dataclass(frozen=True)
class MonitorDecision:
    status: str
    solver_result: str
    solver_duration_ms: float
    monitor_duration_ms: float
    original_action: ControlCommands
    final_action: ControlCommands
    modification_magnitude: float
    violated_properties: tuple[str, ...]
    fallback_activated: bool
    timeout: bool
    unknown: bool
    counterexample: dict[str, str] | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "solver_result": self.solver_result,
            "solver_duration_ms": self.solver_duration_ms,
            "monitor_duration_ms": self.monitor_duration_ms,
            "original_action": self.original_action.model_dump(mode="json"),
            "final_action": self.final_action.model_dump(mode="json"),
            "modification_magnitude": self.modification_magnitude,
            "violated_properties": list(self.violated_properties),
            "fallback_activated": self.fallback_activated,
            "timeout": self.timeout,
            "unknown": self.unknown,
            "counterexample": self.counterexample,
            "reason": self.reason,
        }


class RuntimeSafetyMonitor:
    """Accept, project, or conservatively fall back under a bounded Z3 model."""

    def __init__(
        self,
        config: SafetyConfig,
        solver_backend: SafetySolverBackend | None = None,
    ) -> None:
        self.config = config
        self.solver_backend = solver_backend or Z3SafetySolver(config)
        self.previous_commands = ControlCommands(
            elevator=0.0, aileron=0.0, rudder=0.0,
            throttle_left=0.5, throttle_right=0.5,
        )

    def reset(self) -> None:
        self.previous_commands = ControlCommands(
            elevator=0.0, aileron=0.0, rudder=0.0,
            throttle_left=0.5, throttle_right=0.5,
        )

    def decide(
        self,
        observation: ControllerInput,
        proposed: ControlCommands,
        fallback: ControlCommands,
    ) -> MonitorDecision:
        started = perf_counter()
        invalid_reason = self._invalid_state_reason(observation)
        if invalid_reason:
            return self._fallback_decision(
                started, proposed, fallback, "invalid", (), invalid_reason, None
            )
        proposed_check = self.solver_backend.check_action(
            observation, proposed, self.previous_commands
        )
        if proposed_check.unknown:
            return self._fallback_decision(
                started, proposed, fallback, "unknown", (),
                "Solver returned unknown or timed out; conservative fallback activated.",
                proposed_check,
            )
        if proposed_check.status == "safe":
            self.previous_commands = proposed
            return MonitorDecision(
                status="accept", solver_result="safe",
                solver_duration_ms=proposed_check.duration_ms,
                monitor_duration_ms=(perf_counter() - started) * 1000.0,
                original_action=proposed, final_action=proposed,
                modification_magnitude=0.0, violated_properties=(),
                fallback_activated=False, timeout=False, unknown=False,
                counterexample=None, reason="No bounded one-step counterexample exists.",
            )
        violations = set(proposed_check.violated_properties)
        solver_duration = proposed_check.duration_ms
        for candidate in self._projection_candidates(observation, proposed):
            check = self.solver_backend.check_action(
                observation, candidate, self.previous_commands
            )
            solver_duration += check.duration_ms
            if check.unknown:
                return self._fallback_decision(
                    started, proposed, fallback, "unknown", tuple(sorted(violations)),
                    "Solver returned unknown during projection; conservative fallback activated.",
                    check, solver_duration,
                )
            violations.update(check.violated_properties)
            if check.status == "safe":
                self.previous_commands = candidate
                return MonitorDecision(
                    status="modify", solver_result="safe_after_projection",
                    solver_duration_ms=solver_duration,
                    monitor_duration_ms=(perf_counter() - started) * 1000.0,
                    original_action=proposed, final_action=candidate,
                    modification_magnitude=self._distance(proposed, candidate),
                    violated_properties=tuple(sorted(set(proposed_check.violated_properties))),
                    fallback_activated=False, timeout=False, unknown=False,
                    counterexample=proposed_check.counterexample,
                    reason="Proposed action had a bounded counterexample; nearest enumerated safe candidate selected.",
                )
        return self._fallback_decision(
            started, proposed, fallback, "unsafe", tuple(sorted(violations)),
            "No enumerated safe action was found; deterministic fallback activated.",
            proposed_check, solver_duration,
        )

    def _fallback_decision(
        self,
        started: float,
        proposed: ControlCommands,
        fallback: ControlCommands,
        solver_result: str,
        violations: tuple[str, ...],
        reason: str,
        check: SolverCheck | None,
        solver_duration_ms: float | None = None,
    ) -> MonitorDecision:
        self.previous_commands = fallback
        unknown = check.unknown if check else False
        timeout = check.timeout if check else False
        return MonitorDecision(
            status="unknown_fallback" if unknown else "reject_fallback",
            solver_result=solver_result,
            solver_duration_ms=(solver_duration_ms if solver_duration_ms is not None else check.duration_ms if check else 0.0),
            monitor_duration_ms=(perf_counter() - started) * 1000.0,
            original_action=proposed, final_action=fallback,
            modification_magnitude=self._distance(proposed, fallback),
            violated_properties=violations, fallback_activated=True,
            timeout=timeout, unknown=unknown,
            counterexample=check.counterexample if check else None,
            reason=reason,
        )

    def _invalid_state_reason(self, observation: ControllerInput) -> str | None:
        values = list(observation_to_scalars(observation).values())
        if self.config.monitor.reject_nonfinite_input and not all(math.isfinite(value) for value in values):
            return "nonfinite_observation"
        if self.config.monitor.reject_invalid_navigation and not observation.navigation.data_valid:
            return "navigation_data_invalid"
        state, envelope, bounds = (
            observation.aircraft_state,
            observation.flight_envelope,
            self.config.checked_state_bounds,
        )
        checks = {
            "airspeed_outside_checked_bounds": bounds.minimum_airspeed_kts <= state.airspeed_kts <= bounds.maximum_airspeed_kts,
            "pitch_outside_checked_bounds": bounds.minimum_pitch_deg <= state.pitch_deg <= bounds.maximum_pitch_deg,
            "roll_outside_checked_bounds": abs(state.roll_deg) <= bounds.maximum_absolute_roll_deg,
            "aoa_outside_checked_bounds": bounds.minimum_angle_of_attack_deg <= envelope.angle_of_attack_deg <= bounds.maximum_angle_of_attack_deg,
            "load_outside_checked_bounds": bounds.minimum_load_factor_g <= envelope.load_factor_g <= bounds.maximum_load_factor_g,
            "terrain_outside_checked_bounds": observation.environment.terrain_clearance_ft >= bounds.minimum_terrain_clearance_ft,
        }
        return next((name for name, valid in checks.items() if not valid), None)

    def _projection_candidates(
        self,
        observation: ControllerInput,
        proposed: ControlCommands,
    ) -> list[ControlCommands]:
        rate_limited = self._rate_limit(proposed)
        state, envelope = observation.aircraft_state, observation.flight_envelope
        recovery = ControlCommands(
            elevator=-0.6 if envelope.stall_margin_kts < self.config.thresholds.stall_margin_kts or envelope.angle_of_attack_deg > self.config.thresholds.maximum_angle_of_attack_deg else -0.15 * _sign(state.pitch_deg),
            aileron=-0.6 * _sign(state.roll_deg),
            rudder=-0.3 * _sign(state.yaw_deg),
            throttle_left=1.0 if not observation.engine_health.left_engine_failed else 0.0,
            throttle_right=1.0 if not observation.engine_health.right_engine_failed else 0.0,
        )
        candidates = [rate_limited]
        for blend in self.config.monitor.projection_blend_factors:
            values = {
                name: (1.0 - blend) * getattr(rate_limited, name) + blend * getattr(recovery, name)
                for name in COMMAND_NAMES
            }
            candidates.append(self._rate_limit(ControlCommands(**values)))
        unique: list[ControlCommands] = []
        seen: set[tuple[float, ...]] = set()
        for candidate in sorted(candidates, key=lambda value: self._distance(proposed, value)):
            key = tuple(round(getattr(candidate, name), 9) for name in COMMAND_NAMES)
            if key not in seen and self._distance(proposed, candidate) > 1e-12:
                seen.add(key)
                unique.append(candidate)
        return unique

    def _rate_limit(self, commands: ControlCommands) -> ControlCommands:
        maximum_delta = (
            self.config.thresholds.maximum_control_rate_per_second
            * self.config.dynamics.time_step_seconds
        )
        values = {}
        for name in COMMAND_NAMES:
            previous = getattr(self.previous_commands, name)
            values[name] = max(previous - maximum_delta, min(previous + maximum_delta, getattr(commands, name)))
        return ControlCommands(**values)

    @staticmethod
    def _distance(left: ControlCommands, right: ControlCommands) -> float:
        return math.sqrt(sum((getattr(left, name) - getattr(right, name)) ** 2 for name in COMMAND_NAMES))


def _sign(value: float) -> float:
    return 1.0 if value > 0.0 else -1.0 if value < 0.0 else 0.0


def observation_to_scalars(observation: ControllerInput) -> dict[str, float]:
    state, envelope = observation.aircraft_state, observation.flight_envelope
    return {
        "airspeed_kts": state.airspeed_kts,
        "pitch_deg": state.pitch_deg,
        "roll_deg": state.roll_deg,
        "angle_of_attack_deg": envelope.angle_of_attack_deg,
        "load_factor_g": envelope.load_factor_g,
        "terrain_clearance_ft": observation.environment.terrain_clearance_ft,
    }

