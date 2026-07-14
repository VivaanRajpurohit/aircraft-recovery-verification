"""Time/state scheduled deterministic failure engine."""

from __future__ import annotations

from collections.abc import Mapping

from aircraft_recovery.failures.models import ActiveFailure, FailureDefinition, StateCondition


_OPERATORS = {
    "lt": lambda left, right: left < right,
    "le": lambda left, right: left <= right,
    "gt": lambda left, right: left > right,
    "ge": lambda left, right: left >= right,
    "eq": lambda left, right: left == right,
}


class FailureEngine:
    """Evaluate failure schedules without exposing hidden truth to controllers."""

    def __init__(self, definitions: list[FailureDefinition]) -> None:
        ids = [item.failure_id for item in definitions]
        if len(ids) != len(set(ids)):
            raise ValueError("Failure identifiers must be unique")
        self.definitions = definitions
        self._activated_at: dict[str, float] = {}

    def reset(self) -> None:
        self._activated_at.clear()

    def evaluate(self, time_seconds: float, true_state: Mapping[str, float]) -> list[ActiveFailure]:
        """Return current failures, including ramp intensity and ground truth."""
        active: list[ActiveFailure] = []
        for definition in self.definitions:
            if definition.failure_id not in self._activated_at and self._should_activate(
                definition, time_seconds, true_state
            ):
                self._activated_at[definition.failure_id] = time_seconds
            activated_at = self._activated_at.get(definition.failure_id)
            if activated_at is None:
                continue
            elapsed = max(0.0, time_seconds - activated_at)
            if not definition.permanent and definition.duration_seconds is not None:
                if elapsed >= definition.duration_seconds:
                    continue
            ramp = 1.0
            if definition.ramp_duration_seconds > 0.0:
                ramp = min(1.0, elapsed / definition.ramp_duration_seconds)
            active.append(ActiveFailure(
                failure_id=definition.failure_id,
                failure_type=definition.failure_type,
                affected_subsystem=definition.affected_subsystem,
                intensity=definition.severity * ramp,
                severity=definition.severity,
                observable=definition.observable,
                detected=definition.detected,
                activation_time_seconds=activated_at,
                parameters=definition.parameters,
            ))
        return active

    @staticmethod
    def _should_activate(
        definition: FailureDefinition,
        time_seconds: float,
        true_state: Mapping[str, float],
    ) -> bool:
        if definition.activation == "start":
            return True
        if definition.activation == "time":
            return time_seconds >= (definition.activation_time_seconds or 0.0)
        return FailureEngine._condition_matches(definition.condition, true_state)

    @staticmethod
    def _condition_matches(condition: StateCondition | None, state: Mapping[str, float]) -> bool:
        if condition is None or condition.field not in state:
            return False
        return bool(_OPERATORS[condition.operator](float(state[condition.field]), condition.value))

