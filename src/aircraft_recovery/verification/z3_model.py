"""Z3 encoding of the bounded one-step monitor abstraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Protocol

import z3

from aircraft_recovery.models import ControlCommands, ControllerInput
from aircraft_recovery.safety.config import SafetyConfig


def _real(value: float) -> z3.RatNumRef:
    return z3.RealVal(str(float(value)))


@dataclass(frozen=True)
class SolverCheck:
    status: str
    duration_ms: float
    violated_properties: tuple[str, ...]
    counterexample: dict[str, str] | None
    timeout: bool = False
    unknown: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetySolverBackend(Protocol):
    def check_action(
        self,
        observation: ControllerInput,
        commands: ControlCommands,
        previous_commands: ControlCommands,
    ) -> SolverCheck: ...


class Z3SafetySolver:
    """Search for a bounded disturbance counterexample to candidate safety."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def check_action(
        self,
        observation: ControllerInput,
        commands: ControlCommands,
        previous_commands: ControlCommands,
    ) -> SolverCheck:
        started = perf_counter()
        solver = z3.Solver()
        solver.set(timeout=self.config.monitor.solver_timeout_ms)
        speed_d, pitch_d, roll_d, aoa_d, load_d, terrain_loss = z3.Reals(
            "speed_disturbance pitch_disturbance roll_disturbance aoa_disturbance "
            "load_disturbance terrain_loss"
        )
        dynamics, thresholds = self.config.dynamics, self.config.thresholds
        solver.add(
            speed_d >= -_real(dynamics.maximum_speed_disturbance_kts),
            speed_d <= _real(dynamics.maximum_speed_disturbance_kts),
            pitch_d >= -_real(dynamics.maximum_pitch_disturbance_deg),
            pitch_d <= _real(dynamics.maximum_pitch_disturbance_deg),
            roll_d >= -_real(dynamics.maximum_roll_disturbance_deg),
            roll_d <= _real(dynamics.maximum_roll_disturbance_deg),
            aoa_d >= -_real(dynamics.maximum_aoa_disturbance_deg),
            aoa_d <= _real(dynamics.maximum_aoa_disturbance_deg),
            load_d >= -_real(dynamics.maximum_load_disturbance_g),
            load_d <= _real(dynamics.maximum_load_disturbance_g),
            terrain_loss >= 0,
            terrain_loss <= _real(dynamics.maximum_terrain_loss_ft),
        )
        state, envelope = observation.aircraft_state, observation.flight_envelope
        dt = dynamics.time_step_seconds
        average_throttle = (commands.throttle_left + commands.throttle_right) / 2.0
        airspeed_next = (
            _real(state.airspeed_kts)
            + _real(dt * dynamics.thrust_gain_kts_per_second * (average_throttle - 0.5))
            - _real(dt * dynamics.drag_gain_per_second * (state.airspeed_kts - dynamics.cruise_speed_kts))
            + speed_d
        )
        pitch_next = _real(state.pitch_deg + dt * dynamics.pitch_rate_deg_per_second * commands.elevator * observation.control_health.elevator_effectiveness) + pitch_d
        roll_next = _real(state.roll_deg + dt * dynamics.roll_rate_deg_per_second * commands.aileron * observation.control_health.aileron_effectiveness) + roll_d
        aoa_next = _real(envelope.angle_of_attack_deg + dt * dynamics.angle_of_attack_gain * commands.elevator) + aoa_d
        load_next = _real(envelope.load_factor_g + dynamics.load_factor_gain * abs(commands.aileron)) + load_d
        terrain_next = _real(observation.environment.terrain_clearance_ft + state.vertical_speed_fpm * dt / 60.0) - terrain_loss
        minimum_speed = thresholds.stall_speed_kts + thresholds.stall_margin_kts
        maximum_speed = thresholds.overspeed_kts - thresholds.overspeed_margin_kts
        property_expressions: dict[str, z3.BoolRef] = {
            "stall_margin": airspeed_next < _real(minimum_speed),
            "overspeed_margin": airspeed_next > _real(maximum_speed),
            "bank_angle": z3.Abs(roll_next) > _real(thresholds.maximum_absolute_bank_deg),
            "pitch_range": z3.Or(pitch_next < _real(thresholds.minimum_pitch_deg), pitch_next > _real(thresholds.maximum_pitch_deg)),
            "angle_of_attack": aoa_next > _real(thresholds.maximum_angle_of_attack_deg),
            "load_factor": z3.Or(load_next < _real(thresholds.minimum_load_factor_g), load_next > _real(thresholds.maximum_load_factor_g)),
            "terrain_clearance": terrain_next <= _real(thresholds.minimum_terrain_clearance_ft),
        }
        rate_limit = thresholds.maximum_control_rate_per_second * dt
        for name in ("elevator", "aileron", "rudder", "throttle_left", "throttle_right"):
            property_expressions[f"command_rate_{name}"] = _real(abs(getattr(commands, name) - getattr(previous_commands, name))) > _real(rate_limit)
        solver.add(z3.Or(*property_expressions.values()))
        result = solver.check()
        duration = (perf_counter() - started) * 1000.0
        if result == z3.unsat:
            return SolverCheck("safe", duration, (), None)
        if result == z3.unknown:
            reason = solver.reason_unknown()
            timed_out = "timeout" in reason.lower()
            return SolverCheck("unknown", duration, (), {"reason": reason}, timed_out, True)
        model = solver.model()
        violated = tuple(
            name for name, expression in property_expressions.items()
            if z3.is_true(model.eval(expression, model_completion=True))
        )
        counterexample = {
            str(variable): str(model.eval(variable, model_completion=True))
            for variable in (speed_d, pitch_d, roll_d, aoa_d, load_d, terrain_loss)
        }
        return SolverCheck("unsafe", duration, violated, counterexample)

