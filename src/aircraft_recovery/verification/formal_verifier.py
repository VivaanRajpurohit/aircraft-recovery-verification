"""Structured bounded Z3 verification obligations and examples."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import z3

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers.emergency import ALLOWED_TRANSITIONS, EmergencyMode
from aircraft_recovery.models import ControlCommands
from aircraft_recovery.safety.config import SafetyConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.verification.z3_model import Z3SafetySolver


def _prove_unsat(name: str, constraints: list[z3.BoolRef], config: SafetyConfig) -> dict[str, Any]:
    solver = z3.Solver()
    solver.set(timeout=config.monitor.solver_timeout_ms)
    solver.add(*constraints)
    started = perf_counter()
    result = solver.check()
    duration = (perf_counter() - started) * 1000.0
    counterexample = None
    if result == z3.sat:
        model = solver.model()
        counterexample = {str(item): str(model[item]) for item in model.decls()}
    return {
        "property_name": name,
        "verification_status": "verified_within_bounds" if result == z3.unsat else "failed" if result == z3.sat else "unknown",
        "solver_result": str(result),
        "solver_duration_ms": duration,
        "timeout": result == z3.unknown and "timeout" in solver.reason_unknown().lower(),
        "unknown": result == z3.unknown,
        "counterexample": counterexample,
        "verified_bounds": config.checked_state_bounds.model_dump(mode="json"),
        "assumptions": config.assumptions,
        "limitations": config.limitations,
    }


def run_formal_verification(
    config: SafetyConfig,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Discharge logic invariants and emit safe/unsafe bounded examples."""
    properties: list[dict[str, Any]] = []
    elevator, aileron, rudder, throttle_left, throttle_right = z3.Reals(
        "elevator aileron rudder throttle_left throttle_right"
    )
    action_bounds = z3.And(
        elevator >= -1, elevator <= 1, aileron >= -1, aileron <= 1,
        rudder >= -1, rudder <= 1, throttle_left >= 0, throttle_left <= 1,
        throttle_right >= 0, throttle_right <= 1,
    )
    outside_action_bounds = z3.Or(
        elevator < -1, elevator > 1, aileron < -1, aileron > 1,
        rudder < -1, rudder > 1, throttle_left < 0, throttle_left > 1,
        throttle_right < 0, throttle_right > 1,
    )
    properties.append(_prove_unsat(
        "bounded_output_cannot_exceed_command_limits",
        [action_bounds, outside_action_bounds], config,
    ))
    invalid, violation, accepted, rejected, fallback, unknown = z3.Bools(
        "invalid violation accepted rejected fallback unknown"
    )
    properties.extend([
        _prove_unsat("invalid_input_cannot_be_accepted", [z3.Implies(invalid, z3.Not(accepted)), invalid, accepted], config),
        _prove_unsat("unsafe_action_cannot_be_accepted", [z3.Implies(violation, z3.Not(accepted)), violation, accepted], config),
        _prove_unsat("rejection_requires_fallback", [z3.Implies(rejected, fallback), rejected, z3.Not(fallback)], config),
        _prove_unsat("unknown_requires_fallback", [z3.Implies(unknown, fallback), unknown, z3.Not(fallback)], config),
        _prove_unsat("accepted_action_has_no_encoded_violation", [accepted == z3.Not(violation), accepted, violation], config),
    ])
    mode_values = {mode: index for index, mode in enumerate(EmergencyMode)}
    current, following = z3.Ints("current_mode following_mode")
    allowed = z3.Or(*[
        z3.And(current == mode_values[source], following == mode_values[target])
        for source, targets in ALLOWED_TRANSITIONS.items() for target in targets
    ])
    properties.extend([
        _prove_unsat(
            "unrecoverable_cannot_transition_to_normal",
            [allowed, current == mode_values[EmergencyMode.UNRECOVERABLE], following == mode_values[EmergencyMode.NORMAL]],
            config,
        ),
        _prove_unsat(
            "terminated_has_no_operational_transition",
            [allowed, current == mode_values[EmergencyMode.TERMINATED]],
            config,
        ),
    ])

    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase1_nominal.yaml", ScenarioConfig)
    observation = SimpleAircraftSimulator(aircraft, scenario).reset(scenario.seed)
    # Runtime monitoring retains its strict configured timeout.  Offline report
    # examples use a modestly larger bound so host scheduling jitter does not
    # turn the deterministic witness regression into a flaky UNKNOWN result.
    example_config = config.model_copy(deep=True)
    example_config.monitor.solver_timeout_ms = max(config.monitor.solver_timeout_ms, 250)
    solver = Z3SafetySolver(example_config)
    previous = ControlCommands(
        elevator=0.0, aileron=0.0, rudder=0.0, throttle_left=0.5, throttle_right=0.5
    )
    safe_commands = previous
    safe_example = solver.check_action(observation, safe_commands, previous)
    unsafe_observation = observation.model_copy(deep=True)
    unsafe_observation.aircraft_state.airspeed_kts = config.thresholds.stall_speed_kts + config.thresholds.stall_margin_kts - 2.0
    unsafe_observation.flight_envelope.stall_margin_kts = unsafe_observation.aircraft_state.airspeed_kts - config.thresholds.stall_speed_kts
    unsafe_commands = ControlCommands(
        elevator=0.5, aileron=0.0, rudder=0.0, throttle_left=0.0, throttle_right=0.0
    )
    unsafe_example = solver.check_action(unsafe_observation, unsafe_commands, previous)
    report = {
        "report_format_version": 1,
        "configuration_name": config.configuration_name,
        "example_solver_timeout_ms": example_config.monitor.solver_timeout_ms,
        "scope": "bounded one-step monitor abstraction and finite state-machine rules",
        "properties": properties,
        "examples": {
            "safe_action_no_counterexample": safe_example.to_dict(),
            "unsafe_action_counterexample": unsafe_example.to_dict(),
        },
        "property_coverage": {
            "total_obligations": len(properties),
            "verified_within_bounds": sum(item["verification_status"] == "verified_within_bounds" for item in properties),
            "failed": sum(item["verification_status"] == "failed" for item in properties),
            "unknown": sum(item["verification_status"] == "unknown" for item in properties),
        },
        "assumptions": config.assumptions,
        "limitations": config.limitations,
        "claim_boundary": (
            "These results do not verify the neural network, full simulator, real hardware, "
            "or safety outside the configured variables, bounds, abstraction, and horizon."
        ),
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
