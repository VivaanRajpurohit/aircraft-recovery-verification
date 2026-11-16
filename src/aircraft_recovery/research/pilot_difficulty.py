"""Development-only controller baselines for candidate scenario difficulty.

This module consumes scenario configuration, never candidate transition labels or
the sealed held-out test partition.  A flight is a recovery only after the
aircraft remains inside the preregistered strict envelope for five seconds.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from aircraft_recovery.config import AircraftConfig, load_config
from aircraft_recovery.controllers import DeterministicFallbackController, FallbackConfig
from aircraft_recovery.controllers.neural import NeuralBaselineController
from aircraft_recovery.data_quality.candidate_generation import (
    CandidateFlightConfig,
    CandidateGenerationConfig,
    _effective_scenario,
)
from aircraft_recovery.models import (
    ControlCommands,
    ControllerInput,
    ControllerOutput,
    NavigationCommands,
    SafetyDecision,
    StabilizationTargets,
)
from aircraft_recovery.simulator import SimpleAircraftSimulator


CONTROL_NAMES = ("elevator", "aileron", "rudder", "throttle_left", "throttle_right")
STRICT_HOLD_SECONDS = 5.0


@dataclass
class FixedActionController:
    """A deliberately non-reactive baseline with a constant command vector."""

    name: str
    controls: tuple[float, float, float, float, float]

    def reset(self) -> None:
        return None

    def act(self, observation: ControllerInput) -> ControllerOutput:
        state = observation.aircraft_state
        return ControllerOutput(
            timestamp_seconds=observation.timestamp_seconds,
            control_commands=ControlCommands(
                elevator=self.controls[0],
                aileron=self.controls[1],
                rudder=self.controls[2],
                throttle_left=self.controls[3],
                throttle_right=self.controls[4],
            ),
            stabilization_targets=StabilizationTargets(
                target_pitch_deg=0.0,
                target_roll_deg=0.0,
                target_airspeed_kts=140.0,
                target_altitude_ft=state.altitude_ft,
            ),
            navigation_commands=NavigationCommands(target_heading_deg=state.heading_deg),
            emergency_mode="normal",
            safety_decision=SafetyDecision(
                status="not_checked",
                ai_action_accepted=True,
                fallback_activated=False,
                explanation="Fixed development-only baseline; no runtime monitor.",
            ),
            confidence=1.0,
        )


def _clearance(observation: ControllerInput) -> float:
    clearances = [item.terrain_clearance_ft for item in observation.navigation.nearest_airports]
    return min(clearances) if clearances else observation.aircraft_state.altitude_ft


def _strict_recovery(observation: ControllerInput) -> bool:
    state = observation.aircraft_state
    return (
        abs(state.roll_deg) <= 10.0
        and abs(state.pitch_deg) <= 7.5
        and abs(state.vertical_speed_fpm) <= 750.0
        and observation.flight_envelope.stall_margin_kts >= 10.0
        and _clearance(observation) >= 0.0
    )


def _safety_violation(observation: ControllerInput) -> bool:
    return (
        observation.flight_envelope.stall_margin_kts < 0.0
        or abs(observation.aircraft_state.roll_deg) > 60.0
        or observation.flight_envelope.angle_of_attack_deg > 14.0
        or _clearance(observation) < 0.0
    )


def _recovery_time(samples: list[tuple[float, bool]], reference: float, dt: float) -> float | None:
    required = max(1, int(round(STRICT_HOLD_SECONDS / dt)))
    eligible = [(time_value, safe) for time_value, safe in samples if time_value >= reference]
    run = 0
    for time_value, safe in eligible:
        run = run + 1 if safe else 0
        if run >= required:
            start = time_value - (required - 1) * dt
            return max(0.0, start - reference)
    return None


def run_flight(
    flight: CandidateFlightConfig,
    aircraft: AircraftConfig,
    controller: Any,
) -> dict[str, Any]:
    scenario = _effective_scenario(flight)
    simulator = SimpleAircraftSimulator(aircraft, scenario)
    reset = getattr(controller, "reset", None)
    if callable(reset):
        reset()
    observation = simulator.reset(flight.seed)
    dt = aircraft.time_step_seconds
    samples: list[tuple[float, bool]] = []
    commands: list[list[float]] = []
    latencies: list[float] = []
    known_failure_ids: set[str] = set()
    latest_activation = 0.0
    violation_steps = 0
    min_airspeed = observation.aircraft_state.airspeed_kts
    max_bank = abs(observation.aircraft_state.roll_deg)
    max_pitch = abs(observation.aircraft_state.pitch_deg)
    min_clearance = _clearance(observation)
    while not simulator.is_terminal():
        started = perf_counter()
        action = controller.act(observation)
        latencies.append((perf_counter() - started) * 1000.0)
        command = [float(getattr(action.control_commands, name)) for name in CONTROL_NAMES]
        commands.append(command)
        observation = simulator.step(action)
        audit = simulator.audit_snapshot()
        active = {str(item["failure_id"]): float(item["activation_time_seconds"])
                  for item in audit["ground_truth_failures"]}
        for failure_id, activation_time in active.items():
            if failure_id not in known_failure_ids:
                latest_activation = max(latest_activation, activation_time)
                known_failure_ids.add(failure_id)
        samples.append((observation.timestamp_seconds, _strict_recovery(observation)))
        violation_steps += int(_safety_violation(observation))
        state = observation.aircraft_state
        min_airspeed = min(min_airspeed, state.airspeed_kts)
        max_bank = max(max_bank, abs(state.roll_deg))
        max_pitch = max(max_pitch, abs(state.pitch_deg))
        min_clearance = min(min_clearance, _clearance(observation))
    command_array = np.asarray(commands, dtype=np.float64)
    saturation = np.any(np.abs(command_array[:, :3]) >= 0.99, axis=1) | np.any(
        (command_array[:, 3:] <= 0.01) | (command_array[:, 3:] >= 0.99), axis=1
    )
    recovery_time = _recovery_time(samples, latest_activation, dt)
    return {
        "controller": controller.name,
        "scenario_id": scenario.scenario_id,
        "scenario_instance_id": flight.scenario_instance_id or f"{scenario.scenario_id}:{flight.seed}",
        "split": flight.split,
        "difficulty_tier": flight.difficulty_tier,
        "scenario_family": flight.scenario_family,
        "seed": flight.seed,
        "steps": len(commands),
        "termination_reason": simulator.termination_reason(),
        "failure_reference_time_seconds": latest_activation,
        "strict_recovery_success": recovery_time is not None,
        "strict_recovery_time_seconds": recovery_time,
        "minimum_airspeed_kts": min_airspeed,
        "maximum_absolute_bank_deg": max_bank,
        "maximum_absolute_pitch_deg": max_pitch,
        "minimum_terrain_clearance_ft": min_clearance,
        "safety_violation_count": violation_steps,
        "safety_violation_duration_seconds": violation_steps * dt,
        "command_saturation_frequency": float(np.mean(saturation)),
        "control_oscillation_mean_absolute_delta": (
            float(np.abs(np.diff(command_array, axis=0)).mean()) if len(command_array) > 1 else 0.0
        ),
        "controller_latency_ms_mean": float(np.mean(latencies)),
        "controller_latency_ms_p99": float(np.percentile(latencies, 99)),
        "runtime_fallback_count": int(getattr(controller, "runtime_execution_fallback_count", 0)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["controller"]].append(row)
    output: dict[str, Any] = {}
    for controller, controller_rows in grouped.items():
        tiers: dict[str, Any] = {}
        for tier in sorted({int(row["difficulty_tier"]) for row in controller_rows}):
            subset = [row for row in controller_rows if int(row["difficulty_tier"]) == tier]
            successes = sum(bool(row["strict_recovery_success"]) for row in subset)
            tiers[str(tier)] = {"flights": len(subset), "recoveries": successes,
                                "recovery_rate": successes / len(subset)}
        recoveries = [row["strict_recovery_time_seconds"] for row in controller_rows
                      if row["strict_recovery_time_seconds"] is not None]
        success_count = sum(bool(row["strict_recovery_success"]) for row in controller_rows)
        output[controller] = {
            "flights": len(controller_rows),
            "recoveries": success_count,
            "recovery_rate": success_count / len(controller_rows),
            "recovery_time_seconds_mean_successes_only": float(np.mean(recoveries)) if recoveries else None,
            "flights_with_any_safety_violation": sum(row["safety_violation_count"] > 0 for row in controller_rows),
            "safety_violation_duration_seconds_total": sum(row["safety_violation_duration_seconds"] for row in controller_rows),
            "command_saturation_frequency_mean": float(np.mean([row["command_saturation_frequency"] for row in controller_rows])),
            "control_oscillation_mean_absolute_delta": float(np.mean([row["control_oscillation_mean_absolute_delta"] for row in controller_rows])),
            "runtime_fallback_count": sum(row["runtime_fallback_count"] for row in controller_rows),
            "by_tier": tiers,
        }
    return output


def controller_factories(
    fallback_config_path: str,
    phase3_checkpoint: str,
    smoke_checkpoint: str,
) -> dict[str, Callable[[Any], Any]]:
    fallback_config = load_config(fallback_config_path, FallbackConfig)

    def named_neural(checkpoint: str, name: str) -> NeuralBaselineController:
        controller = NeuralBaselineController(checkpoint, device="cpu")
        # The implementation class intentionally has one historical runtime name;
        # the instance label keeps distinct checkpoint identities in this audit.
        controller.name = name
        return controller

    return {
        "constant_idle": lambda scenario: FixedActionController(
            "constant_idle", (0.0, 0.0, 0.0, 0.0, 0.0)
        ),
        "neutral_cruise": lambda scenario: FixedActionController(
            "neutral_cruise", (0.0, 0.0, 0.0, 0.5, 0.5)
        ),
        "expert": lambda scenario: DeterministicFallbackController(fallback_config, scenario.airports),
        "historical_phase3": lambda scenario: named_neural(phase3_checkpoint, "historical_phase3"),
        "smoke_checkpoint": lambda scenario: named_neural(smoke_checkpoint, "smoke_checkpoint"),
    }


def run_pilot_difficulty(
    config: CandidateGenerationConfig,
    phase3_checkpoint: str,
    smoke_checkpoint: str,
) -> dict[str, Any]:
    aircraft = load_config(config.aircraft_config, AircraftConfig)
    factories = controller_factories(config.fallback_config, phase3_checkpoint, smoke_checkpoint)
    rows: list[dict[str, Any]] = []
    for flight in config.flights:
        scenario = _effective_scenario(flight)
        for factory in factories.values():
            rows.append(run_flight(flight, aircraft, factory(scenario)))
    return {
        "analysis_scope": "development-visible candidate scenarios only; no sealed-test data accessed",
        "recovery_definition": {
            "roll_absolute_max_deg": 10.0,
            "pitch_absolute_max_deg": 7.5,
            "vertical_speed_absolute_max_fpm": 750.0,
            "stall_margin_min_kts": 10.0,
            "terrain_clearance_min_ft": 0.0,
            "continuous_hold_seconds": STRICT_HOLD_SECONDS,
            "reference": "latest observed failure activation, or simulation start for nominal flights",
        },
        "scenario_count": len(config.flights),
        "rows": rows,
        "aggregate": aggregate(rows),
    }
