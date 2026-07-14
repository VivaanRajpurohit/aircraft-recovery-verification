"""Deterministic scenario execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from aircraft_recovery.controllers.base import Controller
from aircraft_recovery.experiments.logger import ExperimentLogger
from aircraft_recovery.metrics import Phase2Metrics
from aircraft_recovery.simulator.base import AircraftSimulator


@dataclass(frozen=True)
class RunResult:
    """Summary and artifact location for one scenario run."""

    run_directory: Path
    steps: int
    termination_reason: str
    elapsed_wall_seconds: float
    minimum_airspeed_margin_kts: float
    maximum_absolute_bank_deg: float


class ScenarioRunner:
    """Run a simulator/controller pair and persist reproducible artifacts."""

    def __init__(self, simulator: AircraftSimulator, controller: Controller) -> None:
        self.simulator = simulator
        self.controller = controller

    def run(self, seed: int, run_directory: str | Path, scenario_id: str) -> RunResult:
        logger = ExperimentLogger(run_directory)
        simulator_config = getattr(self.simulator, "scenario", None)
        logger.write_metadata({
            "scenario_id": scenario_id,
            "random_seed": seed,
            "controller_version": self.controller.name,
            "model_checkpoint_identifier": getattr(self.controller, "model_checkpoint_identifier", None),
            "safety_configuration": None,
            "phase": 3 if self.controller.name == "phase3_neural_baseline" else 2 if self.controller.name == "phase2_deterministic_fallback" else 1,
            "checkpoint_metadata": getattr(self.controller, "checkpoint_metadata", None),
            "scenario_configuration": simulator_config.model_dump(mode="json") if simulator_config else None,
        })
        reset_controller = getattr(self.controller, "reset", None)
        if callable(reset_controller):
            reset_controller()
        observation = self.simulator.reset(seed)
        steps = 0
        min_margin = observation.flight_envelope.stall_margin_kts
        max_bank = abs(observation.aircraft_state.roll_deg)
        dt = float(getattr(getattr(self.simulator, "aircraft", None), "time_step_seconds", 0.1))
        phase2_metrics = Phase2Metrics(dt)
        started = perf_counter()
        while not self.simulator.is_terminal():
            ground_truth_before = self._audit_snapshot()
            controller_started = perf_counter()
            action = self.controller.act(observation)
            controller_latency_ms = (perf_counter() - controller_started) * 1000.0
            next_observation = self.simulator.step(action)
            ground_truth_after = self._audit_snapshot()
            # Phase 1 has no safety filter, so proposed and final actions match.
            proposed_action = getattr(self.controller, "last_proposed_action", None) or action
            monitor_decision = getattr(self.controller, "last_monitor_decision", None)
            logger.append_step(
                observation, proposed_action, action, next_observation,
                ground_truth_before, ground_truth_after, controller_latency_ms,
                monitor_decision.to_dict() if monitor_decision else None,
            )
            phase2_metrics.update(observation, action, ground_truth_before, controller_latency_ms)
            observation = next_observation
            steps += 1
            min_margin = min(min_margin, observation.flight_envelope.stall_margin_kts)
            max_bank = max(max_bank, abs(observation.aircraft_state.roll_deg))
        elapsed = perf_counter() - started
        reason = self.simulator.termination_reason() or "unknown"
        summary = {
            "steps": steps,
            "termination_reason": reason,
            "simulated_duration_seconds": self.simulator.time_seconds,
            "elapsed_wall_seconds": elapsed,
            "minimum_airspeed_margin_kts": min_margin,
            "maximum_absolute_bank_deg": max_bank,
            "final_altitude_ft": observation.aircraft_state.altitude_ft,
            "final_airspeed_kts": observation.aircraft_state.airspeed_kts,
        }
        state_machine = getattr(self.controller, "state_machine", None)
        transitions = list(getattr(state_machine, "transitions", []))
        plan = getattr(self.controller, "last_plan", None)
        selected = getattr(self.controller, "selected_airport", None)
        audit = self._audit_snapshot() or {}
        final_distances = audit.get("airport_distances_nm", {})
        summary.update(phase2_metrics.summary(
            reason,
            len(transitions),
            getattr(plan, "feasibility", None),
            final_distances.get(selected) if selected else None,
        ))
        summary["mode_transitions"] = [transition.to_dict() for transition in transitions]
        summary["diversion_plan"] = plan.to_dict() if plan and selected else None
        monitor_statistics = getattr(self.controller, "monitor_statistics", None)
        summary["monitor_statistics"] = monitor_statistics() if callable(monitor_statistics) else None
        logger.write_summary(summary)
        return RunResult(Path(run_directory), steps, reason, elapsed, min_margin, max_bank)

    def _audit_snapshot(self) -> dict[str, Any] | None:
        snapshot = getattr(self.simulator, "audit_snapshot", None)
        return snapshot() if callable(snapshot) else None
