"""Run one deterministic Phase 1 simulation scenario."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import DeterministicFallbackController, FallbackConfig, StabilizationController
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.simulator import SimpleAircraftSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aircraft", default="configs/aircraft/simple_twin.yaml")
    parser.add_argument(
        "--scenario", "--config", dest="scenario",
        default="configs/scenarios/phase1_nominal.yaml",
        help="Scenario YAML path (--config is a backward-compatible alias).",
    )
    parser.add_argument("--fallback-config", default="configs/controllers/fallback.yaml")
    parser.add_argument("--controller", choices=("auto", "stabilization", "fallback"), default="auto")
    parser.add_argument("--output", default="results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aircraft = load_config(args.aircraft, AircraftConfig)
    scenario = load_config(args.scenario, ScenarioConfig)
    simulator = SimpleAircraftSimulator(aircraft, scenario)
    use_fallback = args.controller == "fallback" or (
        args.controller == "auto" and (scenario.failures or "phase2" in scenario.scenario_id)
    )
    if use_fallback:
        fallback_config = load_config(args.fallback_config, FallbackConfig)
        controller = DeterministicFallbackController(fallback_config, scenario.airports)
    else:
        controller = StabilizationController(target_airspeed_kts=aircraft.cruise_speed_kts)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output) / f"{scenario.scenario_id}_{scenario.seed}_{timestamp}"
    result = ScenarioRunner(simulator, controller).run(scenario.seed, run_dir, scenario.scenario_id)
    print(f"Run complete: {result.run_directory}")
    print(f"Steps: {result.steps}; termination: {result.termination_reason}")
    print(f"Minimum stall margin: {result.minimum_airspeed_margin_kts:.2f} kt")
    summary = json.loads((result.run_directory / "summary.json").read_text(encoding="utf-8"))
    print(f"Recovery result: {summary['recovery_result']}; failures: {summary['active_failure_count']}")
    if summary.get("diversion_plan"):
        print(f"Diversion: {summary['diversion_plan']['selected_airport']} ({summary['diversion_plan']['feasibility']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
