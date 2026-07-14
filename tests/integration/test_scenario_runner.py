import json

import pytest

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import StabilizationController
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.simulator import SimpleAircraftSimulator


def test_scenario_is_deterministic_and_logs_json(tmp_path) -> None:
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase1_nominal.yaml", ScenarioConfig)
    simulator = SimpleAircraftSimulator(aircraft, scenario)
    run_dir = tmp_path / "run"
    result = ScenarioRunner(simulator, StabilizationController()).run(
        scenario.seed, run_dir, scenario.scenario_id
    )
    assert result.termination_reason == "duration_complete"
    assert result.steps == 200
    assert (run_dir / "metadata.json").is_file()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] == 200
    lines = (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200
    assert json.loads(lines[0])["observation"]["timestamp_seconds"] == 0.0
    final_record = json.loads(lines[-1])
    assert final_record["next_observation"]["timestamp_seconds"] == pytest.approx(20.0)
    assert final_record["proposed_action"] == final_record["final_action"]
