import json

import pytest

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import DeterministicFallbackController, FallbackConfig
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.simulator import SimpleAircraftSimulator


SCENARIOS = [
    ("phase2_nominal.yaml", "duration_complete"),
    ("phase2_single_engine.yaml", "duration_complete"),
    ("phase2_engine_aileron.yaml", "duration_complete"),
    ("phase2_sensor_actuator.yaml", "duration_complete"),
    ("phase2_rudder_crosswind.yaml", "duration_complete"),
    ("phase2_unrecoverable.yaml", "unrecoverable_condition"),
]


def build(path: str):
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config(f"configs/scenarios/{path}", ScenarioConfig)
    fallback = load_config("configs/controllers/fallback.yaml", FallbackConfig)
    simulator = SimpleAircraftSimulator(aircraft, scenario)
    return scenario, simulator, DeterministicFallbackController(fallback, scenario.airports)


@pytest.mark.parametrize(("filename", "expected_reason"), SCENARIOS)
def test_required_demonstration_scenarios_run(
    tmp_path, filename: str, expected_reason: str
) -> None:
    scenario, simulator, controller = build(filename)
    result = ScenarioRunner(simulator, controller).run(
        scenario.seed, tmp_path / scenario.scenario_id, scenario.scenario_id
    )
    assert result.termination_reason == expected_reason
    summary = json.loads((result.run_directory / "summary.json").read_text(encoding="utf-8"))
    assert "controller_latency_ms_mean" in summary
    assert "mode_transitions" in summary
    assert summary["termination_reason"] == expected_reason


def test_identical_seeds_produce_identical_transitions(tmp_path) -> None:
    records = []
    for index in range(2):
        scenario, simulator, controller = build("phase2_actuator_turbulence_thrust.yaml")
        result = ScenarioRunner(simulator, controller).run(
            scenario.seed, tmp_path / f"run{index}", scenario.scenario_id
        )
        rows = [json.loads(line) for line in (result.run_directory / "history.jsonl").read_text().splitlines()]
        records.append([
            (row["next_observation"], row["final_action"], row["ground_truth_after"])
            for row in rows
        ])
    assert records[0] == records[1]


def test_phase1_nominal_remains_backward_compatible(tmp_path) -> None:
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase1_nominal.yaml", ScenarioConfig)
    from aircraft_recovery.controllers import StabilizationController

    result = ScenarioRunner(
        SimpleAircraftSimulator(aircraft, scenario), StabilizationController()
    ).run(scenario.seed, tmp_path / "phase1", scenario.scenario_id)
    assert result.steps == 200
    assert result.termination_reason == "duration_complete"
