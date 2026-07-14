import json

import numpy as np

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import FallbackConfig, MonitoredNeuralController
from aircraft_recovery.data.preprocessing import ObservationPreprocessor
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.safety import SafetyConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.training.checkpointing import build_checkpoint, save_checkpoint
from aircraft_recovery.training.models import ImitationPolicyNetwork, PolicyArchitectureConfig


def make_checkpoint(path) -> None:
    model = ImitationPolicyNetwork(PolicyArchitectureConfig(hidden_dimensions=[16, 8]))
    preprocessor = ObservationPreprocessor()
    preprocessor.fit(np.zeros((3, 42), dtype=np.float32))
    save_checkpoint(path, build_checkpoint(
        model, preprocessor.statistics, {}, "dataset", "split", {}, {"total": 1.0}, 1
    ))


def test_monitored_controller_runs_closed_loop_and_preserves_proposal(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    make_checkpoint(checkpoint)
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase2_engine_aileron.yaml", ScenarioConfig)
    scenario.duration_seconds = 2.0
    controller = MonitoredNeuralController(
        checkpoint, "cpu",
        load_config("configs/controllers/fallback.yaml", FallbackConfig),
        scenario.airports,
        load_config("configs/safety/default.yaml", SafetyConfig),
    )
    result = ScenarioRunner(SimpleAircraftSimulator(aircraft, scenario), controller).run(
        scenario.seed, tmp_path / "run", scenario.scenario_id
    )
    rows = [json.loads(line) for line in (result.run_directory / "history.jsonl").read_text().splitlines()]
    assert rows
    assert all("monitor_decision" in row for row in rows)
    assert all("proposed_action" in row and "final_action" in row for row in rows)
    summary = json.loads((result.run_directory / "summary.json").read_text())
    assert sum(summary["monitor_statistics"].values()) == result.steps
