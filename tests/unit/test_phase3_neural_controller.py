import numpy as np
import pytest
import torch

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import NeuralBaselineController
from aircraft_recovery.data.preprocessing import ObservationPreprocessor
from aircraft_recovery.models import ControllerOutput
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.training.checkpointing import build_checkpoint, save_checkpoint
from aircraft_recovery.training.models import ImitationPolicyNetwork, PolicyArchitectureConfig


def make_checkpoint(path) -> None:
    torch.manual_seed(1)
    model = ImitationPolicyNetwork(PolicyArchitectureConfig(hidden_dimensions=[16]))
    preprocessor = ObservationPreprocessor()
    preprocessor.fit(np.zeros((3, 42), dtype=np.float32))
    checkpoint = build_checkpoint(
        model, preprocessor.statistics, {}, "dataset", "split", {"torch": torch.__version__},
        {"total": 1.0}, 1,
    )
    save_checkpoint(path, checkpoint)


def nominal_observation():
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase1_nominal.yaml", ScenarioConfig)
    return SimpleAircraftSimulator(aircraft, scenario).reset(scenario.seed)


def test_neural_controller_cpu_output_schema_and_bounds(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    make_checkpoint(checkpoint)
    controller = NeuralBaselineController(checkpoint, "cpu")
    output = controller.act(nominal_observation())
    assert isinstance(output, ControllerOutput)
    assert output.safety_decision.status == "not_checked"
    assert not output.safety_decision.fallback_activated
    assert -1.0 <= output.control_commands.elevator <= 1.0
    assert 0.0 <= output.control_commands.throttle_left <= 1.0


def test_neural_runtime_failure_is_distinct_execution_fallback(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    make_checkpoint(checkpoint)
    controller = NeuralBaselineController(checkpoint, "cpu")
    monkeypatch.setattr(controller.model, "forward", lambda _: (_ for _ in ()).throw(RuntimeError("test")))
    output = controller.act(nominal_observation())
    assert output.safety_decision.status == "fallback"
    assert "not a Phase 4 safety intervention" in output.safety_decision.explanation
    assert controller.runtime_execution_fallback_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_neural_controller_cuda_inference(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    make_checkpoint(checkpoint)
    output = NeuralBaselineController(checkpoint, "cuda").act(nominal_observation())
    assert isinstance(output, ControllerOutput)

