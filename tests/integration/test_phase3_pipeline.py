import json
from pathlib import Path

import numpy as np
import pytest

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import NeuralBaselineController
from aircraft_recovery.data.demonstrations import (
    DemonstrationConfig,
    ScenarioCategoryConfig,
    generate_demonstrations,
)
from aircraft_recovery.data.splitting import create_split_manifest, validate_no_episode_overlap
from aircraft_recovery.data.validation import validate_dataset
from aircraft_recovery.evaluation import EvaluationConfig, evaluate_policy
from aircraft_recovery.evaluation.policy_evaluator import EvaluationScenario
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.training import PolicyArchitectureConfig, TrainingConfig, train_policy
from aircraft_recovery.training.trainer import LossConfiguration


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3_pipeline")
    dataset_directory = root / "dataset"
    demonstration_config = DemonstrationConfig(
        dataset_name="test_dataset",
        output_directory=str(dataset_directory),
        seed=8000,
        episode_count=6,
        aircraft_config="configs/aircraft/simple_twin.yaml",
        fallback_config="configs/controllers/fallback.yaml",
        train_fraction=0.7,
        validation_fraction=0.15,
        test_fraction=0.15,
        ood_categories=["unrecoverable"],
        scenarios=[
            ScenarioCategoryConfig(category="nominal", scenario="configs/scenarios/phase2_nominal.yaml", weight=1.0),
            ScenarioCategoryConfig(category="engine", scenario="configs/scenarios/phase2_single_engine.yaml", weight=1.0),
            ScenarioCategoryConfig(category="unrecoverable", scenario="configs/scenarios/phase2_unrecoverable.yaml", weight=1.0),
        ],
    )
    metadata = generate_demonstrations(demonstration_config)
    manifest = create_split_manifest(dataset_directory, 8000, 0.7, 0.15, 0.15, {"unrecoverable"})
    report = validate_dataset(dataset_directory)
    checkpoint_directory = root / "checkpoints"
    training_config = TrainingConfig(
        experiment_name="test_training",
        dataset_directory=str(dataset_directory),
        split_manifest=str(dataset_directory / "split_manifest.json"),
        output_directory=str(checkpoint_directory),
        seed=8100,
        device="cpu",
        architecture=PolicyArchitectureConfig(hidden_dimensions=[32, 16]),
        batch_size=128,
        learning_rate=0.001,
        epochs=2,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        early_stopping_patience=2,
        checkpoint_every_epochs=1,
        clip_standard_deviations=5.0,
        balance_emergency_modes=True,
        automatic_mixed_precision=False,
        num_workers=0,
        losses=LossConfiguration(controls=1.0, emergency_mode=0.25, stabilization_targets=0.1),
    )
    training = train_policy(training_config)
    return {
        "root": root,
        "dataset": dataset_directory,
        "metadata": metadata,
        "manifest": manifest,
        "report": report,
        "training_config": training_config,
        "demonstration_config": demonstration_config,
        "training": training,
        "checkpoint": checkpoint_directory / "best.pt",
    }


def test_demonstration_generation_and_no_leakage(trained_artifacts) -> None:
    archive = np.load(trained_artifacts["dataset"] / "transitions.npz", allow_pickle=False)
    assert archive["observations"].shape[1] == 42
    assert trained_artifacts["report"]["valid"]
    assert trained_artifacts["metadata"]["analysis_only_fields"] == ["ground_truth_before", "ground_truth_after"]
    assert "true_state" not in archive.files
    assert (trained_artifacts["dataset"] / "analysis_ground_truth.jsonl").is_file()


def test_demonstration_generation_is_deterministic(trained_artifacts) -> None:
    config = trained_artifacts["demonstration_config"].model_copy(deep=True)
    config.output_directory = str(trained_artifacts["root"] / "dataset_repeat")
    repeated = generate_demonstrations(config)
    assert repeated["dataset_identifier"] == trained_artifacts["metadata"]["dataset_identifier"]


def test_training_smoke_and_checkpoint_metadata(trained_artifacts) -> None:
    assert trained_artifacts["checkpoint"].is_file()
    assert trained_artifacts["training"]["parameter_count"] < 1_000_000
    assert trained_artifacts["training"]["best_epoch"] in {1, 2}
    validate_no_episode_overlap(trained_artifacts["manifest"])


def test_training_is_reproducible_with_identical_seed(trained_artifacts) -> None:
    config = trained_artifacts["training_config"].model_copy(deep=True)
    config.output_directory = str(trained_artifacts["root"] / "checkpoints_repeat")
    repeated = train_policy(config)
    assert repeated["best_validation_loss"] == pytest.approx(
        trained_artifacts["training"]["best_validation_loss"], abs=1e-10
    )
    assert repeated["final_train_losses"] == pytest.approx(
        trained_artifacts["training"]["final_train_losses"], abs=1e-10
    )


def test_closed_loop_neural_policy_receives_own_next_state(trained_artifacts, tmp_path) -> None:
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase1_nominal.yaml", ScenarioConfig)
    controller = NeuralBaselineController(trained_artifacts["checkpoint"], "cpu")
    result = ScenarioRunner(SimpleAircraftSimulator(aircraft, scenario), controller).run(
        scenario.seed, tmp_path / "neural_closed_loop", scenario.scenario_id
    )
    assert result.steps > 0
    assert result.termination_reason in {"duration_complete", "loss_of_control", "ground_impact"}
    rows = (result.run_directory / "history.jsonl").read_text().splitlines()
    assert json.loads(rows[1])["observation"] == json.loads(rows[0])["next_observation"]


def test_paired_evaluation_uses_matching_seeds(trained_artifacts) -> None:
    config = EvaluationConfig(
        evaluation_name="test_evaluation",
        dataset_directory=str(trained_artifacts["dataset"]),
        split_manifest=str(trained_artifacts["dataset"] / "split_manifest.json"),
        checkpoint=str(trained_artifacts["checkpoint"]),
        output_directory=str(trained_artifacts["root"] / "evaluation"),
        aircraft_config="configs/aircraft/simple_twin.yaml",
        fallback_config="configs/controllers/fallback.yaml",
        device="cpu",
        offline_splits=["test", "ood"],
        scenarios=[EvaluationScenario(
            scenario="configs/scenarios/phase2_engine_aileron.yaml",
            category="two_failure",
            seed_offsets=[500],
        )],
    )
    result = evaluate_policy(config)
    assert result["paired_seed_match"]
    pair = result["paired_closed_loop"][0]
    assert pair["seed"] == 2603
    assert "control_mae" in result["offline"]["test"]
