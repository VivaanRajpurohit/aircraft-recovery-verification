from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from aircraft_recovery.data_quality.common import object_hash
from aircraft_recovery.data_quality.final_approved import assert_role_split
from aircraft_recovery.training.v2_1a_optimizer_selection import (
    OPTIMIZER_GRID_RUNS,
    OPTIMIZER_SELECTION_SEED,
    SELECTION_CONTROLS,
    WINNING_ARCHITECTURE,
    V21AControlPolicyNetwork,
    optimizer_selection_preflight,
)


def test_frozen_optimizer_grid_has_exactly_four_combinations() -> None:
    assert len(OPTIMIZER_GRID_RUNS) == 4
    assert {(run.learning_rate, run.batch_size) for run in OPTIMIZER_GRID_RUNS} == {
        (0.0003, 256), (0.0003, 512), (0.001, 256), (0.001, 512)
    }
    assert OPTIMIZER_SELECTION_SEED == 17001
    assert SELECTION_CONTROLS == {
        "optimizer": "AdamW",
        "objective": "mean_squared_error",
        "checkpoint_selection": "minimum_development_mse",
        "weight_decay": 0.0001,
        "maximum_epochs": 50,
        "patience": 8,
        "improvement_tolerance": 1e-10,
        "gradient_norm_clip": 1.0,
    }


def test_winning_architecture_is_fixed_and_has_five_outputs() -> None:
    assert WINNING_ARCHITECTURE.architecture_id == "moderate"
    assert WINNING_ARCHITECTURE.hidden_dimensions == (256, 256, 128)
    assert WINNING_ARCHITECTURE.activation == "silu"
    model = V21AControlPolicyNetwork(WINNING_ARCHITECTURE)
    assert model.parameter_count == 110341
    assert model(torch.zeros((2, 42))).shape == (2, 5)


@pytest.mark.parametrize("role", ["training", "model_selection"])
@pytest.mark.parametrize("split", ["challenge", "ood_stress", "final_sealed_test"])
def test_optimizer_selection_rejects_all_post_selection_splits(role: str, split: str) -> None:
    with pytest.raises(PermissionError):
        assert_role_split(role, split)


def test_optimizer_selection_preflight_is_bound_to_completed_screen() -> None:
    root = Path(__file__).resolve().parents[2]
    result = optimizer_selection_preflight(root, verify_large_file=False)
    assert result["status"] == "PASS"
    assert result["grid_run_count"] == 4
    assert result["architecture"]["architecture_id"] == "moderate"
    assert result["development_checkpoint_selection_metric"] == "mean_squared_error"
    assert result["challenge_opened"] is False
    assert result["ood_opened"] is False
    assert result["sealed_opened"] is False


@pytest.mark.parametrize(
    ("filename", "hash_field"),
    [
        ("final_optimizer_selection_v2_1a.json", "result_logical_sha256"),
        ("final_training_configuration_v2_1a.json", "freeze_logical_sha256"),
    ],
)
def test_selection_and_final_configuration_freezes_have_canonical_hashes(
    filename: str, hash_field: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    artifact = json.loads((root / "artifacts/research_stage" / filename).read_text(encoding="utf-8"))
    declared = artifact.pop(hash_field)
    assert object_hash(artifact) == declared
    assert artifact["challenge_opened"] is False
    assert artifact["ood_opened"] is False
    assert artifact["sealed_opened"] is False
