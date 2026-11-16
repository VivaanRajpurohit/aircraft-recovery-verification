from __future__ import annotations

import pytest
import torch

from aircraft_recovery.data_quality.final_approved import assert_role_split
from aircraft_recovery.training.models import ImitationPolicyNetwork
from aircraft_recovery.training.v2_1_architecture_screen import (
    ARCHITECTURE_CANDIDATES,
    ARCHITECTURE_SCREEN_FORBIDDEN_SPLITS,
    ArchitectureScreenDataSource,
    FINAL_TRAINING_SEEDS,
    OPTIMIZER_GRID,
    SCREEN_TRAINING_CONTROLS,
    assert_architecture_screen_access_policy,
    multi_head_imitation_loss,
    validate_multi_head_label_contract,
)


def _valid_record(split: str) -> dict:
    return {
        "split": split,
        "expert_proposed_controls": [0.0, 0.0, 0.0, 0.5, 0.5],
        "expert_emergency_mode": "normal",
        "expert_stabilization_targets": [0.0, 0.0, 120.0, 5000.0],
    }


def test_frozen_architecture_candidates_and_parameter_counts() -> None:
    observed = [
        (item.hidden_dimensions, item.activation, item.parameter_count)
        for item in ARCHITECTURE_CANDIDATES
    ]
    assert observed == [
        ((128, 64), "relu", 14670),
        ((256, 128), "relu", 45710),
        ((256, 256, 128), "silu", 111502),
    ]


def test_frozen_search_and_training_controls() -> None:
    assert OPTIMIZER_GRID == {
        "learning_rates": (0.0003, 0.001),
        "batch_sizes": (256, 512),
    }
    assert SCREEN_TRAINING_CONTROLS == {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "batch_size": 256,
        "weight_decay": 0.0001,
        "maximum_epochs": 50,
        "patience": 8,
        "improvement_tolerance": 1e-10,
        "gradient_norm_clip": 1.0,
    }
    assert FINAL_TRAINING_SEEDS == (17001, 17002, 17003, 17004, 17005)


def test_data_source_requests_only_train_and_development() -> None:
    calls: list[tuple[str, str]] = []

    def loader(_dataset, *, split: str, role: str):
        calls.append((split, role))
        return iter([_valid_record(split)])

    source = ArchitectureScreenDataSource("approved", loader=loader)
    assert list(source.training_records())[0]["split"] == "train"
    assert list(source.development_records())[0]["split"] == "development"
    assert calls == [("train", "training"), ("development", "model_selection")]
    assert not hasattr(source, "records_for_split")


@pytest.mark.parametrize("role", ["training", "model_selection"])
@pytest.mark.parametrize("split", sorted(ARCHITECTURE_SCREEN_FORBIDDEN_SPLITS))
def test_architecture_roles_reject_held_out_splits(role: str, split: str) -> None:
    with pytest.raises(PermissionError):
        assert_role_split(role, split)


def test_sealed_guard_rejects_final_sealed_test() -> None:
    with pytest.raises(PermissionError, match="final_sealed_test"):
        assert_role_split("training", "final_sealed_test")
    with pytest.raises(PermissionError, match="final_sealed_test"):
        assert_role_split("model_selection", "final_sealed_test")


def test_access_policy_self_check() -> None:
    assert_architecture_screen_access_policy()


def test_approved_v2_1_shape_fails_closed_without_auxiliary_labels() -> None:
    approved_shape = {
        "ordered_observation_42": [0.0] * 42,
        "expert_proposed_controls": [0.0] * 5,
    }
    with pytest.raises(ValueError, match="expert_emergency_mode"):
        validate_multi_head_label_contract(approved_shape)


def test_frozen_loss_uses_mse_cross_entropy_and_normalized_target_mse() -> None:
    model = ImitationPolicyNetwork(ARCHITECTURE_CANDIDATES[0].model_config())
    predictions = model(torch.zeros((2, 42)))
    controls = torch.zeros((2, 5))
    modes = torch.zeros(2, dtype=torch.long)
    targets = torch.zeros((2, 4))
    total, terms = multi_head_imitation_loss(predictions, controls, modes, targets)
    assert total.ndim == 0
    assert set(terms) == {
        "controls_mse",
        "mode_cross_entropy",
        "targets_normalized_mse",
    }

