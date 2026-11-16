from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from aircraft_recovery.data_quality.common import object_hash
from aircraft_recovery.data_quality.final_approved import assert_role_split
from aircraft_recovery.training.v2_1a_final_training import (
    FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH,
    FINAL_TRAINING_CONTROLS,
    FINAL_TRAINING_SEEDS,
    POST_TRAINING_MODEL_RULE,
    WINNING_ARCHITECTURE,
    V21AControlPolicyNetwork,
    descriptive_summary,
    five_seed_training_preflight,
)


def test_final_training_contract_is_exactly_frozen() -> None:
    assert FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH == (
        "288b96bbeed75d4ce19793e28198be25430f8dd260604fb654276c405467917a"
    )
    assert FINAL_TRAINING_SEEDS == (17001, 17002, 17003, 17004, 17005)
    assert FINAL_TRAINING_CONTROLS == {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "batch_size": 512,
        "weight_decay": 0.0001,
        "objective": "mean_squared_error",
        "maximum_epochs": 50,
        "early_stopping_patience": 8,
        "improvement_tolerance": 1e-10,
        "gradient_norm_clip": 1.0,
        "checkpoint_selection": "minimum_development_mse",
    }


def test_final_architecture_remains_fixed_at_five_outputs() -> None:
    model = V21AControlPolicyNetwork(WINNING_ARCHITECTURE)
    assert WINNING_ARCHITECTURE.hidden_dimensions == (256, 256, 128)
    assert WINNING_ARCHITECTURE.activation == "silu"
    assert model.parameter_count == 110341
    assert model(torch.zeros((2, 42))).shape == (2, 5)


@pytest.mark.parametrize("role", ["training", "model_selection"])
@pytest.mark.parametrize("split", ["challenge", "ood_stress", "final_sealed_test"])
def test_final_training_cannot_open_post_selection_splits(role: str, split: str) -> None:
    with pytest.raises(PermissionError):
        assert_role_split(role, split)


def test_no_representative_seed_rule_is_invented() -> None:
    assert POST_TRAINING_MODEL_RULE == "absent_stop_before_challenge"


def test_descriptive_summary_uses_sample_standard_deviation() -> None:
    result = descriptive_summary([1, 2, 3, 4, 5])
    assert result == {
        "mean": 3.0,
        "sample_standard_deviation": pytest.approx(1.5811388300841898),
        "median": 3.0,
        "minimum": 1.0,
        "maximum": 5.0,
    }


def test_final_training_preflight_binds_all_frozen_hashes() -> None:
    root = Path(__file__).resolve().parents[2]
    result = five_seed_training_preflight(root, verify_large_file=False)
    assert result["status"] == "PASS"
    assert result["seeds"] == list(FINAL_TRAINING_SEEDS)
    assert result["final_training_configuration_logical_hash"] == (
        FINAL_TRAINING_CONFIGURATION_LOGICAL_HASH
    )
    assert result["post_training_representative_model_rule"] == POST_TRAINING_MODEL_RULE
    assert result["challenge_opened"] is False
    assert result["ood_opened"] is False
    assert result["sealed_opened"] is False


def test_five_seed_provenance_has_canonical_hash_and_no_seed_selection() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "artifacts/research_stage/final_five_seed_training_v2_1a.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    declared = provenance.pop("provenance_logical_sha256")
    assert object_hash(provenance) == declared
    assert provenance["seeds"] == list(FINAL_TRAINING_SEEDS)
    assert provenance["representative_model_selected"] is False
    assert provenance["representative_model_rule_status"] == POST_TRAINING_MODEL_RULE
    assert provenance["challenge_opened"] is False
    assert provenance["ood_opened"] is False
    assert provenance["sealed_opened"] is False
