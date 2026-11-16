from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aircraft_recovery.data_quality.final_approved import assert_role_split
from aircraft_recovery.evaluation.v2_1b_nonsealed import (
    MODEL_SELECTION_FREEZE_LOGICAL_HASH,
    REPRESENTATIVE_CHECKPOINT_HASH,
    REPRESENTATIVE_SEED,
    V21AControlPolicyController,
    continuous_summary,
    nonsealed_preflight,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[2]


def test_representative_only_checkpoint_contract() -> None:
    assert REPRESENTATIVE_SEED == 17001
    assert REPRESENTATIVE_CHECKPOINT_HASH == (
        "175064a9d978edff8542e41d626ecad2e0cb04b4552ed055668043389b369ef7"
    )
    assert MODEL_SELECTION_FREEZE_LOGICAL_HASH == (
        "50cd2c0b355d5776ac3fb11064734a49dc5a76f0c6fd802b8d81e71d4bac5efc"
    )
    controller = V21AControlPolicyController(
        ROOT / "checkpoints/final_policy_v1/moderate/final_training_v2_1a/seed_17001/best.pt",
        torch.device("cpu"),
    )
    assert controller.model.parameter_count == 110341
    assert controller.model.output_dimension == 5


def test_frozen_statistical_methods() -> None:
    interval = wilson_interval(72, 144)
    assert interval["rate"] == 0.5
    assert interval["lower"] == pytest.approx(0.4194027960425476)
    assert interval["upper"] == pytest.approx(0.5805972039574524)
    summary = continuous_summary([1, 2, 3, 4, 5])
    assert summary["mean"] == 3.0
    assert summary["median"] == 3.0
    assert summary["sample_standard_deviation"] == pytest.approx(1.5811388300841898)


def test_access_roles_are_split_specific_and_sealed_fails_closed() -> None:
    assert_role_split("challenge_evaluation", "challenge")
    assert_role_split("ood_evaluation", "ood_stress")
    with pytest.raises(PermissionError):
        assert_role_split("challenge_evaluation", "ood_stress")
    for role in ("challenge_evaluation", "ood_evaluation"):
        with pytest.raises(PermissionError):
            assert_role_split(role, "final_sealed_test")


def test_challenge_preflight_binds_freeze_without_opening_ood_or_sealed() -> None:
    result = nonsealed_preflight(ROOT, "challenge")
    assert result["status"] == "PASS"
    assert result["representative_seed"] == 17001
    assert result["challenge_opened"] is True
    assert result["ood_opened"] is False
    assert result["sealed_opened"] is False


def test_no_sealed_or_all_five_evaluation_entrypoint() -> None:
    for split in ("final_sealed_test", "development", "train"):
        with pytest.raises(PermissionError):
            nonsealed_preflight(ROOT, split)
