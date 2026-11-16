from __future__ import annotations

import json
from pathlib import Path

import pytest

from aircraft_recovery.data_quality.common import file_hash, object_hash
from aircraft_recovery.data_quality.final_approved import assert_role_split


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = ROOT / "artifacts/research_stage/final_representative_checkpoint_amendment_v2_1b.json"
FREEZE = ROOT / "artifacts/research_stage/final_model_selection_freeze_v2_1b.json"


def _canonical(path: Path, field: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    declared = value.pop(field)
    assert object_hash(value) == declared
    return value


def test_representative_rule_is_first_seed_and_performance_independent() -> None:
    amendment = _canonical(AMENDMENT, "amendment_logical_sha256")
    assert amendment["frozen_seed_registry"] == [17001, 17002, 17003, 17004, 17005]
    assert amendment["representative_seed"] == amendment["frozen_seed_registry"][0] == 17001
    assert amendment["representative_rule"]["performance_independent"] is True
    assert amendment["representative_rule"]["best_seed_claimed"] is False
    assert amendment["nonsealed_evaluation_checkpoint_set"] == "representative_seed_17001_only"
    assert amendment["optional_all_five_nonsealed_evaluation_authorized"] is False


def test_model_selection_freeze_locks_single_path_and_all_checkpoint_hashes() -> None:
    freeze = _canonical(FREEZE, "freeze_logical_sha256")
    assert freeze["architecture_frozen"] is True
    assert freeze["optimizer_hyperparameters_frozen"] is True
    assert freeze["representative_checkpoint_frozen"] is True
    assert freeze["five_seed_checkpoint_set_frozen"] is True
    assert freeze["further_tuning_or_reselection_permitted"] is False
    assert freeze["nonsealed_evaluation_checkpoint_set"] == "representative_seed_17001_only"
    assert freeze["authorized_stage_path"] == [
        "model_selection_freeze", "challenge", "ood_style_simulator_stress", "pre_sealed_freeze"
    ]
    for checkpoint in freeze["five_seed_checkpoints"]:
        path = (
            ROOT / "checkpoints/final_policy_v1/moderate/final_training_v2_1a"
            / f"seed_{checkpoint['seed']}" / "best.pt"
        )
        assert file_hash(path) == checkpoint["checkpoint_sha256"]


@pytest.mark.parametrize("role", ["training", "model_selection"])
def test_sealed_guard_remains_fail_closed(role: str) -> None:
    with pytest.raises(PermissionError):
        assert_role_split(role, "final_sealed_test")
