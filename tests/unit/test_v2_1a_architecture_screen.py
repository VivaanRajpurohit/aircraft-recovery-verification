from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data_quality.common import CONTROL_NAMES, object_hash
from aircraft_recovery.data_quality.final_approved import assert_role_split
from aircraft_recovery.training.v2_1a_architecture_screen import (
    AMENDMENT_LOGICAL_HASH,
    ARCHITECTURE_CANDIDATES,
    ArchitectureScreenDataSource,
    FORBIDDEN_SPLITS,
    INPUT_DIMENSION,
    OUTPUT_DIMENSION,
    TARGET_ORDER,
    V21AControlPolicyNetwork,
    assert_access_policy,
    validate_supervised_record,
)


def _record(split: str = "train") -> dict:
    return {
        "split": split,
        "observation_feature_names": list(OBSERVATION_FEATURES),
        "ordered_observation_42": [0.0] * INPUT_DIMENSION,
        "expert_proposed_controls": [0.0, 0.0, 0.0, 0.5, 0.5],
        "label_semantics": "expert_proposed_action",
        "flight_id": "flight-1",
        "record_id": "record-1",
    }


def test_network_output_dimension_is_exactly_five() -> None:
    for candidate in ARCHITECTURE_CANDIDATES:
        model = V21AControlPolicyNetwork(candidate)
        output = model(torch.zeros((3, INPUT_DIMENSION)))
        assert model.output_dimension == OUTPUT_DIMENSION == 5
        assert output.shape == (3, 5)
        assert set(model.state_dict()) == {
            *(f"trunk.{index}.{name}" for index in range(0, 2 * len(candidate.hidden_dimensions), 2) for name in ("weight", "bias")),
            "control_head.weight",
            "control_head.bias",
        }


def test_independently_computed_parameter_counts() -> None:
    expected = [14085, 44549, 110341]
    assert [V21AControlPolicyNetwork(item).parameter_count for item in ARCHITECTURE_CANDIDATES] == expected
    assert [item.expected_parameter_count for item in ARCHITECTURE_CANDIDATES] == expected


def test_supervised_target_dimension_and_order_are_fixed() -> None:
    assert TARGET_ORDER == tuple(CONTROL_NAMES)
    assert TARGET_ORDER == (
        "elevator", "aileron", "rudder", "throttle_left", "throttle_right"
    )
    validate_supervised_record(_record(), "train")
    bad = _record()
    bad["expert_proposed_controls"] = [0.0] * 4
    with pytest.raises(ValueError, match="target dimension"):
        validate_supervised_record(bad, "train")


def test_data_source_exposes_only_train_and_development_roles() -> None:
    calls: list[tuple[str, str]] = []

    def loader(_dataset, *, split: str, role: str):
        calls.append((split, role))
        return iter([_record(split)])

    source = ArchitectureScreenDataSource("approved", loader=loader)
    assert list(source.training_records())[0]["split"] == "train"
    assert list(source.development_records())[0]["split"] == "development"
    assert calls == [("train", "training"), ("development", "model_selection")]
    assert not hasattr(source, "records_for_split")


@pytest.mark.parametrize("role", ["training", "model_selection"])
@pytest.mark.parametrize("split", sorted(FORBIDDEN_SPLITS))
def test_optimization_roles_reject_challenge_ood_and_sealed(role: str, split: str) -> None:
    with pytest.raises(PermissionError):
        assert_role_split(role, split)


def test_sealed_guard_fails_closed() -> None:
    for role in ("training", "model_selection"):
        with pytest.raises(PermissionError, match="final_sealed_test"):
            assert_role_split(role, "final_sealed_test")
    assert_access_policy()


def test_amendment_canonical_hash_and_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "artifacts/research_stage/final_training_protocol_amendment_v2_1a.json"
    amendment = json.loads(path.read_text(encoding="utf-8"))
    declared = amendment.pop("amendment_logical_hash")
    assert declared == AMENDMENT_LOGICAL_HASH
    assert object_hash(amendment) == AMENDMENT_LOGICAL_HASH
    assert amendment["output_contract"]["dimension"] == 5
    assert tuple(amendment["output_contract"]["target_order"]) == TARGET_ORDER
    assert amendment["optimization"]["checkpoint_selection"] == "minimum_development_mse"


def test_control_bounds_are_enforced_by_output_activation() -> None:
    model = V21AControlPolicyNetwork(ARCHITECTURE_CANDIDATES[0])
    output = model(torch.randn((32, INPUT_DIMENSION)) * 100.0)
    assert torch.all(output[:, :3] >= -1.0)
    assert torch.all(output[:, :3] <= 1.0)
    assert torch.all(output[:, 3:] >= 0.0)
    assert torch.all(output[:, 3:] <= 1.0)
