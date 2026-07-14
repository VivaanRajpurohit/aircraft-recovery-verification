import numpy as np
import pytest
import torch

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data.preprocessing import ObservationPreprocessor
from aircraft_recovery.training.checkpointing import build_checkpoint, load_checkpoint, save_checkpoint
from aircraft_recovery.training.models import ImitationPolicyNetwork, PolicyArchitectureConfig


def checkpoint_fixture():
    model = ImitationPolicyNetwork(PolicyArchitectureConfig(hidden_dimensions=[16]))
    preprocessor = ObservationPreprocessor()
    preprocessor.fit(np.zeros((2, 42), dtype=np.float32))
    return build_checkpoint(
        model, preprocessor.statistics, {}, "dataset", "split", {"torch": torch.__version__},
        {"total": 1.0}, 1,
    )


def test_checkpoint_round_trip_and_schema(tmp_path) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(path, checkpoint_fixture())
    loaded = load_checkpoint(path)
    assert loaded["observation_feature_names"] == list(OBSERVATION_FEATURES)
    assert loaded["normalization_statistics"]["fitted_transition_count"] == 2


def test_checkpoint_rejects_observation_version_mismatch(tmp_path) -> None:
    checkpoint = checkpoint_fixture()
    checkpoint["observation_vector_version"] = "wrong"
    path = tmp_path / "bad.pt"
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match="version mismatch"):
        load_checkpoint(path)


def test_invalid_checkpoint_is_handled_safely(tmp_path) -> None:
    path = tmp_path / "invalid.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="Unable to load"):
        load_checkpoint(path)

