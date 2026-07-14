import json

import numpy as np
import pytest

from aircraft_recovery.data.dataset import DemonstrationDataset


def write_metadata(directory) -> None:
    (directory / "metadata.json").write_text(json.dumps({"dataset_identifier": "test"}), encoding="utf-8")


def test_hidden_ground_truth_cannot_enter_training_loader(tmp_path) -> None:
    directory = tmp_path / "dataset"
    directory.mkdir()
    write_metadata(directory)
    common = {
        "observations": np.zeros((1, 42), dtype=np.float32),
        "controls": np.zeros((1, 5), dtype=np.float32),
        "target_values": np.zeros((1, 4), dtype=np.float32),
        "modes": np.zeros(1, dtype=np.int64),
        "validity_masks": np.ones((1, 2), dtype=np.float32),
        "episode_ids": np.asarray(["episode"]),
        "scenario_categories": np.asarray(["category"]),
        "step_indices": np.zeros(1, dtype=np.int32),
        "seeds": np.zeros(1, dtype=np.int64),
        "true_state": np.zeros((1, 7), dtype=np.float32),
    }
    np.savez_compressed(directory / "transitions.npz", **common)
    with pytest.raises(ValueError, match="unrecognized|forbidden"):
        DemonstrationDataset(directory)

