import json

import numpy as np
import pytest

from aircraft_recovery.data.splitting import create_split_manifest, validate_no_episode_overlap


def make_archive(directory) -> None:
    directory.mkdir()
    episode_ids = np.repeat(np.asarray([f"episode-{index}" for index in range(10)]), 2)
    categories = np.repeat(np.asarray(["ood" if index >= 8 else "trainable" for index in range(10)]), 2)
    np.savez_compressed(directory / "transitions.npz", episode_ids=episode_ids, scenario_categories=categories)


def test_episode_level_splits_are_deterministic_and_disjoint(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    make_archive(dataset)
    first = create_split_manifest(dataset, 10, 0.7, 0.15, 0.15, {"ood"})
    second = create_split_manifest(dataset, 10, 0.7, 0.15, 0.15, {"ood"})
    assert first == second
    validate_no_episode_overlap(first)
    assert set(first["ood"]) == {"episode-8", "episode-9"}
    all_ids = first["train"] + first["validation"] + first["test"] + first["ood"]
    assert len(all_ids) == len(set(all_ids)) == 10


def test_overlap_validation_rejects_timestep_leakage() -> None:
    manifest = {"train": ["episode"], "validation": ["episode"], "test": [], "ood": []}
    with pytest.raises(ValueError, match="Episode leakage"):
        validate_no_episode_overlap(manifest)

