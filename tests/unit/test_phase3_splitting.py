import json

import numpy as np
import pytest

from aircraft_recovery.data.splitting import (
    create_split_manifest,
    validate_no_episode_overlap,
    validate_no_seed_overlap,
)


def make_archive(directory) -> None:
    directory.mkdir()
    episode_ids = np.repeat(np.asarray([f"episode-{index}" for index in range(10)]), 2)
    categories = np.repeat(np.asarray(["ood" if index >= 8 else "trainable" for index in range(10)]), 2)
    np.savez_compressed(
        directory / "transitions.npz",
        episode_ids=episode_ids,
        scenario_categories=categories,
        seeds=np.repeat(np.arange(100, 110), 2),
        scenario_ids=episode_ids,
    )


def test_episode_level_splits_are_deterministic_and_disjoint(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    make_archive(dataset)
    first = create_split_manifest(dataset, 10, 0.7, 0.15, 0.15, {"ood"})
    second = create_split_manifest(dataset, 10, 0.7, 0.15, 0.15, {"ood"})
    assert first == second
    validate_no_episode_overlap(first)
    validate_no_seed_overlap(first)
    assert set(first["ood"]) == {"episode-8", "episode-9"}
    all_ids = first["train"] + first["validation"] + first["test"] + first["ood"]
    assert len(all_ids) == len(set(all_ids)) == 10


def test_overlap_validation_rejects_timestep_leakage() -> None:
    manifest = {"train": ["episode"], "validation": ["episode"], "test": [], "ood": []}
    with pytest.raises(ValueError, match="Episode leakage"):
        validate_no_episode_overlap(manifest)


def test_reserved_test_categories_and_reporting_groups(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    make_archive(dataset)
    manifest = create_split_manifest(
        dataset, 10, 0.7, 0.15, 0.15, reserved_test_categories={"ood"},
        test_reporting_categories={"unseen": {"ood"}},
    )
    assert set(manifest["test"]) == {"episode-8", "episode-9"}
    assert manifest["test_reporting_groups"]["unseen"] == ["episode-8", "episode-9"]
    validate_no_seed_overlap(manifest)
