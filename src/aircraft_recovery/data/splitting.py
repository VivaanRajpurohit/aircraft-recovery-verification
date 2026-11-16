"""Deterministic episode-level train/validation/test/OOD splitting."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from aircraft_recovery.data.hashing import canonical_hash


def create_split_manifest(
    dataset_directory: str | Path,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    ood_categories: set[str] | None = None,
    reserved_test_categories: set[str] | None = None,
    test_reporting_categories: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Split complete episodes, optionally reserving categories as OOD."""
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test fractions must sum to one")
    directory = Path(dataset_directory)
    archive = np.load(directory / "transitions.npz", allow_pickle=False)
    episode_ids = archive["episode_ids"].astype(str)
    categories = archive["scenario_categories"].astype(str)
    seeds = archive["seeds"].astype(int) if "seeds" in archive.files else np.arange(len(episode_ids))
    scenario_ids = archive["scenario_ids"].astype(str) if "scenario_ids" in archive.files else episode_ids
    episode_category: dict[str, str] = {}
    episode_seed: dict[str, int] = {}
    episode_scenario: dict[str, str] = {}
    for episode_id, category, episode_seed_value, scenario_id in zip(
        episode_ids, categories, seeds, scenario_ids, strict=True
    ):
        episode_category.setdefault(str(episode_id), str(category))
        episode_seed.setdefault(str(episode_id), int(episode_seed_value))
        episode_scenario.setdefault(str(episode_id), str(scenario_id))
    ood_set = ood_categories or set()
    ood = sorted([episode for episode, category in episode_category.items() if category in ood_set])
    reserved_categories = reserved_test_categories or set()
    reserved_test = sorted(
        episode for episode, category in episode_category.items()
        if category in reserved_categories and episode not in set(ood)
    )
    remaining = sorted(set(episode_category) - set(ood) - set(reserved_test))
    random.Random(seed).shuffle(remaining)
    count = len(remaining) + len(reserved_test)
    validation_count = max(1, round(count * validation_fraction)) if count >= 3 else 0
    target_test_count = max(1, round(count * test_fraction)) if count >= 3 else 0
    if len(reserved_test) > target_test_count:
        raise ValueError("Reserved test categories exceed the requested test fraction")
    additional_test_count = target_test_count - len(reserved_test)
    if validation_count + target_test_count >= count:
        validation_count = 1 if count >= 3 else 0
        target_test_count = 1 if count >= 3 else 0
        additional_test_count = max(0, target_test_count - len(reserved_test))
    train_count = len(remaining) - validation_count - additional_test_count
    validation = remaining[train_count:train_count + validation_count]
    test = remaining[train_count + validation_count:] + reserved_test
    reporting_groups = {
        name: sorted(
            episode for episode in test
            if episode_category[episode] in reporting_categories
        )
        for name, reporting_categories in (test_reporting_categories or {}).items()
    }
    manifest: dict[str, Any] = {
        "format_version": 1,
        "seed": seed,
        "strategy": "episode_level_seeded_shuffle_with_category_ood",
        "train": sorted(remaining[:train_count]),
        "validation": sorted(validation),
        "test": sorted(test),
        "ood": ood,
        "episode_categories": episode_category,
        "episode_seeds": episode_seed,
        "episode_scenario_ids": episode_scenario,
        "test_reporting_groups": reporting_groups,
    }
    manifest["identifier"] = canonical_hash(manifest)
    (directory / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_no_episode_overlap(manifest: dict[str, Any]) -> None:
    """Raise if any episode occurs in more than one split."""
    splits = [set(manifest[name]) for name in ("train", "validation", "test", "ood")]
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            overlap = left & right
            if overlap:
                raise ValueError(f"Episode leakage across splits: {sorted(overlap)}")


def validate_no_seed_overlap(manifest: dict[str, Any]) -> None:
    """Raise if a random seed occurs in more than one split."""
    episode_seeds = manifest.get("episode_seeds", {})
    split_seeds = [
        {episode_seeds[episode] for episode in manifest[name]}
        for name in ("train", "validation", "test", "ood")
    ]
    for index, left in enumerate(split_seeds):
        for right in split_seeds[index + 1:]:
            overlap = left & right
            if overlap:
                raise ValueError(f"Seed leakage across splits: {sorted(overlap)}")
