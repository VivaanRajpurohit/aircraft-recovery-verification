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
) -> dict[str, Any]:
    """Split complete episodes, optionally reserving categories as OOD."""
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test fractions must sum to one")
    directory = Path(dataset_directory)
    archive = np.load(directory / "transitions.npz", allow_pickle=False)
    episode_ids = archive["episode_ids"].astype(str)
    categories = archive["scenario_categories"].astype(str)
    episode_category: dict[str, str] = {}
    for episode_id, category in zip(episode_ids, categories, strict=True):
        episode_category.setdefault(str(episode_id), str(category))
    ood_set = ood_categories or set()
    ood = sorted([episode for episode, category in episode_category.items() if category in ood_set])
    remaining = sorted(set(episode_category) - set(ood))
    random.Random(seed).shuffle(remaining)
    count = len(remaining)
    validation_count = max(1, round(count * validation_fraction)) if count >= 3 else 0
    test_count = max(1, round(count * test_fraction)) if count >= 3 else 0
    if validation_count + test_count >= count:
        validation_count = 1 if count >= 3 else 0
        test_count = 1 if count >= 3 else 0
    train_count = count - validation_count - test_count
    manifest: dict[str, Any] = {
        "format_version": 1,
        "seed": seed,
        "strategy": "episode_level_seeded_shuffle_with_category_ood",
        "train": sorted(remaining[:train_count]),
        "validation": sorted(remaining[train_count:train_count + validation_count]),
        "test": sorted(remaining[train_count + validation_count:]),
        "ood": ood,
        "episode_categories": episode_category,
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

