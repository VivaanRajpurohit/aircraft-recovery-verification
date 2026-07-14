"""Dataset integrity, leakage, units, bounds, and split validation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data.dataset import FORBIDDEN_TRAINING_TERMS
from aircraft_recovery.data.splitting import validate_no_episode_overlap


def validate_dataset(dataset_directory: str | Path) -> dict[str, Any]:
    """Validate a generated dataset and write a machine-readable report."""
    directory = Path(dataset_directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    archive = np.load(directory / "transitions.npz", allow_pickle=False)
    errors: list[str] = []
    observations = archive["observations"]
    controls = archive["controls"]
    if observations.ndim != 2 or observations.shape[1] != 42:
        errors.append("observation_vector_length_is_not_42")
    if not np.isfinite(observations).all():
        errors.append("non_finite_training_observation")
    if controls.shape[1:] != (5,):
        errors.append("control_shape_is_not_5")
    elif not (
        np.all((-1.0 <= controls[:, :3]) & (controls[:, :3] <= 1.0))
        and np.all((0.0 <= controls[:, 3:]) & (controls[:, 3:] <= 1.0))
    ):
        errors.append("control_command_out_of_bounds")
    leaked = [name for name in archive.files if any(term in name.lower() for term in FORBIDDEN_TRAINING_TERMS)]
    if leaked:
        errors.append(f"forbidden_training_fields:{','.join(leaked)}")
    if metadata.get("observation_feature_names") != list(OBSERVATION_FEATURES):
        errors.append("feature_names_or_units_changed")
    nav_mask = observations[:, OBSERVATION_FEATURES.index("navigation_data_valid")]
    airport_mask = observations[:, OBSERVATION_FEATURES.index("airport_available")]
    masks = archive["validity_masks"]
    if not (np.array_equal(nav_mask, masks[:, 0]) and np.array_equal(airport_mask, masks[:, 1])):
        errors.append("validity_masks_do_not_match_observations")
    manifest_path = directory / "split_manifest.json"
    split_sizes: dict[str, int] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            validate_no_episode_overlap(manifest)
        except ValueError as error:
            errors.append(str(error))
        split_sizes = {name: len(manifest[name]) for name in ("train", "validation", "test", "ood")}
        dataset_episodes = set(archive["episode_ids"].astype(str).tolist())
        manifest_episodes = set().union(*(set(manifest[name]) for name in split_sizes))
        if dataset_episodes != manifest_episodes:
            errors.append("split_manifest_does_not_cover_dataset_episodes")
    categories = Counter(archive["scenario_categories"].astype(str).tolist())
    report = {
        "valid": not errors,
        "errors": errors,
        "observation_shape": list(observations.shape),
        "transition_count": len(observations),
        "episode_count": len(set(archive["episode_ids"].astype(str).tolist())),
        "scenario_category_transition_counts": dict(sorted(categories.items())),
        "split_episode_counts": split_sizes,
        "checks": {
            "finite_observations": np.isfinite(observations).all().item(),
            "control_bounds": "control_command_out_of_bounds" not in errors,
            "hidden_ground_truth_excluded": not leaked,
            "validity_masks_match": "validity_masks_do_not_match_observations" not in errors,
            "episode_splits_disjoint": not any("Episode leakage" in error for error in errors),
            "unit_feature_contract_unchanged": "feature_names_or_units_changed" not in errors,
        },
    }
    (directory / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError(f"Dataset validation failed: {errors}")
    return report

