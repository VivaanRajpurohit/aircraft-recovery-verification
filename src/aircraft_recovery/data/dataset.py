"""Allowlisted loader for compressed demonstration arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


MODE_NAMES = ("normal", "stabilize", "divert", "approach", "land")
TRAINING_ARRAYS = frozenset({
    "observations", "controls", "target_values", "modes", "validity_masks",
    "episode_ids", "scenario_categories", "step_indices", "seeds",
})
FORBIDDEN_TRAINING_TERMS = frozenset({
    "true_state", "ground_truth", "recoverability", "future", "termination_outcome",
    "expert_pid", "airport_scores",
})


class DemonstrationDataset(Dataset[dict[str, torch.Tensor]]):
    """Torch dataset that loads only explicitly training-visible arrays."""

    def __init__(self, dataset_directory: str | Path, episode_ids: set[str] | None = None) -> None:
        self.directory = Path(dataset_directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        archive = np.load(self.directory / "transitions.npz", allow_pickle=False)
        unexpected = set(archive.files) - TRAINING_ARRAYS - {
            "timestamps", "scenario_ids", "known_failures", "previous_modes",
            "current_modes", "termination_flags", "termination_reasons",
            "navigation_targets", "selected_airports",
        }
        if unexpected:
            raise ValueError(f"Dataset contains unrecognized arrays: {sorted(unexpected)}")
        leaked = [name for name in archive.files if any(term in name.lower() for term in FORBIDDEN_TRAINING_TERMS)]
        if leaked:
            raise ValueError(f"Training archive contains forbidden analysis fields: {leaked}")
        all_episode_ids = archive["episode_ids"].astype(str)
        mask = np.ones(len(all_episode_ids), dtype=bool)
        if episode_ids is not None:
            mask = np.isin(all_episode_ids, sorted(episode_ids))
        self.observations = archive["observations"][mask].astype(np.float32, copy=True)
        self.controls = archive["controls"][mask].astype(np.float32, copy=True)
        self.target_values = archive["target_values"][mask].astype(np.float32, copy=True)
        self.modes = archive["modes"][mask].astype(np.int64, copy=True)
        self.validity_masks = archive["validity_masks"][mask].astype(np.float32, copy=True)
        self.episode_ids = all_episode_ids[mask]
        self.scenario_categories = archive["scenario_categories"][mask].astype(str)
        self.seeds = archive["seeds"][mask].astype(np.int64, copy=True)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation": torch.from_numpy(self.observations[index]),
            "controls": torch.from_numpy(self.controls[index]),
            "targets": torch.from_numpy(self.target_values[index]),
            "mode": torch.tensor(self.modes[index], dtype=torch.long),
        }

    def category_counts(self) -> dict[str, int]:
        names, counts = np.unique(self.scenario_categories, return_counts=True)
        return {str(name): int(count) for name, count in zip(names, counts, strict=True)}

    def mode_counts(self) -> dict[str, int]:
        values, counts = np.unique(self.modes, return_counts=True)
        return {MODE_NAMES[int(value)]: int(count) for value, count in zip(values, counts, strict=True)}

    def as_metadata(self) -> dict[str, Any]:
        return {
            "transitions": len(self),
            "episodes": len(set(self.episode_ids.tolist())),
            "category_counts": self.category_counts(),
            "mode_counts": self.mode_counts(),
        }

