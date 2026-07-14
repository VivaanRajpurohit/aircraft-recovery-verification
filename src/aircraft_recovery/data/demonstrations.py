"""Closed-loop expert demonstration generation with separated audit truth."""

from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aircraft_recovery.common.observations import OBSERVATION_FEATURES, observation_to_vector
from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import DeterministicFallbackController, FallbackConfig
from aircraft_recovery.data.dataset import MODE_NAMES
from aircraft_recovery.data.hashing import canonical_hash
from aircraft_recovery.simulator import SimpleAircraftSimulator


class ScenarioCategoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=1)
    scenario: str
    weight: float = Field(gt=0.0)


class DemonstrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str
    output_directory: str
    seed: int = Field(ge=0)
    episode_count: int = Field(gt=2)
    aircraft_config: str
    fallback_config: str
    scenarios: list[ScenarioCategoryConfig] = Field(min_length=1)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    ood_categories: list[str] = Field(default_factory=list)


def _category_schedule(config: DemonstrationConfig) -> list[ScenarioCategoryConfig]:
    total_weight = sum(item.weight for item in config.scenarios)
    exact = [config.episode_count * item.weight / total_weight for item in config.scenarios]
    counts = [int(value) for value in exact]
    if config.episode_count >= len(config.scenarios):
        counts = [max(1, count) for count in counts]
    while sum(counts) > config.episode_count:
        candidates = [index for index, count in enumerate(counts) if count > 1]
        counts[min(candidates, key=lambda index: exact[index] - counts[index])] -= 1
    while sum(counts) < config.episode_count:
        index = max(range(len(counts)), key=lambda item: exact[item] - counts[item])
        counts[index] += 1
    schedule = [item for item, count in zip(config.scenarios, counts, strict=True) for _ in range(count)]
    random.Random(config.seed).shuffle(schedule)
    return schedule


def _array_content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def generate_demonstrations(config: DemonstrationConfig) -> dict[str, Any]:
    """Generate an expert dataset without admitting audit fields to tensors."""
    output = Path(config.output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Dataset output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    aircraft = load_config(config.aircraft_config, AircraftConfig)
    fallback_config = load_config(config.fallback_config, FallbackConfig)
    records: dict[str, list[Any]] = {name: [] for name in (
        "observations", "controls", "target_values", "modes", "validity_masks",
        "episode_ids", "scenario_categories", "step_indices", "seeds", "timestamps",
        "scenario_ids", "known_failures", "previous_modes", "current_modes",
        "termination_flags", "termination_reasons", "navigation_targets", "selected_airports",
    )}
    category_episode_counts: Counter[str] = Counter()
    category_transition_counts: Counter[str] = Counter()
    analysis_path = output / "analysis_ground_truth.jsonl"
    with analysis_path.open("w", encoding="utf-8") as analysis_stream:
        for episode_index, category_config in enumerate(_category_schedule(config)):
            scenario = load_config(category_config.scenario, ScenarioConfig)
            episode_seed = config.seed + episode_index
            scenario = ScenarioConfig.model_validate({
                **scenario.model_dump(),
                "seed": episode_seed,
            })
            episode_id = f"{scenario.scenario_id}-{episode_seed:08d}-{episode_index:04d}"
            simulator = SimpleAircraftSimulator(aircraft, scenario)
            expert = DeterministicFallbackController(fallback_config, scenario.airports)
            observation = simulator.reset(episode_seed)
            expert.reset()
            episode_row_indices: list[int] = []
            step_index = 0
            while not simulator.is_terminal():
                vector = observation_to_vector(observation)
                previous_mode = expert.state_machine.mode.value
                truth_before = simulator.audit_snapshot()
                action = expert.act(observation)
                current_mode = expert.state_machine.mode.value
                next_observation = simulator.step(action)
                truth_after = simulator.audit_snapshot()
                commands = action.control_commands
                targets = action.stabilization_targets
                navigation = action.navigation_commands
                row_index = len(records["observations"])
                episode_row_indices.append(row_index)
                records["observations"].append(vector)
                records["controls"].append([
                    commands.elevator, commands.aileron, commands.rudder,
                    commands.throttle_left, commands.throttle_right,
                ])
                records["target_values"].append([
                    targets.target_pitch_deg, targets.target_roll_deg,
                    targets.target_airspeed_kts, targets.target_altitude_ft,
                ])
                records["modes"].append(MODE_NAMES.index(action.emergency_mode))
                records["validity_masks"].append([
                    float(observation.navigation.data_valid),
                    float(bool(observation.navigation.nearest_airports)),
                ])
                records["episode_ids"].append(episode_id)
                records["scenario_categories"].append(category_config.category)
                records["step_indices"].append(step_index)
                records["seeds"].append(episode_seed)
                records["timestamps"].append(observation.timestamp_seconds)
                records["scenario_ids"].append(scenario.scenario_id)
                records["known_failures"].append(json.dumps(observation.active_failures, separators=(",", ":")))
                records["previous_modes"].append(previous_mode)
                records["current_modes"].append(current_mode)
                records["termination_flags"].append(False)
                records["termination_reasons"].append("")
                records["navigation_targets"].append([
                    navigation.target_heading_deg,
                    navigation.selected_runway_heading_deg if navigation.selected_runway_heading_deg is not None else -1.0,
                    float(navigation.selected_airport is not None),
                ])
                records["selected_airports"].append(navigation.selected_airport or "")
                analysis_stream.write(json.dumps({
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "ground_truth_before": truth_before,
                    "ground_truth_after": truth_after,
                }, allow_nan=False, separators=(",", ":")) + "\n")
                observation = next_observation
                step_index += 1
            if episode_row_indices:
                records["termination_flags"][episode_row_indices[-1]] = True
                records["termination_reasons"][episode_row_indices[-1]] = simulator.termination_reason() or "unknown"
            category_episode_counts[category_config.category] += 1
            category_transition_counts[category_config.category] += step_index

    arrays = {
        "observations": np.asarray(records["observations"], dtype=np.float32),
        "controls": np.asarray(records["controls"], dtype=np.float32),
        "target_values": np.asarray(records["target_values"], dtype=np.float32),
        "modes": np.asarray(records["modes"], dtype=np.int64),
        "validity_masks": np.asarray(records["validity_masks"], dtype=np.float32),
        "episode_ids": np.asarray(records["episode_ids"], dtype="U96"),
        "scenario_categories": np.asarray(records["scenario_categories"], dtype="U64"),
        "step_indices": np.asarray(records["step_indices"], dtype=np.int32),
        "seeds": np.asarray(records["seeds"], dtype=np.int64),
        "timestamps": np.asarray(records["timestamps"], dtype=np.float64),
        "scenario_ids": np.asarray(records["scenario_ids"], dtype="U64"),
        "known_failures": np.asarray(records["known_failures"], dtype="U512"),
        "previous_modes": np.asarray(records["previous_modes"], dtype="U32"),
        "current_modes": np.asarray(records["current_modes"], dtype="U32"),
        "termination_flags": np.asarray(records["termination_flags"], dtype=bool),
        "termination_reasons": np.asarray(records["termination_reasons"], dtype="U64"),
        "navigation_targets": np.asarray(records["navigation_targets"], dtype=np.float32),
        "selected_airports": np.asarray(records["selected_airports"], dtype="U16"),
    }
    np.savez_compressed(output / "transitions.npz", **arrays)
    dataset_identifier = _array_content_hash(arrays)
    metadata = {
        "format_version": 1,
        "dataset_name": config.dataset_name,
        "dataset_identifier": dataset_identifier,
        "observation_vector_version": "phase1-42-v1",
        "observation_feature_names": list(OBSERVATION_FEATURES),
        "training_visible_fields": [
            "observations", "controls", "target_values", "modes", "validity_masks"
        ],
        "analysis_only_fields": ["ground_truth_before", "ground_truth_after"],
        "metadata_fields": [
            "episode_ids", "seeds", "scenario_ids", "scenario_categories",
            "step_indices", "timestamps", "known_failures", "previous_modes",
            "current_modes", "termination_flags", "termination_reasons",
            "navigation_targets", "selected_airports",
        ],
        "episode_count": config.episode_count,
        "transition_count": len(arrays["observations"]),
        "scenario_category_episode_counts": dict(sorted(category_episode_counts.items())),
        "scenario_category_transition_counts": dict(sorted(category_transition_counts.items())),
        "configuration_hash": canonical_hash(config.model_dump(mode="json")),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "pydantic": version("pydantic"),
            "torch": version("torch"),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
