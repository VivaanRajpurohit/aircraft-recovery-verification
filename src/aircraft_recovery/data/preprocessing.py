"""Train-only observation normalization with preserved mask semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aircraft_recovery.common.observations import OBSERVATION_FEATURES


BINARY_OR_MASK_FEATURES = frozenset({
    "elevator_stuck", "aileron_stuck", "rudder_stuck", "left_engine_failed",
    "right_engine_failed", "asymmetric_thrust", "navigation_data_valid",
    "emergency_declared", "airport_available", "nearest_airport_within_glide_range",
})
MASK_INDICES = tuple(OBSERVATION_FEATURES.index(name) for name in BINARY_OR_MASK_FEATURES)


@dataclass(frozen=True)
class PreprocessingStatistics:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    normalized: tuple[bool, ...]
    clip_standard_deviations: float
    fitted_transition_count: int
    observation_vector_version: str = "phase1-42-v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["feature_names"] = list(self.feature_names)
        value["means"] = list(self.means)
        value["standard_deviations"] = list(self.standard_deviations)
        value["normalized"] = list(self.normalized)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PreprocessingStatistics":
        return cls(
            feature_names=tuple(value["feature_names"]),
            means=tuple(value["means"]),
            standard_deviations=tuple(value["standard_deviations"]),
            normalized=tuple(value["normalized"]),
            clip_standard_deviations=float(value["clip_standard_deviations"]),
            fitted_transition_count=int(value["fitted_transition_count"]),
            observation_vector_version=str(value["observation_vector_version"]),
        )


class ObservationPreprocessor:
    """Replace non-finite values, z-score continuous fields, preserve masks."""

    def __init__(self, statistics: PreprocessingStatistics | None = None) -> None:
        self.statistics = statistics

    def fit(self, training_observations: np.ndarray, clip_standard_deviations: float = 5.0) -> PreprocessingStatistics:
        if training_observations.ndim != 2 or training_observations.shape[1] != 42:
            raise ValueError("Training observations must have shape [N, 42]")
        finite = np.nan_to_num(training_observations.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        means = finite.mean(axis=0)
        standard_deviations = finite.std(axis=0)
        standard_deviations[standard_deviations < 1e-6] = 1.0
        normalized = np.ones(42, dtype=bool)
        normalized[list(MASK_INDICES)] = False
        means[~normalized] = 0.0
        standard_deviations[~normalized] = 1.0
        self.statistics = PreprocessingStatistics(
            feature_names=tuple(OBSERVATION_FEATURES),
            means=tuple(float(value) for value in means),
            standard_deviations=tuple(float(value) for value in standard_deviations),
            normalized=tuple(bool(value) for value in normalized),
            clip_standard_deviations=clip_standard_deviations,
            fitted_transition_count=len(training_observations),
        )
        return self.statistics

    def transform(self, observations: np.ndarray) -> np.ndarray:
        statistics = self._validated_statistics()
        array = np.asarray(observations, dtype=np.float32)
        was_vector = array.ndim == 1
        if was_vector:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != 42:
            raise ValueError("Observations must have final dimension 42")
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        means = np.asarray(statistics.means, dtype=np.float32)
        std = np.asarray(statistics.standard_deviations, dtype=np.float32)
        normalized = np.asarray(statistics.normalized, dtype=bool)
        result = array.copy()
        result[:, normalized] = (result[:, normalized] - means[normalized]) / std[normalized]
        result[:, normalized] = np.clip(
            result[:, normalized],
            -statistics.clip_standard_deviations,
            statistics.clip_standard_deviations,
        )
        return result[0] if was_vector else result

    def _validated_statistics(self) -> PreprocessingStatistics:
        if self.statistics is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        if self.statistics.feature_names != tuple(OBSERVATION_FEATURES):
            raise ValueError("Preprocessor feature names do not match observation vector")
        if self.statistics.observation_vector_version != "phase1-42-v1":
            raise ValueError("Unsupported observation-vector version")
        return self.statistics

