"""Auditable pre-generation contracts that never execute simulation or training."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.common import file_hash, object_hash, write_json


SPLITS = ("train", "development", "challenge", "ood_stress", "final_sealed_test")


def classify_difficulty_tier(
    failure_count: int, maximum_component_severity: float, *, boundary_case: bool = False
) -> int:
    """Classify the frozen synthetic probes used to test numeric tier boundaries."""
    if failure_count == 0 and maximum_component_severity == 0.0:
        return 0
    if boundary_case and 1 <= failure_count <= 3 and 0.80 <= maximum_component_severity <= 0.85:
        return 5
    if failure_count == 1 and 0.20 <= maximum_component_severity <= 0.35:
        return 1
    if failure_count == 1 and 0.45 <= maximum_component_severity <= 0.60:
        return 2
    if failure_count == 2 and 0.30 <= maximum_component_severity <= 0.60:
        return 3
    if 2 <= failure_count <= 3 and 0.70 <= maximum_component_severity <= 0.80:
        return 4
    raise ValueError("probe does not satisfy exactly one frozen tier contract")


def expand_seed_registry(registry: dict[str, Any]) -> dict[str, list[int]]:
    """Expand declared inclusive ranges and reject count or overlap mistakes."""
    expanded: dict[str, list[int]] = {}
    seen: set[int] = set()
    for split in SPLITS:
        row = registry["dataset_generation"][split]
        values = list(range(int(row["first"]), int(row["last"]) + 1))
        if len(values) != int(row["count"]):
            raise ValueError(f"seed count mismatch: {split}")
        overlap = seen.intersection(values)
        if overlap:
            raise ValueError(f"seed overlap: {split}: {sorted(overlap)[:3]}")
        seen.update(values)
        expanded[split] = values
    return expanded


def validate_tier_contract(contract: dict[str, Any]) -> list[str]:
    """Check completeness and the explicitly ordinal impairment bounds."""
    errors: list[str] = []
    tiers = contract.get("tiers", {})
    if set(map(int, tiers)) != set(range(6)):
        errors.append("tiers_must_be_exactly_zero_through_five")
        return errors
    if list(map(float, tiers[0]["surface_failure_severity"])) != [0.0, 0.0]:
        errors.append("tier_zero_must_have_no_surface_impairment")
    expected = {1: (0.20, 0.35), 2: (0.45, 0.60), 4: (0.70, 0.80), 5: (0.80, 0.85)}
    for tier, bounds in expected.items():
        key = "surface_failure_severity" if tier < 3 else "component_severity"
        if tuple(map(float, tiers[tier][key])) != bounds:
            errors.append(f"unexpected_severity_bounds:tier_{tier}")
    for tier, row in tiers.items():
        if int(tier) and not (row.get("allowed_families") or row.get("allowed_combinations")):
            errors.append(f"missing_allowed_cases:tier_{tier}")
    return errors


def build_generation_manifest(root: Path) -> dict[str, Any]:
    """Build the design freeze manifest without reading or creating dataset records."""
    paths = [
        "configs/training/final_policy_dataset_design_v1.yaml",
        "configs/research/final_tiers_v1.yaml",
        "configs/research/final_metrics_v1.yaml",
        "configs/research/final_monitor_v1.yaml",
        "configs/research/final_fallback_v1.yaml",
        "configs/research/final_evaluation_design_v1.yaml",
        "configs/research/final_heldout_factor_matrix_v1.yaml",
        "configs/research/final_seed_registry_v1.json",
        "artifacts/research_stage/expert_controller_manifest.json",
    ]
    design = load_yaml(root / paths[0])
    registry = json.loads((root / paths[7]).read_text(encoding="utf-8"))
    expanded = expand_seed_registry(registry)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    files = {path: file_hash(root / path) for path in paths}
    manifest: dict[str, Any] = {
        "manifest_id": "final_generation_freeze_v1",
        "status": "DESIGN_FROZEN_GENERATION_NOT_AUTHORIZED",
        "dataset_id": "final_policy_candidate_v1",
        "schema_version": 2,
        "simulator_id": design["declared_simulator"],
        "source_git_commit": git_commit,
        "expected_flights": int(design["total_flights"]),
        "expected_split_counts": design["split_counts"],
        "seed_counts": {key: len(value) for key, value in expanded.items()},
        "files": files,
        "generation_authorized": False,
        "training_authorized": False,
        "final_sealed_test_authorized": False,
        "sealed_test_opened": False,
        "large_dataset_generated": False,
    }
    manifest["freeze_hash"] = object_hash(manifest)
    return manifest


def write_generation_manifest(root: Path) -> Path:
    output = root / "artifacts/research_stage/final_generation_manifest.json"
    write_json(output, build_generation_manifest(root))
    return output
