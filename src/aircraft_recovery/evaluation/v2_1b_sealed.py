"""Committed-gate final sealed evaluation for the frozen v2.1B representative policy."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import torch

from aircraft_recovery.config import AircraftConfig, load_config, load_yaml
from aircraft_recovery.controllers import FallbackConfig
from aircraft_recovery.data_quality.common import file_hash, object_hash
from aircraft_recovery.evaluation.v2_1b_nonsealed import (
    MODEL_SELECTION_FREEZE_LOGICAL_HASH,
    NORMALIZATION_LOGICAL_HASH,
    REPRESENTATIVE_CHECKPOINT_HASH,
    REPRESENTATIVE_SEED,
    _aggregate,
    _flight_spec,
    _grouped,
    _paired,
    _run_flight,
    _scenario,
)
from aircraft_recovery.safety import SafetyConfig


CANDIDATE_ID = "final_policy_candidate_v2_1"
CANDIDATE_LOGICAL_HASH = "4db2d75ab341e4573019d09176507b03851d7d26dd2eb360d42ecfc2adc28706"
CANDIDATE_FLIGHTS_HASH = "895d043f509c5d06837a9222a40a064a350760fdcf8b47ce25cdb4ac0b0d5c9d"
CANDIDATE_TRANSITIONS_HASH = "5ceed471458518d1790eb81266a20fa7f29a4abd072def2411ad5ce0ac0c1fd3"
GENERATION_FREEZE_HASH = "c0b831ede8a5d75027dcce70ec1c116fc5ec58d479cf500939f2075e3d16d0dc"
SEALED_SPLIT = "final_sealed_test"
EXPECTED_FLIGHTS = 144
EXPECTED_TRANSITIONS = 37440


def _canonical(document: dict[str, Any], field: str) -> str:
    return object_hash({key: value for key, value in document.items() if key != field})


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )


def _require_committed_freeze(root: Path, freeze_path: Path) -> None:
    relative = str(freeze_path.relative_to(root)).replace("\\", "/")
    tracked = _git(root, "ls-files", "--error-unmatch", "--", relative)
    if tracked.returncode != 0:
        raise PermissionError("pre-sealed freeze is not tracked")
    unchanged = _git(root, "diff", "--quiet", "HEAD", "--", relative)
    if unchanged.returncode != 0:
        raise PermissionError("pre-sealed freeze is not committed unchanged at HEAD")


def sealed_preflight(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    freeze_path = root / "artifacts/research_stage/final_presealed_freeze_v2_1b.json"
    if not freeze_path.is_file():
        raise PermissionError("committed pre-sealed freeze is required")
    _require_committed_freeze(root, freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_PRE_SEALED":
        raise PermissionError("pre-sealed freeze status is not active")
    if freeze.get("freeze_logical_sha256") != _canonical(freeze, "freeze_logical_sha256"):
        raise ValueError("pre-sealed freeze logical hash mismatch")
    sealed = freeze.get("sealed_evaluation", {})
    if sealed.get("authorized_after_commit") is not True or sealed.get("access_count_at_freeze") != 0:
        raise PermissionError("sealed evaluation was not prospectively authorized")
    if sealed.get("split") != SEALED_SPLIT or sealed.get("scenario_count") != EXPECTED_FLIGHTS:
        raise ValueError("sealed split geometry is not frozen")
    for relative, expected in freeze.get("frozen_files", {}).items():
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"pre-sealed frozen file mismatch: {relative}")
    for split in ("challenge", "ood_stress"):
        entry = freeze["nonsealed_results"][split]
        for filename, expected in entry["file_sha256"].items():
            path = root / "results/final_nonsealed_evaluation_v2_1b" / split / filename
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"nonsealed result changed after freeze: {split}/{filename}")
    checkpoint = root / "checkpoints/final_policy_v1/moderate/final_training_v2_1a/seed_17001/best.pt"
    if file_hash(checkpoint) != REPRESENTATIVE_CHECKPOINT_HASH:
        raise ValueError("representative checkpoint hash mismatch")
    manifest_path = root / f"datasets/candidate_policy_data/{CANDIDATE_ID}/manifests/candidate_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_logical_hash") != CANDIDATE_LOGICAL_HASH:
        raise ValueError("candidate dataset logical hash mismatch")
    if manifest.get("generation_freeze_hash") != GENERATION_FREEZE_HASH:
        raise ValueError("candidate generation freeze mismatch")
    artifacts = manifest.get("artifact_hashes", {})
    if artifacts.get("raw/flights.jsonl") != CANDIDATE_FLIGHTS_HASH:
        raise ValueError("candidate flight hash is not frozen")
    if artifacts.get("raw/transitions.jsonl") != CANDIDATE_TRANSITIONS_HASH:
        raise ValueError("candidate transition hash is not frozen")
    config_path = root / "configs/evaluation/final_sealed_evaluation_v2_1b.yaml"
    config = load_yaml(config_path)
    required = {
        "split": SEALED_SPLIT,
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_checkpoint_sha256": REPRESENTATIVE_CHECKPOINT_HASH,
        "model_selection_freeze_logical_sha256": MODEL_SELECTION_FREEZE_LOGICAL_HASH,
        "candidate_dataset_logical_sha256": CANDIDATE_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "flight_count": EXPECTED_FLIGHTS,
        "transition_count": EXPECTED_TRANSITIONS,
        "all_five_seed_heldout_evaluation_authorized": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"sealed configuration mismatch: {key}")
    return {
        "status": "PASS",
        "split": SEALED_SPLIT,
        "role": "final_confirmatory_evaluation",
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_checkpoint_sha256": REPRESENTATIVE_CHECKPOINT_HASH,
        "presealed_freeze_logical_sha256": freeze["freeze_logical_sha256"],
        "presealed_freeze_file_sha256": file_hash(freeze_path),
        "evaluation_configuration_sha256": file_hash(config_path),
        "candidate_dataset_logical_sha256": CANDIDATE_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "challenge_opened": True,
        "ood_opened": True,
        "sealed_opened": False,
    }


def _load_sealed_flights(candidate: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (candidate / "raw/flights.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == SEALED_SPLIT:
                rows.append(row)
    if len(rows) != EXPECTED_FLIGHTS:
        raise ValueError(f"sealed flight count mismatch: {len(rows)}")
    return rows


def _expected_sealed_hashes(candidate: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    count = 0
    with (candidate / "raw/transitions.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == SEALED_SPLIT:
                count += 1
                hashes.setdefault(row["flight_id"], row["effective_scenario_configuration_hash"])
    if count != EXPECTED_TRANSITIONS or len(hashes) != EXPECTED_FLIGHTS:
        raise ValueError(f"sealed transition geometry mismatch: {len(hashes)}/{count}")
    return hashes


def run_sealed_evaluation(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    preflight = sealed_preflight(root)
    output = root / "results/final_sealed_evaluation_v2_1b/final_sealed_test"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite sealed evaluation: {output}")
    output.mkdir(parents=True, exist_ok=True)
    access = {
        **preflight,
        "sealed_opened": True,
        "first_outcome_access_utc": datetime.now(timezone.utc).isoformat(),
        "technical_rerun": False,
    }
    (output / "sealed_access_record.json").write_text(
        json.dumps(access, indent=2) + "\n", encoding="utf-8"
    )
    candidate = root / "datasets/candidate_policy_data" / CANDIDATE_ID
    raw_hashes = {
        "raw/flights.jsonl": file_hash(candidate / "raw/flights.jsonl"),
        "raw/transitions.jsonl": file_hash(candidate / "raw/transitions.jsonl"),
    }
    if raw_hashes != {
        "raw/flights.jsonl": CANDIDATE_FLIGHTS_HASH,
        "raw/transitions.jsonl": CANDIDATE_TRANSITIONS_HASH,
    }:
        raise ValueError("candidate raw content changed before sealed evaluation")
    flights = _load_sealed_flights(candidate)
    expected_hashes = _expected_sealed_hashes(candidate)
    aircraft = load_config(root / "configs/aircraft/simple_twin.yaml", AircraftConfig)
    fallback = load_config(root / "configs/research/final_fallback_v1.yaml", FallbackConfig)
    safety = load_config(root / "configs/research/final_monitor_v1.yaml", SafetyConfig)
    metrics = load_yaml(root / "configs/research/final_metrics_v1.yaml")
    checkpoint = root / "checkpoints/final_policy_v1/moderate/final_training_v2_1a/seed_17001/best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traces_path = output / "action_traces.jsonl"
    flight_records: list[dict[str, Any]] = []
    all_latency = {
        condition: {name: [] for name in ("inference", "monitor", "full_controller")}
        for condition in ("unmonitored", "monitored")
    }
    started = datetime.now(timezone.utc)
    timer = perf_counter()
    with traces_path.open("x", encoding="utf-8", newline="\n") as trace_stream:
        for flight in flights:
            scenario = _scenario(root, _flight_spec(flight))
            observed = object_hash(scenario.model_dump(mode="json"))
            if observed != expected_hashes[flight["flight_id"]]:
                raise ValueError(f"sealed scenario reconstruction hash mismatch: {flight['flight_id']}")
            for condition in ("unmonitored", "monitored"):
                record, latencies = _run_flight(
                    root=root,
                    flight=flight,
                    scenario=scenario,
                    condition=condition,
                    checkpoint=checkpoint,
                    device=device,
                    aircraft=aircraft,
                    fallback_config=fallback,
                    safety_config=safety,
                    metrics=metrics,
                    trace_stream=trace_stream,
                )
                flight_records.append(record)
                for name, values in latencies.items():
                    all_latency[condition][name].extend(values)
    records_path = output / "flight_records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in flight_records:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    by_condition = {
        condition: [row for row in flight_records if row["controller_condition"] == condition]
        for condition in ("unmonitored", "monitored")
    }
    summary = {
        "result_version": "final-sealed-evaluation-v2.1b-v1",
        "status": "COMPLETE",
        "split": SEALED_SPLIT,
        "split_wording": "Final sealed test",
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_checkpoint_sha256": REPRESENTATIVE_CHECKPOINT_HASH,
        "controller_conditions": ["unmonitored", "monitored"],
        "overall": {
            condition: _aggregate(by_condition[condition], all_latency[condition])
            for condition in ("unmonitored", "monitored")
        },
        "paired": _paired(flight_records),
        "grouped": {
            condition: _grouped(by_condition[condition])
            for condition in ("unmonitored", "monitored")
        },
        "intervention_taxonomy": {
            condition: {
                key: dict(sum((Counter(row["intervention_taxonomy"][key]) for row in by_condition[condition]), Counter()))
                for key in ("status_counts", "solver_result_counts", "violated_property_counts")
            }
            for condition in ("unmonitored", "monitored")
        },
        "flight_record_count": len(flight_records),
        "trace_record_count": sum(row["step_count"] for row in flight_records),
        "flight_records_path": str(records_path.relative_to(root)).replace("\\", "/"),
        "flight_records_sha256": file_hash(records_path),
        "action_traces_path": str(traces_path.relative_to(root)).replace("\\", "/"),
        "action_traces_sha256": file_hash(traces_path),
        "sealed_access_record_sha256": file_hash(output / "sealed_access_record.json"),
        "candidate_raw_sha256": raw_hashes,
        "evaluation_start_utc": started.isoformat(),
        "evaluation_end_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": perf_counter() - timer,
        "source_git_revision": _git(root, "rev-parse", "HEAD").stdout.strip(),
        "source_file_sha256": file_hash(root / "src/aircraft_recovery/evaluation/v2_1b_sealed.py"),
        "evaluation_configuration_sha256": preflight["evaluation_configuration_sha256"],
        "presealed_freeze_logical_sha256": preflight["presealed_freeze_logical_sha256"],
        "candidate_dataset_logical_sha256": CANDIDATE_LOGICAL_HASH,
        "normalization_logical_sha256": NORMALIZATION_LOGICAL_HASH,
        "challenge_opened": True,
        "ood_opened": True,
        "sealed_opened": True,
        "technical_rerun_count": 0,
        "model_changed_from_results": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
