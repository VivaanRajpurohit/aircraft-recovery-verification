"""Frozen v2.1B confirmatory and secondary analysis over completed flight records."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np

from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.common import file_hash
from aircraft_recovery.evaluation.v2_1b_nonsealed import wilson_interval


SPLITS = ("challenge", "ood_stress", "final_sealed_test")
CONDITIONS = ("unmonitored", "monitored")
SPLIT_LABELS = {
    "challenge": "Challenge",
    "ood_stress": "OOD-style simulator stress",
    "final_sealed_test": "Final sealed test",
}


def paired_binary_effect(
    monitored: Iterable[bool | int],
    unmonitored: Iterable[bool | int],
    *,
    seed: int = 19001,
    resamples: int = 10000,
) -> dict[str, Any]:
    left = np.asarray(list(monitored), dtype=float)
    right = np.asarray(list(unmonitored), dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired binary vectors must be nonempty and shape matched")
    differences = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    draws = differences[indices].mean(axis=1)
    both = int(np.sum((left == 1) & (right == 1)))
    monitored_only = int(np.sum((left == 1) & (right == 0)))
    unmonitored_only = int(np.sum((left == 0) & (right == 1)))
    neither = int(np.sum((left == 0) & (right == 0)))
    return {
        "pair_count": int(len(differences)),
        "effect_definition": "monitored_minus_unmonitored_absolute_recovery_rate",
        "absolute_rate_difference": float(differences.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "both_recovered": both,
        "monitored_only_recovered": monitored_only,
        "unmonitored_only_recovered": unmonitored_only,
        "neither_recovered": neither,
        "discordant_pair_count": monitored_only + unmonitored_only,
        "hypothesis_test": "none_not_preregistered",
        "p_value": None,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _paths(root: Path, split: str) -> tuple[Path, Path]:
    if split == "final_sealed_test":
        directory = root / "results/final_sealed_evaluation_v2_1b/final_sealed_test"
    else:
        directory = root / "results/final_nonsealed_evaluation_v2_1b" / split
    return directory / "summary.json", directory / "flight_records.jsonl"


def _validate_completed_split(root: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path, records_path = _paths(root, split)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = _read_jsonl(records_path)
    if summary.get("status") != "COMPLETE" or summary.get("split") != split:
        raise ValueError(f"incomplete split summary: {split}")
    if len(records) != 288 or summary.get("flight_record_count") != 288:
        raise ValueError(f"flight record geometry mismatch: {split}")
    if summary.get("flight_records_sha256") != file_hash(records_path):
        raise ValueError(f"flight record hash mismatch: {split}")
    keys = [(row["flight_id"], row["controller_condition"]) for row in records]
    if len(set(keys)) != 288:
        raise ValueError(f"duplicate flight-condition records: {split}")
    by_flight: dict[str, set[str]] = {}
    for flight_id, condition in keys:
        by_flight.setdefault(flight_id, set()).add(condition)
    if len(by_flight) != 144 or any(value != set(CONDITIONS) for value in by_flight.values()):
        raise ValueError(f"incomplete paired records: {split}")
    return summary, records


def _split_analysis(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_wording": summary["split_wording"],
        "representative_seed": summary["representative_seed"],
        "representative_checkpoint_sha256": summary["representative_checkpoint_sha256"],
        "controller_conditions": summary["controller_conditions"],
        "overall": summary["overall"],
        "paired": summary["paired"],
        "grouped": summary["grouped"],
        "intervention_taxonomy": summary["intervention_taxonomy"],
        "flight_record_count": summary["flight_record_count"],
        "trace_record_count": summary["trace_record_count"],
        "flight_records_sha256": summary["flight_records_sha256"],
        "action_traces_sha256": summary["action_traces_sha256"],
    }


def _training_variability(root: Path) -> dict[str, Any]:
    path = root / "artifacts/research_stage/final_five_seed_training_v2_1a.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        "provenance_logical_sha256": document["provenance_logical_sha256"],
        "seeds": document["seeds"],
        "individual_seed_runs": document["individual_seed_runs"],
        "across_seed_development_summary": document["across_seed_development_summary"],
        "role": "training_seed_variability_only",
        "heldout_seed_selection_permitted": False,
    }


def build_analysis(root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = root / "configs/research/final_analysis_v2_1b.yaml"
    config = load_yaml(config_path)
    summaries: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    inputs: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        summary, rows = _validate_completed_split(root, split)
        summaries[split] = summary
        records[split] = rows
        summary_path, records_path = _paths(root, split)
        inputs[split] = {
            "summary_path": str(summary_path.relative_to(root)).replace("\\", "/"),
            "summary_sha256": file_hash(summary_path),
            "flight_records_path": str(records_path.relative_to(root)).replace("\\", "/"),
            "flight_records_sha256": file_hash(records_path),
            "action_traces_sha256": summary["action_traces_sha256"],
        }
    sealed_by_condition = {
        condition: {
            row["flight_id"]: bool(row["strict_recovery_success"])
            for row in records["final_sealed_test"]
            if row["controller_condition"] == condition
        }
        for condition in CONDITIONS
    }
    flight_ids = sorted(sealed_by_condition["monitored"])
    if flight_ids != sorted(sealed_by_condition["unmonitored"]):
        raise ValueError("sealed pair identities do not match")
    primary = paired_binary_effect(
        [sealed_by_condition["monitored"][key] for key in flight_ids],
        [sealed_by_condition["unmonitored"][key] for key in flight_ids],
        seed=int(config["bootstrap_seed"]),
        resamples=int(config["bootstrap_resamples"]),
    )
    primary.update(
        {
            "split": "final_sealed_test",
            "split_wording": "Final sealed test",
            "endpoint": "strict_recovery_success",
            "monitored": summaries["final_sealed_test"]["overall"]["monitored"]["strict_recovery"],
            "unmonitored": summaries["final_sealed_test"]["overall"]["unmonitored"]["strict_recovery"],
            "confirmatory_status": "primary_confirmatory",
        }
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    analysis = {
        "analysis_version": "final-v2.1b-analysis-v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_revision": revision,
        "analysis_configuration": config,
        "analysis_configuration_sha256": file_hash(config_path),
        "primary_confirmatory_result": primary,
        "secondary_exploratory_results": {
            "split_results": {split: _split_analysis(summaries[split]) for split in SPLITS},
            "five_seed_training_variability": _training_variability(root),
            "interpretation": {
                "challenge": "post-selection evaluation; no tuning or reselection",
                "ood_stress": "OOD-style simulator stress only; no real-world OOD claim",
                "subgroups": "exploratory; small cells are not confirmatory",
            },
        },
        "multiplicity": config["multiple_comparisons"],
        "p_values_generated": False,
        "input_artifacts": inputs,
        "legacy_primary_evidence_used": False,
        "model_changed_after_nonsealed_or_sealed_results": False,
    }
    destination = Path(output) if output else root / "results/final_v2_1b_analysis/analysis.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite final analysis: {destination}")
    destination.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "artifact": str(destination.relative_to(root)).replace("\\", "/"),
        "sha256": file_hash(destination),
        "inputs": inputs,
        "analysis_configuration_sha256": file_hash(config_path),
        "source_file_sha256": file_hash(root / "src/aircraft_recovery/analysis/v2_1b_final.py"),
        "source_git_revision": revision,
    }
    (destination.parent / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return analysis


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_tables(root: str | Path, analysis_path: str | Path | None = None) -> list[Path]:
    root = Path(root).resolve()
    source = Path(analysis_path) if analysis_path else root / "results/final_v2_1b_analysis/analysis.json"
    analysis = json.loads(source.read_text(encoding="utf-8"))
    out = source.parent / "tables"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite final tables: {out}")
    out.mkdir(parents=True, exist_ok=True)
    splits = analysis["secondary_exploratory_results"]["split_results"]
    overall_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for condition in CONDITIONS:
            item = splits[split]["overall"][condition]
            recovery = item["strict_recovery"]
            safety = item["safety_violation_flights"]
            overall_rows.append(
                {
                    "split": SPLIT_LABELS[split],
                    "condition": condition,
                    "flights": item["flight_count"],
                    "strict_recovery_count": recovery["successes"],
                    "strict_recovery_rate": recovery["rate"],
                    "recovery_ci_low": recovery["lower"],
                    "recovery_ci_high": recovery["upper"],
                    "safety_violation_flights": safety["successes"],
                    "safety_violation_rate": safety["rate"],
                    "mean_recovery_time_seconds": item["strict_recovery_time_seconds"]["mean"],
                    "monitor_intervention_frequency": item["monitor_intervention_frequency"],
                    "fallback_step_frequency": item["fallback_step_frequency"],
                    "full_controller_p95_ms": item["latency"]["full_controller"]["p95_ms"],
                }
            )
    primary = analysis["primary_confirmatory_result"]
    primary_rows = [{
        "pairs": primary["pair_count"],
        "monitored_rate": primary["monitored"]["rate"],
        "unmonitored_rate": primary["unmonitored"]["rate"],
        "absolute_rate_difference": primary["absolute_rate_difference"],
        "paired_bootstrap_ci_low": primary["paired_bootstrap_95_ci"][0],
        "paired_bootstrap_ci_high": primary["paired_bootstrap_95_ci"][1],
        "both_recovered": primary["both_recovered"],
        "monitored_only_recovered": primary["monitored_only_recovered"],
        "unmonitored_only_recovered": primary["unmonitored_only_recovered"],
        "neither_recovered": primary["neither_recovered"],
        "p_value": "not_preregistered",
    }]
    sealed = splits["final_sealed_test"]
    group_tables: dict[str, list[dict[str, Any]]] = {}
    for grouping in ("difficulty_tier", "isolated_vs_concurrent", "scenario_family"):
        rows: list[dict[str, Any]] = []
        levels = sorted(sealed["grouped"]["monitored"][grouping], key=str)
        for level in levels:
            for condition in CONDITIONS:
                item = sealed["grouped"][condition][grouping][str(level)]
                rows.append({
                    "group": level,
                    "condition": condition,
                    "flights": item["flight_count"],
                    "recovery_count": item["strict_recovery"]["successes"],
                    "recovery_rate": item["strict_recovery"]["rate"],
                    "safety_violation_count": item["safety_violation_flights"]["successes"],
                    "safety_violation_rate": item["safety_violation_flights"]["rate"],
                    "mean_recovery_time_seconds": item["strict_recovery_time_seconds"]["mean"],
                })
        group_tables[grouping] = rows
    training = analysis["secondary_exploratory_results"]["five_seed_training_variability"]
    training_rows = []
    for run in training["individual_seed_runs"]:
        metrics = run["development_metrics"]
        training_rows.append({
            "seed": run["seed"], "epochs_completed": run["epochs_completed"], "best_epoch": run["best_epoch"],
            "development_mse": metrics["mse"], "development_mae": metrics["mae"], "development_rmse": metrics["rmse"],
            "checkpoint_sha256": run["checkpoint_sha256"],
        })
    taxonomy_rows = []
    for split in SPLITS:
        taxonomy = splits[split]["intervention_taxonomy"]["monitored"]
        for taxonomy_type, counts in taxonomy.items():
            for category, count in sorted(counts.items()):
                taxonomy_rows.append({"split": SPLIT_LABELS[split], "taxonomy_type": taxonomy_type, "category": category, "count": count})
    outputs = [
        out / "overall_by_split_and_condition.csv",
        out / "sealed_primary_paired_effect.csv",
        out / "sealed_by_tier.csv",
        out / "sealed_by_concurrency.csv",
        out / "sealed_by_failure_family.csv",
        out / "five_seed_development_metrics.csv",
        out / "intervention_taxonomy.csv",
    ]
    for path, rows in zip(outputs, [overall_rows, primary_rows, group_tables["difficulty_tier"], group_tables["isolated_vs_concurrent"], group_tables["scenario_family"], training_rows, taxonomy_rows], strict=True):
        _write_csv(path, rows)
    return outputs
