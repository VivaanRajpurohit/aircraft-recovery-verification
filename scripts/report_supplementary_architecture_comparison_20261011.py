"""Read completed supplementary evidence, validate it, and publish descriptive tables.

No fitting, simulator execution, OOD access, sealed access, or model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ID = "supplementary_architecture_comparison_20261011"
OUT = ROOT / "results" / ID
ART = ROOT / "artifacts/research_stage" / ID
QA = ROOT / "outputs/editorial_jais_20261011"
CONDITIONS = ("unmonitored", "monitored")
MODELS = ("small_reference", "current_reference", "moderate")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def immutable_bytes(path, value):
    if path.exists():
        if path.read_bytes() != value:
            raise FileExistsError(f"Existing completed output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def publish(path, value):
    immutable_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_records(rows, expected_ids=None):
    if len(rows) != 288:
        raise ValueError("Only a complete 144-pair comparison can be reported")
    ids = {r["flight_id"] for r in rows}
    if len(ids) != 144 or (expected_ids is not None and ids != expected_ids):
        raise ValueError("Scenario identities differ from the retained Challenge comparison")
    seen = Counter()
    tiers = Counter()
    for row in rows:
        if row["split"] != "challenge" or row["invalid_experiment"] or row["step_count"] != 260:
            raise ValueError("Forbidden split, invalid flight, or incomplete trace")
        if row["controller_condition"] not in CONDITIONS:
            raise ValueError("Unexpected controller condition")
        seen[(row["flight_id"], row["controller_condition"])] += 1
        tiers[(row["difficulty_tier"], row["controller_condition"])] += 1
    if any(seen[(ident, condition)] != 1 for ident in ids for condition in CONDITIONS):
        raise ValueError("Duplicate or missing paired condition")
    if tiers != Counter({(tier, condition): 24 for tier in range(6) for condition in CONDITIONS}):
        raise ValueError("Six-tier geometry changed")
    return ids


def validate_summary(summary, rows):
    pair_count = summary.get("pair_count", summary.get("paired", {}).get("pair_count"))
    if summary["status"] != "COMPLETE" or pair_count != 144:
        raise ValueError("Partial summary cannot be reported")
    for condition in CONDITIONS:
        selected = [r for r in rows if r["controller_condition"] == condition]
        outcome = summary["overall"][condition]
        if sum(r["strict_recovery_success"] for r in selected) != outcome["strict_recovery"]["successes"]:
            raise ValueError("Recovery count does not match flight records")
        if sum(r["safety"]["flight_has_violation"] for r in selected) != outcome["safety_violation_flights"]["successes"]:
            raise ValueError("Violation count does not match flight records")
        if sum(r["safety"]["violation_property_step_count"] for r in selected) != outcome["safety_violation_property_step_count"]:
            raise ValueError("Property-step count does not match flight records")
        duration = sum(r["safety"]["violation_duration_seconds"] for r in selected)
        if not math.isclose(duration, outcome["safety_violation_duration_seconds"], abs_tol=1e-9):
            raise ValueError("Violation duration does not match flight records")
        times = [r["strict_recovery_time_seconds"] for r in selected if r["strict_recovery_success"]]
        if not times:
            if outcome["strict_recovery_time_seconds"]["count"] != 0:
                raise ValueError("Recovered-flight timing count does not match records")
            continue
        for key, value in (("mean", statistics.mean(times)), ("median", statistics.median(times))):
            if not math.isclose(value, outcome["strict_recovery_time_seconds"][key], abs_tol=1e-9):
                raise ValueError("Recovery onset statistic does not match flight records")


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    immutable_bytes(path, stream.getvalue().encode("utf-8"))


def archive_initial():
    delivery = read(QA / "initial_delivery.json")
    for kind in ("docx", "pdf"):
        if file_hash(Path(delivery[kind])) != delivery[kind + "_sha256"]:
            raise ValueError("Initial manuscript delivery changed; inspect before archiving")
    publish(ART / "initial_manuscript_revision.json", {
        "date": "2026-10-11", "stage": "verified_original_results_framing",
        "delivery": delivery, "diff_audit_sha256": file_hash(QA / "initial_diff.json"),
        "all_25_rendered_pages_visually_reviewed": True,
        "original_title_figures_equations_references_preserved": True,
        "unsupported_final_test_safety_reduction_claim_removed": True,
        "new_partial_challenge_outcomes_used": False,
        "implementation_sha256": file_hash(ROOT / "scripts/revise_jais_performance_framing_20261011.py"),
    })


def verify_checkpoints():
    """Independently reload all three comparison checkpoints using DEVELOPMENT only."""
    if not (OUT / "complete.json").exists():
        raise ValueError("Wait for the full Challenge comparison before final verification")
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from aircraft_recovery.training import v2_1a_architecture_screen as train
    torch.set_num_threads(4)
    approved = ROOT / "datasets/approved_policy_data" / train.APPROVED_ID
    norm = read(approved / "preprocessing/train_only_normalization.json")
    source = train.ArchitectureScreenDataSource(approved)
    dev = train._read_split(source.development_records(), expected_split="development", expected_count=37440, expected_flights=144)
    x = torch.from_numpy(train._normalize(dev.observations, norm))
    y = torch.from_numpy(dev.controls)
    final = read(ROOT / "results/final_five_seed_training_v2_1a/five_seed_training_summary.json")["individual_seed_runs"][0]
    results = []
    for candidate in train.ARCHITECTURE_CANDIDATES:
        if candidate.architecture_id == "moderate":
            metadata = final
            checkpoint = ROOT / final["checkpoint_path"]
            expected = final["checkpoint_sha256"]
        else:
            metadata = read(OUT / "training" / candidate.architecture_id / "complete.json")
            checkpoint = ROOT / metadata["best_checkpoint_path"]
            expected = metadata["best_checkpoint_sha256"]
        if file_hash(checkpoint) != expected:
            raise ValueError("Checkpoint hash changed")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = train.V21AControlPolicyNetwork(candidate)
        model.load_state_dict(state["model_state_dict"])
        if model.parameter_count != candidate.expected_parameter_count or tuple(model(x[:2]).shape) != (2, 5):
            raise ValueError("Model input/output/parameter contract changed")
        actual = train._development_metrics(model, x, y, 512)
        saved = metadata["development_metrics"]
        for key in ("mse", "mae", "rmse"):
            if not np.isclose(actual[key], saved[key], rtol=1e-5, atol=1e-9):
                raise ValueError(f"Independent DEVELOPMENT reload failed: {candidate.architecture_id}: {key}")
        for target in saved["per_output"]:
            for key in ("mae", "rmse"):
                if not np.isclose(actual["per_output"][target][key], saved["per_output"][target][key], rtol=1e-5, atol=1e-9):
                    raise ValueError("Per-output DEVELOPMENT reload failed")
        results.append({"architecture_id": candidate.architecture_id, "checkpoint_sha256": expected,
                        "input_dimension": 42, "output_dimension": 5, "parameter_count": model.parameter_count,
                        "reproduced_development_metrics": actual})
    train.assert_access_policy()
    publish(ART / "independent_checkpoint_verification.json", {
        "status": "PASS", "verification_device": "cpu", "relative_tolerance": 1e-5,
        "records_accessed": "development_only", "models": results,
    })


def report():
    complete = read(OUT / "complete.json")
    if complete["status"] != "COMPLETE" or complete["sealed_opened"] or complete["ood_opened"] or complete["primary_reselected"]:
        raise ValueError("Incomplete comparison or forbidden access/reselection")
    plan = read(ART / "plan.json")
    if complete["plan_logical_sha256"] != plan["plan_logical_sha256"]:
        raise ValueError("Plan hash changed")
    for path, expected in read(ART / "protected_inputs.json")["files"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"Original scientific input changed: {path}")
    retained = ROOT / "results/final_nonsealed_evaluation_v2_1b/challenge"
    original = records(retained / "flight_records.jsonl")
    ids = validate_records(original)
    original_by_id = {r["flight_id"]: r for r in original}
    original_summary = read(retained / "summary.json")
    if file_hash(retained / "summary.json") != complete["retained_moderate_challenge_summary_sha256"]:
        raise ValueError("Retained representative Challenge results changed")
    final = read(ROOT / "results/final_five_seed_training_v2_1a/five_seed_training_summary.json")["individual_seed_runs"][0]
    overall_rows, tier_rows, flight_rows, model_data, hashes = [], [], [], {}, {}
    for ident in MODELS:
        if ident == "moderate":
            rows, summary, training = original, original_summary, final
            checkpoint_hash, checkpoint_path = final["checkpoint_sha256"], final["checkpoint_path"]
            hashes[ident] = {"summary_sha256": file_hash(retained / "summary.json"), "flight_records_sha256": file_hash(retained / "flight_records.jsonl")}
        else:
            directory = OUT / "challenge" / ident
            rows, summary = records(directory / "flight_records.jsonl"), read(directory / "summary.json")
            training = read(OUT / "training" / ident / "complete.json")
            checkpoint_hash, checkpoint_path = training["best_checkpoint_sha256"], training["best_checkpoint_path"]
            validate_records(rows, ids)
            for row in rows:
                for key in ("scenario_instance_id", "random_seed", "difficulty_tier", "scenario_family", "severity_band", "isolated_vs_concurrent", "failure_count"):
                    if row[key] != original_by_id[row["flight_id"]][key]:
                        raise ValueError(f"Scenario metadata mismatch: {key}")
            row_by_key = {(r["flight_id"], r["controller_condition"]): r for r in rows}
            training_directory = OUT / "training" / ident
            for name, expected_key in (("config.json", "configuration_sha256"), ("history.json", "history_sha256"), ("last.pt", "last_checkpoint_sha256")):
                if file_hash(training_directory / name) != training[expected_key]:
                    raise ValueError("Completed training artifact hash changed")
            markers = list(directory.glob("*/pair_complete.json"))
            if len(markers) != 144:
                raise ValueError("Missing paired completion markers")
            condition_hashes = {}
            marker_ids = set()
            for marker in markers:
                item = read(marker)
                marker_ids.add(item["flight_id"])
                if item["status"] != "COMPLETE" or item["plan_logical_sha256"] != plan["plan_logical_sha256"]:
                    raise ValueError("Pair plan identity mismatch")
                for condition in CONDITIONS:
                    path = marker.parent / (condition + ".json.gz")
                    actual_hash = file_hash(path)
                    if actual_hash != item["condition_sha256"][condition]:
                        raise ValueError("Saved condition hash mismatch")
                    value = json.loads(gzip.decompress(path.read_bytes()))
                    if value["checkpoint_sha256"] != checkpoint_hash or value["condition"] != condition or value["flight_id"] != item["flight_id"] or len(value["traces"]) != 260:
                        raise ValueError("Saved checkpoint/condition/flight/trace mismatch")
                    if value["record"] != row_by_key[(item["flight_id"], condition)]:
                        raise ValueError("Aggregated flight record differs from saved paired condition")
                    condition_hashes[path.relative_to(ROOT).as_posix()] = actual_hash
            if marker_ids != ids:
                raise ValueError("Completion-marker scenario identities mismatch")
            hashes[ident] = {"summary_sha256": file_hash(directory / "summary.json"), "flight_records_sha256": file_hash(directory / "flight_records.jsonl"), "paired_condition_files": condition_hashes}
        validate_summary(summary, rows)
        if file_hash(ROOT / checkpoint_path) != checkpoint_hash:
            raise ValueError("Model checkpoint changed")
        architecture = next(a for a in plan["architectures"] if a["architecture_id"] == ident)
        model_data[ident] = {"architecture": architecture, "checkpoint_path": checkpoint_path,
                            "checkpoint_sha256": checkpoint_hash, "training": training, "challenge": summary,
                            "results_reused": ident == "moderate"}
        for condition in CONDITIONS:
            aggregate = summary["overall"][condition]
            rec = aggregate["strict_recovery"]
            timing = aggregate["strict_recovery_time_seconds"]
            violation = aggregate["safety_violation_flights"]
            overall_rows.append({"architecture_id": ident, "condition": condition, "parameters": architecture["parameter_count"],
                "development_mse": training["development_metrics"]["mse"], "seed": 17001,
                "strict_recovery_count": rec["successes"], "flight_count": 144, "strict_recovery_rate": rec["rate"],
                "recovery_wilson_lower": rec["lower"], "recovery_wilson_upper": rec["upper"],
                "mean_recovery_onset_seconds": timing["mean"], "median_recovery_onset_seconds": timing["median"],
                "onset_sample_sd_seconds": timing["sample_standard_deviation"], "onset_minimum_seconds": timing["minimum"], "onset_maximum_seconds": timing["maximum"],
                "violation_flights": violation["successes"], "violation_rate": violation["rate"],
                "violation_property_steps": aggregate["safety_violation_property_step_count"],
                "total_violation_duration_seconds": aggregate["safety_violation_duration_seconds"],
                "monitor_step_frequency": aggregate["monitor_intervention_frequency"], "fallback_step_frequency": aggregate["fallback_step_frequency"],
                "results_reused": ident == "moderate", "checkpoint_sha256": checkpoint_hash})
            for tier in range(6):
                grouped = summary["grouped"][condition]["difficulty_tier"][str(tier)]
                tier_rows.append({"architecture_id": ident, "condition": condition, "tier": tier, "flight_count": grouped["flight_count"],
                    "strict_recovery_count": grouped["strict_recovery"]["successes"], "strict_recovery_rate": grouped["strict_recovery"]["rate"],
                    "mean_recovery_onset_seconds": grouped["strict_recovery_time_seconds"]["mean"],
                    "median_recovery_onset_seconds": grouped["strict_recovery_time_seconds"]["median"],
                    "violation_flights": grouped["safety_violation_flights"]["successes"],
                    "total_violation_duration_seconds": grouped["safety_violation_duration_seconds"]})
        for row in rows:
            flight_rows.append({"architecture_id": ident, "flight_id": row["flight_id"], "scenario_instance_id": row["scenario_instance_id"],
                "condition": row["controller_condition"], "tier": row["difficulty_tier"], "scenario_family": row["scenario_family"],
                "isolated_vs_concurrent": row["isolated_vs_concurrent"], "failure_count": row["failure_count"],
                "strict_recovery_success": row["strict_recovery_success"], "recovery_onset_seconds": row["strict_recovery_time_seconds"],
                "recovery_reference_seconds": row["recovery_timer_reference_seconds"], "violation_flight": row["safety"]["flight_has_violation"],
                "violation_duration_seconds": row["safety"]["violation_duration_seconds"], "monitor_modification_count": row["monitor_modification_count"],
                "fallback_step_count": row["fallback_step_count"], "checkpoint_sha256": checkpoint_hash})
    write_csv(OUT / "architecture_challenge_overall.csv", overall_rows)
    write_csv(OUT / "architecture_challenge_by_tier.csv", tier_rows)
    write_csv(OUT / "architecture_challenge_864_condition_records.csv", flight_rows)
    publish(OUT / "comparison_report.json", {"status": "COMPLETE", "evidence_status": "exploratory_post_original_final_test",
        "plan_logical_sha256": plan["plan_logical_sha256"], "unique_challenge_scenarios": 144,
        "model_scenario_pairs": 432, "condition_records": 864, "new_pairs": 288, "new_condition_records": 576,
        "original_training_fits": 12, "additional_training_fits": 2, "combined_training_fits": 14,
        "combined_distinct_evaluation_scenarios": 432, "combined_model_scenario_pairs": 720, "combined_condition_records": 1440,
        "models": model_data, "overall_table": overall_rows, "tier_table": tier_rows,
        "no_ood_or_sealed_extension": True, "no_primary_reselection": True,
        "latency_ranking_supported": False, "reason": "Representative results reused from a different measurement session",
        "inference_limitations": "One common seed for smaller models; width/depth and activation vary together; exploratory comparison; coupled tier factors",
    })
    manifest = {"complete_record_sha256": file_hash(OUT / "complete.json"), "report_sha256": file_hash(OUT / "comparison_report.json"),
        "plan_logical_sha256": plan["plan_logical_sha256"], "models": hashes,
        "tables": {p.name: file_hash(p) for p in OUT.glob("architecture_challenge*.csv")},
        "dataset_logical_sha256": plan["dataset_logical_sha256"], "normalization_logical_sha256": plan["normalization_logical_sha256"],
        "protected_inputs_verified": True, "ood_opened_by_extension": False, "sealed_opened_by_extension": False, "primary_reselected": False}
    publish(ART / "completed_comparison_provenance.json", manifest)
    print(json.dumps(overall_rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-initial", action="store_true")
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.archive_initial:
        archive_initial()
    elif args.verify_checkpoints:
        verify_checkpoints()
    else:
        report()
