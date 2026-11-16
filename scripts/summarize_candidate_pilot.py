"""Summarize an unsealed candidate pilot from its serialized on-disk records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from aircraft_recovery.data_quality.common import assert_not_sealed_path, file_hash, object_hash


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
        "standard_deviation": float(np.std(values)),
    }


def summarize(dataset_directory: str | Path) -> dict:
    root = assert_not_sealed_path(dataset_directory)
    transitions = _read_jsonl(root / "raw" / "transitions.jsonl")
    flights = _read_jsonl(root / "raw" / "flights.jsonl")
    validation = json.loads((root / "validation" / "validation_report.json").read_text(encoding="utf-8"))
    leakage = json.loads((root / "validation" / "split_leakage_audit.json").read_text(encoding="utf-8"))
    results = {row["record_id"]: row for row in validation["record_results"]}
    review_reasons: Counter[str] = Counter()
    status_by_flight: dict[str, Counter[str]] = defaultdict(Counter)
    for row in transitions:
        result = results[row["record_id"]]
        status_by_flight[row["flight_id"]][result["status"]] += 1
        if result["status"] == "REVIEW_REQUIRED":
            review_reasons.update(result["reasons"])
    state_names = ("airspeed_kts", "altitude_ft", "pitch_deg", "roll_deg", "vertical_speed_fpm")
    states = {
        name: _distribution(np.asarray([row["aircraft_state_before"][name] for row in transitions], dtype=float))
        for name in state_names
    }
    action_names = ("elevator", "aileron", "rudder", "throttle_left", "throttle_right")
    action_array = np.asarray([row["expert_proposed_controls"] for row in transitions], dtype=float)
    actions = {name: _distribution(action_array[:, index]) for index, name in enumerate(action_names)}
    saturation = np.any(np.abs(action_array[:, :3]) >= 0.99, axis=1) | np.any(
        (action_array[:, 3:] <= 0.01) | (action_array[:, 3:] >= 0.99), axis=1
    )
    report = {
        "dataset_id": validation["dataset_id"],
        "candidate_dataset_path": root.as_posix(),
        "candidate_dataset_files": {
            path.relative_to(root).as_posix(): file_hash(path)
            for path in sorted(root.rglob("*")) if path.is_file()
        },
        "sealed_test_opened": False,
        "flight_count": len(flights),
        "transition_count": len(transitions),
        "record_status_counts": validation["record_status_counts"],
        "validation_integrity_errors": validation["integrity_errors"],
        "leakage_errors": leakage["all_errors"],
        "review_reason_counts": dict(review_reasons.most_common()),
        "flight_review_status": {key: dict(value) for key, value in sorted(status_by_flight.items())},
        "flight_counts_by_split": dict(Counter(row["split"] for row in flights)),
        "flight_counts_by_tier": dict(Counter(str(row["difficulty_tier"]) for row in flights)),
        "flight_counts_by_family": dict(Counter(row["scenario_family"] for row in flights)),
        "flight_counts_by_severity": dict(Counter(row["severity_band"] for row in flights)),
        "flight_outcomes": dict(Counter(row["outcome_label"] for row in flights)),
        "termination_reasons": dict(Counter(str(row["termination_reason"]) for row in flights)),
        "state_distributions": states,
        "expert_proposed_action_distributions": actions,
        "expert_action_saturation_frequency": float(np.mean(saturation)),
        "unique_record_hashes": len({row["record_hash"] for row in transitions}),
        "unique_flight_ids": len({row["flight_id"] for row in flights}),
        "approval_gate": {
            "passed": validation["record_status_counts"]["REVIEW_REQUIRED"] == 0
                      and validation["record_status_counts"]["FAIL"] == 0,
            "reason": (
                "Independent validation has no FAIL or REVIEW_REQUIRED records; export still requires explicit authorization."
                if validation["record_status_counts"]["REVIEW_REQUIRED"] == 0
                and validation["record_status_counts"]["FAIL"] == 0
                else "Every REVIEW_REQUIRED record needs an explicit independent disposition before export."
            ),
            "approved_training_dataset_created": False,
        },
    }
    report["report_hash"] = object_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/candidate_policy_data/final_policy_pilot_v2")
    parser.add_argument("--output", default="artifacts/research_stage/pilot_data_quality_report.json")
    args = parser.parse_args()
    report = summarize(args.dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(
        "# Pilot Data-Quality Report\n\n"
        f"- Dataset: `{report['dataset_id']}`\n"
        f"- Flights/transitions: {report['flight_count']} / {report['transition_count']}\n"
        f"- Record status: {report['record_status_counts']}\n"
        f"- Integrity errors: {report['validation_integrity_errors'] or 'none'}\n"
        f"- Leakage errors: {report['leakage_errors'] or 'none'}\n"
        f"- Splits: {report['flight_counts_by_split']}\n"
        f"- Tiers: {report['flight_counts_by_tier']}\n"
        f"- Families: {report['flight_counts_by_family']}\n"
        f"- Outcomes: {report['flight_outcomes']}\n"
        f"- Proposed-action saturation frequency: {report['expert_action_saturation_frequency']:.6f}\n"
        f"- Approval gate passed: {report['approval_gate']['passed']}\n\n"
        "The candidate records remain immutable. No sealed-test record was opened. "
        "`REVIEW_REQUIRED` is not equivalent to invalid data, but it is not silently treated as approval.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "flight_count", "transition_count", "record_status_counts", "flight_counts_by_split",
        "flight_counts_by_tier", "flight_counts_by_family", "flight_outcomes", "approval_gate",
    )}, indent=2))


if __name__ == "__main__":
    main()
