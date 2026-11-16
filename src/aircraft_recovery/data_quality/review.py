"""Stage 3: append-only human review over stored records and validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

from aircraft_recovery.data_quality.common import (
    WORKFLOW_VERSION,
    assert_not_sealed_path,
    file_hash,
    load_jsonl,
    object_hash,
    write_json,
)


Decision = Literal["ACCEPT", "REJECT", "FLAG", "SKIP"]


def _load_context(dataset_directory: str | Path) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    directory = assert_not_sealed_path(dataset_directory)
    records = load_jsonl(directory / "raw" / "transitions.jsonl")
    report = json.loads((directory / "validation" / "validation_report.json").read_text(encoding="utf-8"))
    expected = report.get("validator_report_hash")
    actual = object_hash({key: value for key, value in report.items() if key != "validator_report_hash"})
    if expected != actual:
        raise ValueError("Validator report hash is invalid")
    return directory, records, report


def _latest_decisions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for decision in load_jsonl(path):
        latest[decision["record_id"]] = decision
    return latest


def _write_review_metadata(directory: Path, records: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    decisions_path = directory / "review" / "decisions.jsonl"
    decisions = load_jsonl(decisions_path) if decisions_path.is_file() else []
    latest = _latest_decisions(decisions_path)
    by_scenario: dict[str, list[str]] = defaultdict(list)
    by_outcome: dict[str, list[str]] = defaultdict(list)
    by_severity: dict[str, list[str]] = defaultdict(list)
    record_map = {row["record_id"]: row for row in records}
    for record in records:
        decision = latest.get(record["record_id"])
        value = decision["decision"] if decision else "UNREVIEWED"
        by_scenario[record["scenario_id"]].append(value)
        by_outcome[str(record.get("outcome_label"))].append(value)
        severities = {f"{item.get('failure_type')}:{item.get('severity')}" for item in record.get("active_failures", [])} or {"none"}
        for severity in severities:
            by_severity[severity].append(value)

    def summarize(groups: dict[str, list[str]]) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(groups.items()):
            counts = Counter(values)
            total = len(values)
            result[name] = {
                "total": total,
                **{decision: counts.get(decision, 0) for decision in (
                    "ACCEPT", "REJECT", "FLAG", "SKIP", "UNREVIEWED"
                )},
                "acceptance_rate": counts.get("ACCEPT", 0) / total,
                "rejection_rate": counts.get("REJECT", 0) / total,
                "flag_rate": counts.get("FLAG", 0) / total,
            }
        return result
    summary = {
        "dataset_id": report["dataset_id"],
        "decision_event_count": len(decisions),
        "latest_decision_counts": {
            name: sum(1 for row in latest.values() if row["decision"] == name)
            for name in ("ACCEPT", "REJECT", "FLAG", "SKIP")
        },
        "unreviewed_count": len(records) - len(latest),
        "rates_by_scenario": summarize(by_scenario),
        "rates_by_outcome": summarize(by_outcome),
        "rates_by_failure_severity": summarize(by_severity),
        "failed_recovery_exclusions": [
            {"record_id": record_id, "decision": decision["decision"], "reason_code": decision["reason_code"]}
            for record_id, decision in latest.items()
            if record_map[record_id].get("outcome_label") == "unsuccessful" and decision["decision"] == "REJECT"
        ],
        "scientific_bias_warning": "A valid failed recovery is controller performance data, not automatically bad data.",
    }
    summary["review_summary_hash"] = object_hash(summary)
    write_json(directory / "review" / "review_summary.json", summary)
    manifest = {
        "dataset_id": report["dataset_id"],
        "review_tool_version": WORKFLOW_VERSION,
        "validator_report_hash": report["validator_report_hash"],
        "decisions_append_only": True,
        "decision_event_count": len(decisions),
        "decisions_file_hash": file_hash(decisions_path) if decisions_path.is_file() else None,
        "review_summary_hash": summary["review_summary_hash"],
    }
    manifest["review_manifest_hash"] = object_hash(manifest)
    write_json(directory / "review" / "review_manifest.json", manifest)
    return summary


def append_review_decision(
    dataset_directory: str | Path,
    record_id: str,
    decision: Decision,
    reviewer_id: str,
    reason_code: str | None = None,
    comment: str | None = None,
) -> dict[str, Any]:
    directory, records, report = _load_context(dataset_directory)
    record_map = {row["record_id"]: row for row in records}
    if record_id not in record_map:
        raise KeyError(f"Unknown record ID: {record_id}")
    result_map = {row["record_id"]: row for row in report["record_results"]}
    recommendation = result_map[record_id]["status"]
    override = (
        (recommendation == "FAIL" and decision == "ACCEPT")
        or (recommendation == "PASS" and decision == "REJECT")
        or (recommendation == "REVIEW_REQUIRED" and decision in {"ACCEPT", "REJECT"})
    )
    if override and not reason_code:
        raise ValueError("A written reason code is required to override the automatic recommendation")
    event = {
        "record_id": record_id,
        "decision": decision,
        "reviewer_id": reviewer_id,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "reason_code": reason_code,
        "comment": comment,
        "automatic_recommendation": recommendation,
        "validator_report_hash": report["validator_report_hash"],
        "review_tool_version": WORKFLOW_VERSION,
    }
    event["decision_event_hash"] = object_hash(event)
    decisions_path = directory / "review" / "decisions.jsonl"
    with decisions_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, separators=(",", ":")) + "\n")
    _write_review_metadata(directory, records, report)
    return event


def render_record_context(
    records: list[dict[str, Any]], report: dict[str, Any], index: int, radius: int = 2,
) -> str:
    record = records[index]
    result = {row["record_id"]: row for row in report["record_results"]}[record["record_id"]]
    state_before, state_after = record["aircraft_state_before"], record["aircraft_state_after"]
    lines = [
        f"DATASET {record['dataset_id']}", f"RECORD {record['record_id']}",
        f"flight={record['flight_id']} scenario={record['scenario_id']} seed={record['random_seed']}",
        f"step={record['timestep_index']} sim_time={record['simulation_timestamp_seconds']:.2f}s",
        f"automatic={result['status']} reasons={result['reasons'] or ['none']}",
        f"before={state_before}",
        "observation=" + " ".join(
            f"{name}={value}" for name, value in zip(record["observation_feature_names"], record["ordered_observation_42"], strict=True)
        ),
        f"proposed={record['expert_proposed_controls']}", f"executed={record['executed_controls']}",
        f"failures={record['active_failures']}", f"actuator={record['actuator_state']}",
        f"after={state_after}",
        "delta=" + str({name: float(state_after[name]) - float(state_before[name]) for name in (
            "altitude_ft", "airspeed_kts", "pitch_deg", "roll_deg", "vertical_speed_fpm"
        )}),
        "checks=" + json.dumps(result["checks"], separators=(",", ":")),
        "context=" + ", ".join(
            f"{records[pos]['timestep_index']}@{records[pos]['simulation_timestamp_seconds']:.2f}s"
            for pos in range(max(0, index - radius), min(len(records), index + radius + 1))
            if records[pos]["flight_id"] == record["flight_id"]
        ),
        "Controls: A accept | R reject | F flag | S skip | P previous | N next | C comment | Q save/quit",
    ]
    return "\n".join(lines)


def visualize_record_window(records: list[dict[str, Any]], index: int, radius: int = 10) -> None:
    import matplotlib.pyplot as plt

    record = records[index]
    window = [row for row in records[max(0, index - radius):index + radius + 1] if row["flight_id"] == record["flight_id"]]
    times = [row["simulation_timestamp_seconds"] for row in window]
    figure, axes = plt.subplots(2, 1, sharex=True)
    for name in ("pitch_deg", "roll_deg"):
        axes[0].plot(times, [row["aircraft_state_before"][name] for row in window], label=name)
    for control_index, name in enumerate(("elevator", "aileron", "rudder")):
        axes[1].plot(times, [row["executed_controls"][control_index] for row in window], label=name)
    for axis in axes:
        axis.axvline(record["simulation_timestamp_seconds"], color="black", linestyle="--")
        axis.legend()
    axes[1].set_xlabel("simulated time (s)")
    figure.suptitle(f"{record['record_id']} | failures: {', '.join(f['failure_id'] for f in record['active_failures']) or 'none'}")
    plt.show()
