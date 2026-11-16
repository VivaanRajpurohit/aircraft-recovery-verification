"""Create auditable flight-level review evidence without changing candidate raw data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from aircraft_recovery.data_quality.common import assert_not_sealed_path, file_hash, load_jsonl, object_hash, write_json
from aircraft_recovery.data_quality.contracts import flight_contract_checks
from aircraft_recovery.data_quality.validation import VALIDATOR_VERSION


OLD_ACTUATOR_REASON = "actuator_degradation_requires_human_review_of_serialized_history"
OLD_SENSOR_REASON = "observed_true_state_difference_under_declared_sensor_fault"


def _ranges(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    output: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            output.append([start, previous])
            start = value
        previous = value
    output.append([start, previous])
    return output


def _minmax(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    names = ("airspeed_kts", "altitude_ft", "pitch_deg", "roll_deg", "vertical_speed_fpm")
    return {name: {
        "minimum": min(float(row["aircraft_state_before"][name]) for row in rows),
        "maximum": max(float(row["aircraft_state_before"][name]) for row in rows),
    } for name in names}


def _representatives(rows: list[dict[str, Any]], result_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    flagged = [index for index, row in enumerate(rows) if result_map[row["record_id"]]["status"] == "REVIEW_REQUIRED"]
    selected = sorted(set(flagged[:1] + flagged[len(flagged) // 2:len(flagged) // 2 + 1] + flagged[-1:]))
    packets = []
    for index in selected:
        context = []
        for neighbor in range(max(0, index - 1), min(len(rows), index + 2)):
            row = rows[neighbor]
            context.append({
                "record_id": row["record_id"], "record_hash": row["record_hash"],
                "timestep_index": row["timestep_index"],
                "timestamp_seconds": row["simulation_timestamp_seconds"],
                "validator_status_before": result_map[row["record_id"]]["status"],
                "validator_reasons_before": result_map[row["record_id"]]["reasons"],
                "true_state": row["aircraft_state_before"],
                "observed_state_42": row["ordered_observation_42"],
                "proposed_controls": row["expert_proposed_controls"],
                "executed_controls": row["executed_controls"],
                "actuator_effectiveness": row.get("actuator_effectiveness"),
                "actuator_delay_seconds": row.get("actuator_delay_seconds"),
            })
        packets.append({"representative_record_id": rows[index]["record_id"], "neighboring_context": context})
    return packets


def capture_before(dataset: Path, output: Path) -> None:
    report_path = dataset / "validation" / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = load_jsonl(dataset / "raw" / "transitions.jsonl")
    flights = load_jsonl(dataset / "raw" / "flights.jsonl")
    result_map = {row["record_id"]: row for row in report["record_results"]}
    by_flight: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_flight[row["flight_id"]].append(row)
    baseline_path = Path("artifacts/research_stage/pilot_difficulty_baselines.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {"rows": []}
    expert_results = {(row["scenario_id"], int(row["seed"])): row for row in baseline.get("rows", [])
                      if row["controller"] == "phase2_deterministic_fallback"}
    flight_meta = {row["flight_id"]: row for row in flights}
    timestamp = datetime.now(timezone.utc).isoformat()
    contract_by_record: dict[str, list[dict[str, Any]]] = {}
    for flight_rows in by_flight.values():
        ordered = sorted(flight_rows, key=lambda row: int(row["timestep_index"]))
        contract_by_record.update(flight_contract_checks(ordered))

    flagged = [row for row in rows if result_map[row["record_id"]]["status"] == "REVIEW_REQUIRED"]
    reason_counts = Counter(reason for row in flagged for reason in result_map[row["record_id"]]["reasons"])
    scenario_reason_counts = Counter((row["scenario_id"], reason) for row in flagged
                                     for reason in result_map[row["record_id"]]["reasons"])
    failure_counts = Counter()
    severity_counts = Counter()
    sensor_magnitudes = Counter()
    for row in flagged:
        for failure in row.get("ground_truth_failures", []):
            failure_counts[str(failure.get("failure_type"))] += 1
            severity_counts[f"{failure.get('failure_type')}:{failure.get('severity')}"] += 1
            if failure.get("failure_type") in {"airspeed_bias", "altitude_bias", "heading_bias"}:
                magnitude = float(failure.get("intensity", 0.0)) * float(failure.get("parameters", {}).get("maximum_bias", 0.0))
                sensor_magnitudes[f"{failure.get('failure_type')}:{magnitude:.9g}"] += 1
    root_causes = [
        {
            "class_id": "RC-ACT-001", "validator_reason": OLD_ACTUATOR_REASON,
            "occurrences": reason_counts[OLD_ACTUATOR_REASON], "classification": "VALIDATOR_BUG",
            "underlying_behavior": "EXPECTED_BY_MODEL",
            "explanation": "The old rule flagged every degradation/delay record without reconstructing the serialized command history. Effectiveness is applied in state dynamics, not to stored executed_controls; delay is independently recoverable from prior proposed controls.",
        },
        {
            "class_id": "RC-SEN-001", "validator_reason": OLD_SENSOR_REASON,
            "occurrences": reason_counts[OLD_SENSOR_REASON], "classification": "VALIDATOR_BUG",
            "underlying_behavior": "EXPECTED_BY_MODEL",
            "explanation": "The old rule detected a true/observed difference but did not evaluate the deterministic bias contract. Ground truth and controller observation are stored separately and the observed bias equals intensity times maximum_bias.",
        },
    ]
    packets_dir = output / "flight_packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    packet_inventory = []
    dispositions = []
    for flight_id, flight_rows in sorted(by_flight.items()):
        ordered = sorted(flight_rows, key=lambda row: int(row["timestep_index"]))
        flagged_rows = [row for row in ordered if result_map[row["record_id"]]["status"] == "REVIEW_REQUIRED"]
        if not flagged_rows:
            continue
        meta = flight_meta[flight_id]
        grouped_reasons = Counter(reason for row in flagged_rows for reason in result_map[row["record_id"]]["reasons"])
        reason_steps: dict[str, list[int]] = defaultdict(list)
        for row in flagged_rows:
            for reason in result_map[row["record_id"]]["reasons"]:
                reason_steps[reason].append(int(row["timestep_index"]))
        failures = ordered[0].get("failure_schedule", [])
        all_contract_pass = all(
            all(check["status"] == "PASS" for check in contract_by_record[row["record_id"]])
            for row in flagged_rows
        )
        expert = expert_results.get((meta["scenario_id"], int(meta["random_seed"])))
        packet = {
            "packet_version": 1, "created_utc": timestamp, "flight_id": flight_id,
            "scenario_id": meta["scenario_id"], "difficulty_tier": meta.get("difficulty_tier"),
            "scenario_family": meta.get("scenario_family"), "seed": meta["random_seed"],
            "transition_count": len(ordered),
            "pass_record_count_before": sum(result_map[row["record_id"]]["status"] == "PASS" for row in ordered),
            "review_record_count_before": len(flagged_rows),
            "grouped_review_reasons": dict(grouped_reasons),
            "reason_timestep_ranges": {key: _ranges(sorted(value)) for key, value in reason_steps.items()},
            "failure_schedule": failures,
            "failure_activation_times_seconds": {item["failure_id"]: item.get("activation_time_seconds", 0.0) for item in failures},
            "actuator_effectiveness_ranges": {name: {
                "minimum": min(float(row["actuator_effectiveness"][name]) for row in ordered),
                "maximum": max(float(row["actuator_effectiveness"][name]) for row in ordered),
            } for name in ("elevator", "aileron", "rudder")},
            "sensor_fault_parameters": [item for item in failures if "sensor" in str(item.get("affected_subsystem", "")) or "bias" in str(item.get("failure_type", ""))],
            "recovery_outcome": meta.get("outcome_label"),
            "strict_envelope_recovery": None if expert is None else {
                "success": expert["strict_recovery_success"], "time_seconds": expert["strict_recovery_time_seconds"],
            },
            "key_state_ranges": _minmax(ordered),
            "automatic_validator_recommendation": "APPROVE" if all_contract_pass else "HUMAN_REVIEW_REQUIRED",
            "representative_flagged_timesteps": _representatives(ordered, result_map),
            "all_flagged_record_ids": [row["record_id"] for row in flagged_rows],
            "all_flagged_record_hashes": [row["record_hash"] for row in flagged_rows],
            "source_record_set_hash": object_hash([row["record_hash"] for row in ordered]),
            "independent_contract_check_hash": object_hash({row["record_id"]: contract_by_record[row["record_id"]] for row in flagged_rows}),
        }
        packet["packet_hash"] = object_hash(packet)
        packet_path = packets_dir / f"{object_hash(flight_id)[:16]}.json"
        write_json(packet_path, packet)
        packet_inventory.append({"flight_id": flight_id, "path": packet_path.as_posix(), "packet_hash": packet["packet_hash"],
                                 "recommendation": packet["automatic_validator_recommendation"]})
        for row in flagged_rows:
            checks = contract_by_record[row["record_id"]]
            rules = []
            old_reasons = result_map[row["record_id"]]["reasons"]
            if OLD_ACTUATOR_REASON in old_reasons:
                rules.append("ACT-SERIALIZED-HISTORY-001")
            if OLD_SENSOR_REASON in old_reasons:
                rules.append("SENSOR-DETERMINISTIC-BIAS-001")
            decision = "APPROVE" if all(check["status"] == "PASS" for check in checks) else "HUMAN_REVIEW_REQUIRED"
            dispositions.append({
                "record_id": row["record_id"], "flight_id": flight_id, "disposition": decision,
                "rule_ids": rules, "evidence": checks, "validator_version": VALIDATOR_VERSION,
                "timestamp_utc": timestamp, "source_hash": row["record_hash"],
                "explanation": "Independent deterministic contract checks pass." if decision == "APPROVE" else "At least one contract cannot be verified automatically.",
            })
    disposition_manifest = {
        "version": 1, "created_utc": timestamp, "validator_version": VALIDATOR_VERSION,
        "rules": {
            "ACT-SERIALIZED-HISTORY-001": "Approve only when visible effectiveness, delay metadata, and every executed control equal the independently reconstructed serialized-history contract.",
            "SENSOR-DETERMINISTIC-BIAS-001": "Approve only when fault metadata identifies the sensor and the observed value equals true value plus intensity times maximum_bias, with documented clamping/wrapping.",
        },
        "dispositions": dispositions,
        "counts": dict(Counter(item["disposition"] for item in dispositions)),
        "approved_dataset_exported": False,
    }
    disposition_manifest["manifest_hash"] = object_hash(disposition_manifest)
    write_json(output / "automated_dispositions.json", disposition_manifest)
    before = {
        "captured_utc": timestamp, "validator_report_hash": report["validator_report_hash"],
        "validation_report_file_sha256": file_hash(report_path),
        "record_status_counts": report["record_status_counts"],
        "flight_status_counts": {
            "PASS_ONLY": sum(all(result_map[row["record_id"]]["status"] == "PASS" for row in values) for values in by_flight.values()),
            "REVIEW_REQUIRED": sum(any(result_map[row["record_id"]]["status"] == "REVIEW_REQUIRED" for row in values) for values in by_flight.values()),
            "FAIL": sum(any(result_map[row["record_id"]]["status"] == "FAIL" for row in values) for values in by_flight.values()),
        },
        "raw_transitions_sha256": file_hash(dataset / "raw" / "transitions.jsonl"),
        "raw_flights_sha256": file_hash(dataset / "raw" / "flights.jsonl"),
    }
    write_json(output / "before_summary.json", before)
    audit = {
        "audit_version": 1, "created_utc": timestamp, "dataset_id": report["dataset_id"],
        "sealed_test_opened": False, "raw_candidate_modified": False,
        "before": before, "root_cause_classes": root_causes,
        "reason_counts": dict(reason_counts),
        "scenario_reason_counts": {f"{key[0]}::{key[1]}": value for key, value in scenario_reason_counts.items()},
        "active_failure_type_occurrences_across_flagged_records": dict(failure_counts),
        "failure_severity_occurrences_across_flagged_records": dict(severity_counts),
        "sensor_bias_magnitude_occurrences": dict(sensor_magnitudes),
        "packet_inventory": packet_inventory,
        "disposition_counts": disposition_manifest["counts"],
        "remaining_human_review_records": sum(item["disposition"] == "HUMAN_REVIEW_REQUIRED" for item in dispositions),
    }
    audit["audit_hash"] = object_hash(audit)
    write_json(output / "root_cause_audit.json", audit)


def capture_after(dataset: Path, output: Path) -> None:
    before = json.loads((output / "before_summary.json").read_text(encoding="utf-8"))
    report_path = dataset / "validation" / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report["record_results"]
    by_flight: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_to_flight = {row["record_id"]: row["flight_id"] for row in load_jsonl(dataset / "raw" / "transitions.jsonl")}
    for result in results:
        by_flight[record_to_flight[result["record_id"]]].append(result)
    after = {
        "validator_version": report["validator_version"],
        "validator_report_hash": report["validator_report_hash"],
        "validation_report_file_sha256": file_hash(report_path),
        "record_status_counts": report["record_status_counts"],
        "flight_status_counts": {
            "PASS_ONLY": sum(all(row["status"] == "PASS" for row in values) for values in by_flight.values()),
            "REVIEW_REQUIRED": sum(any(row["status"] == "REVIEW_REQUIRED" for row in values) for values in by_flight.values()),
            "FAIL": sum(any(row["status"] == "FAIL" for row in values) for values in by_flight.values()),
        },
        "integrity_errors": report["integrity_errors"], "leakage_errors": report["leakage_errors"],
        "raw_transitions_sha256": file_hash(dataset / "raw" / "transitions.jsonl"),
        "raw_flights_sha256": file_hash(dataset / "raw" / "flights.jsonl"),
    }
    comparison = {"before": before, "after": after, "raw_hashes_unchanged": (
        before["raw_transitions_sha256"] == after["raw_transitions_sha256"]
        and before["raw_flights_sha256"] == after["raw_flights_sha256"]
    )}
    comparison["comparison_hash"] = object_hash(comparison)
    write_json(output / "before_after_comparison.json", comparison)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/candidate_policy_data/final_policy_pilot_v2")
    parser.add_argument("--output", default="artifacts/research_stage/pilot_review_audit")
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    dataset = assert_not_sealed_path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "before":
        capture_before(dataset, output)
    else:
        capture_after(dataset, output)
    print(f"Captured {args.phase} audit at {output}")


if __name__ == "__main__":
    main()
