"""Review immutable candidate records in a local terminal interface."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.data_quality.review import (
    _load_context,
    append_review_decision,
    render_record_context,
    visualize_record_window,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--record-id")
    parser.add_argument("--reviewer")
    parser.add_argument("--decision", choices=("ACCEPT", "REJECT", "FLAG", "SKIP"))
    parser.add_argument("--reason-code")
    parser.add_argument("--comment")
    parser.add_argument("--show-representatives", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--flight")
    parser.add_argument("--scenario")
    parser.add_argument("--failure")
    parser.add_argument("--status", choices=("PASS", "FAIL", "REVIEW_REQUIRED"))
    args = parser.parse_args()
    _, records, report = _load_context(args.dataset)
    result_map = {row["record_id"]: row for row in report["record_results"]}
    records = [
        row for row in records
        if (not args.flight or row["flight_id"] == args.flight)
        and (not args.scenario or row["scenario_id"] == args.scenario)
        and (not args.failure or any(
            args.failure in {failure.get("failure_id"), failure.get("failure_type")}
            for failure in row.get("active_failures", [])
        ))
        and (not args.status or result_map[row["record_id"]]["status"] == args.status)
    ]
    if not records:
        parser.error("No records match the selected filters")
    if args.show_representatives:
        shown = set()
        for index, record in enumerate(records):
            status = result_map[record["record_id"]]["status"]
            if status not in shown:
                print(render_record_context(records, report, index), "\n" + "=" * 80)
                shown.add(status)
            if shown == {"PASS", "FAIL", "REVIEW_REQUIRED"}:
                break
        return 0
    index = next((i for i, row in enumerate(records) if row["record_id"] == args.record_id), 0)
    print(render_record_context(records, report, index))
    if args.visualize:
        visualize_record_window(records, index)
    if args.decision:
        if not args.reviewer:
            parser.error("--reviewer is required with --decision")
        event = append_review_decision(
            args.dataset, records[index]["record_id"], args.decision, args.reviewer,
            args.reason_code, args.comment,
        )
        print(json.dumps(event, indent=2))
        return 0
    if args.record_id:
        return 0
    reviewer = input("Reviewer ID: ").strip()
    pending_comment = None
    while True:
        print("\n" + render_record_context(records, report, index))
        key = input("Decision/navigation: ").strip().upper()
        if key == "Q":
            break
        if key == "P":
            index = max(0, index - 1)
        elif key == "N":
            index = min(len(records) - 1, index + 1)
        elif key == "C":
            pending_comment = input("Reviewer comment for the next decision: ").strip() or None
        elif key in {"A", "R", "F", "S"}:
            decision = {"A": "ACCEPT", "R": "REJECT", "F": "FLAG", "S": "SKIP"}[key]
            reason = input("Reason code (required for overrides): ").strip() or None
            comment = input("Optional comment: ").strip() or pending_comment
            append_review_decision(args.dataset, records[index]["record_id"], decision, reviewer, reason, comment)
            pending_comment = None
            index = min(len(records) - 1, index + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
