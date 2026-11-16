"""Independently validate a serialized candidate policy dataset."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.data_quality.validation import validate_candidate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    report = validate_candidate_dataset(args.dataset)
    print(json.dumps({
        "dataset_id": report["dataset_id"],
        "record_count": report["record_count"],
        "flight_count": report["flight_count"],
        "record_status_counts": report["record_status_counts"],
        "integrity_errors": report["integrity_errors"],
        "leakage_errors": report["leakage_errors"],
        "validator_report_hash": report["validator_report_hash"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

