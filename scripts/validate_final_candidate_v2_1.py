"""Run independent validation with sealed outcome summaries suppressed."""

import json

from aircraft_recovery.data_quality.validation import validate_candidate_dataset


if __name__ == "__main__":
    report = validate_candidate_dataset(
        "datasets/candidate_policy_data/final_policy_candidate_v2_1",
        suppress_outcomes_for_splits={"final_sealed_test"},
        embed_record_results=False,
    )
    print(json.dumps({
        "validator_version": report["validator_version"],
        "record_count": report["record_count"],
        "flight_count": report["flight_count"],
        "record_status_counts": report["record_status_counts"],
        "integrity_errors": report["integrity_errors"],
        "leakage_errors": report["leakage_errors"],
        "validator_report_hash": report["validator_report_hash"],
        "suppressed_splits": report["quality_summary"]["outcome_summary_suppressed_splits"],
    }, indent=2))
