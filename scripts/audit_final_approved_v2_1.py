"""Independently compare the final approved export with its candidate source."""

import json
from pathlib import Path

from aircraft_recovery.data_quality.final_approved import audit_final_approved_export


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = audit_final_approved_export(
        root / "datasets/candidate_policy_data/final_policy_candidate_v2_1",
        root / "datasets/approved_policy_data/final_policy_approved_v2_1",
    )
    print(json.dumps({
        "status": result["status"],
        "audit_hash": result["audit_hash"],
        "approved_flights": result["flight_exact_copy"]["approved_count"],
        "approved_transitions": result["transition_exact_copy"]["approved_count"],
        "sealed_records_in_export": result["sealed_records_in_approved_training_export"],
        "modified_or_omitted_records": (
            result["transition_exact_copy"]["byte_mismatch_count"]
            + result["transition_exact_copy"]["omitted_count"]
        ),
    }, indent=2))
