"""Approve and exactly export the validated nonsealed final v2.1 candidate."""

import json
from pathlib import Path

from aircraft_recovery.data_quality.final_approved import export_final_approved_dataset


if __name__ == "__main__":
    result = export_final_approved_dataset(Path(__file__).resolve().parents[1])
    print(json.dumps({
        "review_status": result["approval"]["review_status"],
        "approval_status": result["approval"]["approval_status"],
        "approved_dataset_id": result["manifest"]["approved_dataset_id"],
        "approved_dataset_logical_hash": result["manifest"]["approved_dataset_logical_hash"],
        "approved_flight_count": result["manifest"]["approved_flight_count"],
        "approved_transition_count": result["manifest"]["approved_transition_count"],
        "sealed_records_in_approved_export": result["manifest"]["sealed_records_in_approved_export"],
        "integrity_audit_status": result["audit"]["status"],
        "normalization_artifact_hash": result["normalization"]["normalization_artifact_hash"],
        "training_input_sanity_status": result["sanity"]["status"],
    }, indent=2))
