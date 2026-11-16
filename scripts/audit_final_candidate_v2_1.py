"""Build final design-integrity evidence for the generated v2.1 candidate."""

import json
from pathlib import Path

from aircraft_recovery.data_quality.final_candidate_audit import audit_final_candidate


if __name__ == "__main__":
    result = audit_final_candidate(Path(__file__).resolve().parents[1])
    print(json.dumps({
        "dataset_id": result["dataset_id"],
        "dataset_logical_hash": result["dataset_logical_hash"],
        "record_status_counts": result["record_status_counts"],
        "flight_status_counts": result["flight_status_counts"],
        "leakage_audit_hash": result["leakage_audit_hash"],
        "holdout_audit_hash": result["holdout_audit_hash"],
        "sealed_boundary_audit_hash": result["sealed_boundary_audit_hash"],
    }, indent=2))
