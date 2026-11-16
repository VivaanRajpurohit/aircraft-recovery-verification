"""Export explicitly accepted records to a new immutable approved dataset."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.data_quality.export import export_approved_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--approved-dataset-id", required=True)
    parser.add_argument("--output-root", default="datasets/approved_policy_data")
    parser.add_argument("--failed-recovery-exclusion-justification")
    parser.add_argument("--disposition-manifest")
    args = parser.parse_args()
    result = export_approved_dataset(
        args.dataset, args.approved_dataset_id, args.output_root,
        args.failed_recovery_exclusion_justification,
        args.disposition_manifest,
    )
    print(json.dumps({
        "approved_dataset_id": result["approved_dataset_id"],
        "source_candidate_dataset_id": result["source_candidate_dataset_id"],
        "included_record_count": result["included_record_count"],
        "approved_dataset_hash": result["approved_dataset_hash"],
        "file_checksums": result["file_checksums"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
