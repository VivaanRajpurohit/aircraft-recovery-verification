"""Validate the design-only final dataset matrix without generating data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aircraft_recovery.data_quality.common import write_json
from aircraft_recovery.research.dataset_design import audit_dataset_design


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/final_policy_dataset_design_v1.yaml")
    parser.add_argument("--output", default="artifacts/research_stage/final_dataset_design_audit.json")
    args = parser.parse_args()
    result = audit_dataset_design(args.config)
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
