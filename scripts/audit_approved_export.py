"""Audit an approved dataset against its immutable candidate source."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.data_quality.approved_audit import audit_approved_export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--approved", required=True)
    args = parser.parse_args()
    result = audit_approved_export(args.candidate, args.approved)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
