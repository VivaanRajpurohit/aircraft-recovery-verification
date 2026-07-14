"""Validate Phase 3 dataset bounds, masks, leakage, and episode splits."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.data.validation import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset directory containing transitions.npz.")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

