"""Freeze pre-generation design metadata; this script cannot generate flights."""

from __future__ import annotations

import json
from pathlib import Path

from aircraft_recovery.config import load_yaml
from aircraft_recovery.research.pre_generation import (
    validate_tier_contract,
    write_generation_manifest,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    errors = validate_tier_contract(load_yaml(root / "configs/research/final_tiers_v1.yaml"))
    if errors:
        raise SystemExit("Tier contract invalid: " + ", ".join(errors))
    output = write_generation_manifest(root)
    print(json.dumps(json.loads(output.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
