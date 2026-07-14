"""Run bounded Phase 4 Z3 obligations and structured examples."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.config import load_config
from aircraft_recovery.safety import SafetyConfig
from aircraft_recovery.verification import run_formal_verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/safety/default.yaml")
    parser.add_argument("--output", default="results/phase4_verification")
    args = parser.parse_args()
    config = load_config(args.config, SafetyConfig)
    report = run_formal_verification(config, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
