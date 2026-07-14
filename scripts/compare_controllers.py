"""Run the configured paired expert-versus-neural closed-loop comparison."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.config import load_config
from aircraft_recovery.evaluation import EvaluationConfig, evaluate_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/phase3_in_distribution.yaml")
    args = parser.parse_args()
    result = evaluate_policy(load_config(args.config, EvaluationConfig))
    print(json.dumps({
        "paired_closed_loop": result["paired_closed_loop"],
        "aggregate": result["closed_loop_aggregate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

