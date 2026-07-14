"""Evaluate a checkpoint offline and in paired own-state closed-loop runs."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.config import load_config
from aircraft_recovery.evaluation import EvaluationConfig, evaluate_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/phase3_in_distribution.yaml")
    parser.add_argument("--checkpoint", help="Override checkpoint path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override configured device.")
    args = parser.parse_args()
    config = load_config(args.config, EvaluationConfig)
    if args.checkpoint:
        config.checkpoint = args.checkpoint
    if args.device:
        config.device = args.device
    result = evaluate_policy(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

