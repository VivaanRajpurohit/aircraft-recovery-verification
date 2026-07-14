"""Compare expert, neural baseline, and bounded monitored neural controllers."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.config import load_config
from aircraft_recovery.evaluation import MonitorEvaluationConfig, evaluate_monitor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/phase4_quick.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config, MonitorEvaluationConfig)
    if args.device:
        config.device = args.device
    print(json.dumps(evaluate_monitor(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

