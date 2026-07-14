"""Run a controlled ReLU-versus-SiLU policy experiment."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.config import load_config
from aircraft_recovery.evaluation import ActivationComparisonConfig, run_activation_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/activation_comparison.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override both training and evaluation devices.")
    args = parser.parse_args()
    config = load_config(args.config, ActivationComparisonConfig)
    print(json.dumps(run_activation_comparison(config, args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

