"""Train a bounded Phase 3 imitation policy."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.config import load_config
from aircraft_recovery.training import TrainingConfig, train_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/imitation_default.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override configured device.")
    parser.add_argument("--resume", help="Resume from a compatible checkpoint.")
    args = parser.parse_args()
    config = load_config(args.config, TrainingConfig)
    if args.device:
        config.device = args.device
    if args.resume:
        config.resume_checkpoint = args.resume
    print(json.dumps(train_policy(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

