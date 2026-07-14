"""Validate and print non-tensor checkpoint metadata."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.training.checkpointing import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    excluded = {"model_state_dict", "optimizer_state_dict"}
    print(json.dumps({key: value for key, value in checkpoint.items() if key not in excluded}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

