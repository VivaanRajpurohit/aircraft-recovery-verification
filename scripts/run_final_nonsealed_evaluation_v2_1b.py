"""Run one authorized v2.1B nonsealed split in the frozen sequence."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.evaluation.v2_1b_nonsealed import run_nonsealed_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("challenge", "ood_stress"))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(run_nonsealed_evaluation(args.root, args.split), indent=2))


if __name__ == "__main__":
    main()
