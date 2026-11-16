"""Run the committed v2.1B final sealed evaluation exactly once."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.evaluation.v2_1b_sealed import run_sealed_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run_sealed_evaluation(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
