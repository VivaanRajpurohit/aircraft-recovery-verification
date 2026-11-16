"""Run the advanced v2.1 architecture-screen preflight and stop if blocked."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aircraft_recovery.data_quality.common import file_hash
from aircraft_recovery.training.v2_1_architecture_screen import (
    ArchitectureScreenBlocked,
    architecture_screen_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="results/final_architecture_screen_v2_1/preflight.json",
    )
    parser.add_argument("--skip-large-file-hash", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / args.output
    try:
        report = architecture_screen_preflight(
            root, verify_large_file=not args.skip_large_file_hash
        )
    except ArchitectureScreenBlocked as exc:
        report = exc.report
        exit_code = 2
    else:
        exit_code = 0
    report["source_file_sha256"] = file_hash(
        root / "src/aircraft_recovery/training/v2_1_architecture_screen.py"
    )
    report["configuration_file_sha256"] = file_hash(
        root / "configs/training/final_architecture_screen_v2_1.yaml"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

