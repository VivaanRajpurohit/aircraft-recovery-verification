"""Execute the authorized advanced v2.1A architecture screen."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.training.v2_1a_architecture_screen import run_architecture_screen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-large-file-hash", action="store_true")
    args = parser.parse_args()
    result = run_architecture_screen(
        args.root, verify_large_file=not args.skip_large_file_hash
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
