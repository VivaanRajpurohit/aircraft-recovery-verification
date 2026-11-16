"""Generate an immutable raw candidate policy dataset."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.config import load_config
from aircraft_recovery.data_quality.candidate_generation import (
    CandidateGenerationConfig,
    generate_candidate_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config, CandidateGenerationConfig)
    print(json.dumps(generate_candidate_dataset(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

