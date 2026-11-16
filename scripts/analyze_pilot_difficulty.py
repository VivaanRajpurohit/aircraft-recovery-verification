"""Run preregistered non-sealed baseline checks on the candidate pilot scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aircraft_recovery.config import load_config
from aircraft_recovery.data_quality.candidate_generation import CandidateGenerationConfig
from aircraft_recovery.research.pilot_difficulty import run_pilot_difficulty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/final_policy_pilot_candidate_v2.yaml")
    parser.add_argument("--phase3-checkpoint", default="checkpoints/phase3_quick/best.pt")
    parser.add_argument("--smoke-checkpoint", default="checkpoints/policy_smoke_v1/best.pt")
    parser.add_argument("--output", default="artifacts/research_stage/pilot_difficulty_baselines.json")
    args = parser.parse_args()
    config = load_config(args.config, CandidateGenerationConfig)
    result = run_pilot_difficulty(config, args.phase3_checkpoint, args.smoke_checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
