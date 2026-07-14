"""Run a reproducible paired expert/neural/monitored experiment batch."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from aircraft_recovery.config import load_config
from aircraft_recovery.experiments import ExperimentBatchConfig, run_experiment_batch

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase5_quick.yaml")
    parser.add_argument("--force", action="store_true", help="Replace the configured results directory.")
    args = parser.parse_args()
    config = load_config(args.config, ExperimentBatchConfig)
    output = Path(config.output_directory).resolve()
    results_root = Path("results").resolve()
    if args.force and output.exists():
        if results_root not in output.parents:
            raise ValueError("--force only permits deletion below the repository results directory")
        shutil.rmtree(output)
    print(json.dumps(run_experiment_batch(config), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
