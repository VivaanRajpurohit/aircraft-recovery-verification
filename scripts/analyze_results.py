"""Compute Phase 5 Wilson and paired-bootstrap uncertainty estimates."""
import argparse, json
from aircraft_recovery.analysis.statistics import analyze_batch
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("batch", nargs="?", default="results/phase5_quick")
    args = parser.parse_args(); print(json.dumps(analyze_batch(args.batch), indent=2))
