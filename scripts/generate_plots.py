"""Generate Phase 5 publication plots."""
import argparse, json
from aircraft_recovery.analysis.plots import generate_plots
if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("batch",nargs="?",default="results/phase5_quick"); a=p.parse_args(); print(json.dumps(generate_plots(a.batch),indent=2))
