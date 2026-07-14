"""Audit diversion behavior without modifying source experiment artifacts."""
import argparse,json
from aircraft_recovery.analysis.diversion_validation import analyze_batch_diversions
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("batch",nargs="?",default="results/phase5_quick"); p.add_argument("--output",default="results/final_validation"); a=p.parse_args(); print(json.dumps(analyze_batch_diversions(a.batch,a.output)["aggregate"],indent=2))
