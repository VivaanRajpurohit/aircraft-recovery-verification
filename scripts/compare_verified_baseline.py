"""Print the bounded monitored-versus-unmonitored comparison."""
import argparse, json
from pathlib import Path
if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("batch",nargs="?",default="results/phase5_quick"); a=p.parse_args(); d=json.loads((Path(a.batch)/"analysis.json").read_text()); print(json.dumps({"claim_boundary":"Bounded runtime monitoring in the configured simulator; not universal neural-network verification.","monitored":d["controllers"]["monitored"],"neural":d["controllers"]["neural"],"paired_monitored_minus_neural":d["paired_effects"]["monitored_minus_neural"]},indent=2))
