"""Export Phase 5 CSV, JSON, Markdown, and LaTeX tables."""
import argparse, json
from aircraft_recovery.analysis.tables import export_tables
if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("batch",nargs="?",default="results/phase5_quick"); a=p.parse_args(); print(json.dumps(export_tables(a.batch),indent=2))
