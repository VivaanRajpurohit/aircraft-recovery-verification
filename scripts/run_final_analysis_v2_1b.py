"""Build frozen v2.1B analysis, tables, and figures."""

from __future__ import annotations

import argparse
import json

from aircraft_recovery.analysis.v2_1b_final import build_analysis, export_tables
from aircraft_recovery.analysis.v2_1b_reporting import generate_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    analysis = build_analysis(args.root)
    tables = export_tables(args.root)
    figures = generate_figures(args.root)
    print(json.dumps({"analysis_status": analysis["status"], "tables": [str(p) for p in tables], "figures": figures}, indent=2))


if __name__ == "__main__":
    main()
