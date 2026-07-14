"""Render a multi-panel replay from a simulator run directory."""
import argparse
from aircraft_recovery.visualization import render_replay
if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("run_directory"); p.add_argument("--output"); p.add_argument("--show",action="store_true"); a=p.parse_args(); print(render_replay(a.run_directory,a.output,a.show))
