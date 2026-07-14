"""Measure exact-checkpoint IBP feasibility without claiming plant verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from aircraft_recovery.data.preprocessing import PreprocessingStatistics
from aircraft_recovery.training.checkpointing import load_checkpoint
from aircraft_recovery.verification.neural_bounds import Interval, bound_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--standard-deviation-radius", type=float, default=0.01)
    parser.add_argument("--vary-binary-masks", action="store_true")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    statistics = PreprocessingStatistics.from_dict(checkpoint["normalization_statistics"])
    means = np.asarray(statistics.means, dtype=np.float64)
    deviations = np.asarray(statistics.standard_deviations, dtype=np.float64)
    normalized = np.asarray(statistics.normalized, dtype=bool)
    lower = means.copy()
    upper = means.copy()
    lower[normalized] -= args.standard_deviation_radius * deviations[normalized]
    upper[normalized] += args.standard_deviation_radius * deviations[normalized]
    lower[~normalized] = 0.0
    upper[~normalized] = 1.0 if args.vary_binary_masks else 0.0
    started = perf_counter()
    result = bound_checkpoint(args.checkpoint, Interval(lower, upper))
    elapsed = perf_counter() - started
    print(json.dumps({
        "status": "neural_bound_feasibility_only",
        "method": result.method,
        "checkpoint_sha256": result.checkpoint_sha256,
        "input_radius_standard_deviations": args.standard_deviation_radius,
        "binary_masks": "varied_0_to_1" if args.vary_binary_masks else "fixed_zero",
        "control_lower": result.controls.lower.tolist(),
        "control_upper": result.controls.upper.tolist(),
        "maximum_control_width": float(np.max(result.controls.upper - result.controls.lower)),
        "runtime_seconds": elapsed,
        "closed_loop_classification": "not_run_gate_c_pending",
    }, indent=2))


if __name__ == "__main__":
    main()
