"""Pre-specified descriptive and paired uncertainty estimates for Phase 5."""
from __future__ import annotations
import json
from math import sqrt
from pathlib import Path
from typing import Any
import numpy as np

BINARY_METRICS = ("recovery_success", "stabilization_success", "diversion_success")
CONTINUOUS_METRICS = (
    "time_to_stabilization_seconds", "minimum_stall_margin_kts",
    "maximum_absolute_roll_deg", "minimum_terrain_clearance_ft",
    "safety_violation_duration_seconds", "controller_latency_ms_mean",
    "command_smoothness", "command_saturation_frequency",
)

def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    half = z*sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return center-half, center+half

def paired_bootstrap_difference(a: list[float], b: list[float], seed: int, resamples: int) -> dict[str, float | int]:
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    valid = np.isfinite(aa) & np.isfinite(bb)
    differences = aa[valid] - bb[valid]
    if not len(differences):
        return {"n_pairs": 0, "mean_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "cohen_dz": float("nan")}
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(0, len(differences), size=(resamples, len(differences)))].mean(axis=1)
    sd = differences.std(ddof=1) if len(differences) > 1 else 0.0
    dz = differences.mean()/sd if sd > 0 else (0.0 if differences.mean() == 0 else float("inf"))
    return {"n_pairs": len(differences), "mean_difference": float(differences.mean()), "ci_low": float(np.quantile(sampled, .025)), "ci_high": float(np.quantile(sampled, .975)), "cohen_dz": float(dz)}

def _finite(records: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(r[metric]) for r in records if r.get(metric) is not None and np.isfinite(float(r[metric]))]

def analyze_batch(batch_directory: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    batch = Path(batch_directory)
    records = json.loads((batch / "batch_records.json").read_text(encoding="utf-8"))
    manifest = json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8"))
    seed = manifest["configuration"]["bootstrap_seed"]
    resamples = manifest["configuration"]["bootstrap_resamples"]
    controllers: dict[str, Any] = {}
    for name in ("expert", "neural", "monitored"):
        subset = [r for r in records if r["controller"] == name]
        summary: dict[str, Any] = {"n": len(subset)}
        for metric in BINARY_METRICS:
            k = sum(bool(r[metric]) for r in subset)
            lo, hi = wilson_interval(k, len(subset))
            summary[metric] = {"successes": k, "rate": k/len(subset) if subset else None, "wilson_95_ci": [lo, hi]}
        for metric in CONTINUOUS_METRICS:
            values = _finite(subset, metric)
            summary[metric] = {"n": len(values), "mean": float(np.mean(values)) if values else None, "median": float(np.median(values)) if values else None, "std": float(np.std(values, ddof=1)) if len(values)>1 else 0.0 if values else None}
        controllers[name] = summary
    paired: dict[str, Any] = {}
    by_pair = {(r["pair_id"], r["controller"]): r for r in records}
    pair_ids = sorted({r["pair_id"] for r in records})
    for comparison_index, (left, right) in enumerate((("monitored", "neural"), ("monitored", "expert"))):
        paired[f"{left}_minus_{right}"] = {}
        for metric_index, metric in enumerate((*BINARY_METRICS, *CONTINUOUS_METRICS)):
            a, b = [], []
            for pair_id in pair_ids:
                av, bv = by_pair[(pair_id, left)].get(metric), by_pair[(pair_id, right)].get(metric)
                if av is not None and bv is not None:
                    a.append(float(av)); b.append(float(bv))
            paired[f"{left}_minus_{right}"][metric] = paired_bootstrap_difference(a, b, seed + comparison_index*100 + metric_index, resamples)
    strata: dict[str, Any] = {}
    for field in ("distribution", "expected_recoverability"):
        strata[field] = {}
        for level in sorted({r[field] for r in records}):
            strata[field][level] = {name: {metric: float(np.mean([float(r[metric]) for r in records if r[field] == level and r["controller"] == name])) for metric in BINARY_METRICS} for name in ("expert", "neural", "monitored")}
    result = {"analysis_type": "descriptive_with_pre_specified_uncertainty", "statistical_tests": "none", "multiple_comparison_correction": "not_applicable", "controllers": controllers, "paired_effects": paired, "strata": strata, "limitations": ["Bootstrap intervals summarize this simulator sample only.", "The quick batch is underpowered for confirmatory inference.", "No p-values or certification claims are made."]}
    destination = Path(output) if output else batch / "analysis.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return result
