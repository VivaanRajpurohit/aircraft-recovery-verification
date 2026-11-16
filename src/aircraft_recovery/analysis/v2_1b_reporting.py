"""Frozen publication figures and provenance for final v2.1B results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aircraft_recovery.config import load_yaml
from aircraft_recovery.data_quality.common import file_hash
from aircraft_recovery.analysis.v2_1b_final import CONDITIONS, SPLITS, SPLIT_LABELS


def _save(fig: Any, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_figures(root: str | Path, analysis_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    source = Path(analysis_path) if analysis_path else root / "results/final_v2_1b_analysis/analysis.json"
    analysis = json.loads(source.read_text(encoding="utf-8"))
    config_path = root / "configs/research/final_reporting_v2_1b.yaml"
    config = load_yaml(config_path)
    out = source.parent / "figures"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite final figures: {out}")
    out.mkdir(parents=True, exist_ok=True)
    colors = config["colors"]
    dpi = int(config["figure_dpi"])
    split_results = analysis["secondary_exploratory_results"]["split_results"]
    x = np.arange(len(SPLITS)); width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, condition in enumerate(CONDITIONS):
        rates = [split_results[s]["overall"][condition]["strict_recovery"]["rate"] for s in SPLITS]
        lows = [split_results[s]["overall"][condition]["strict_recovery"]["lower"] for s in SPLITS]
        highs = [split_results[s]["overall"][condition]["strict_recovery"]["upper"] for s in SPLITS]
        pos = x + (offset - 0.5) * width
        ax.bar(pos, rates, width, label=condition.capitalize(), color=colors[condition])
        ax.errorbar(pos, rates, yerr=[np.asarray(rates)-np.asarray(lows), np.asarray(highs)-np.asarray(rates)], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in SPLITS])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Strict recovery rate"); ax.legend(frameon=False)
    recovery_path = out / "recovery_by_split_and_condition.png"; _save(fig, recovery_path, dpi)

    tiers = [str(i) for i in range(6)]; x = np.arange(6)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, condition in enumerate(CONDITIONS):
        rates = [split_results["final_sealed_test"]["grouped"][condition]["difficulty_tier"][tier]["strict_recovery"]["rate"] for tier in tiers]
        ax.bar(x + (offset - 0.5) * width, rates, width, label=condition.capitalize(), color=colors[condition])
    ax.set_xticks(x, [f"Tier {tier}" for tier in tiers]); ax.set_ylim(0, 1.08); ax.set_ylabel("Strict recovery rate"); ax.legend(frameon=False)
    tier_path = out / "sealed_recovery_by_tier.png"; _save(fig, tier_path, dpi)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, condition in enumerate(CONDITIONS):
        rates = [split_results["final_sealed_test"]["grouped"][condition]["difficulty_tier"][tier]["safety_violation_flights"]["rate"] for tier in tiers]
        ax.bar(x + (offset - 0.5) * width, rates, width, label=condition.capitalize(), color=colors[condition])
    ax.set_xticks(x, [f"Tier {tier}" for tier in tiers]); ax.set_ylim(0, 1.08); ax.set_ylabel("Flights with simulator-domain safety violation"); ax.legend(frameon=False)
    safety_path = out / "sealed_safety_by_tier.png"; _save(fig, safety_path, dpi)

    counts = split_results["final_sealed_test"]["intervention_taxonomy"]["monitored"]["violated_property_counts"]
    if not counts:
        counts = {"no_recorded_violated_property_events": 0}
    labels = list(sorted(counts, key=lambda key: (-counts[key], key)))
    values = [counts[key] for key in labels]
    fig, ax = plt.subplots(figsize=(max(8.2, len(labels) * 0.7), 4.8))
    ax.bar(np.arange(len(labels)), values, color=colors["monitored"])
    ax.set_xticks(np.arange(len(labels)), [label.replace("_", " ") for label in labels], rotation=30, ha="right")
    ax.set_ylabel("Recorded monitor-rule events")
    taxonomy_path = out / "sealed_intervention_taxonomy.png"; _save(fig, taxonomy_path, dpi)

    captions = {
        recovery_path.name: "Strict recovery by split and controller condition for the frozen seed 17001 checkpoint. Error bars are frozen 95% Wilson intervals; each split contains 144 paired scenarios.",
        tier_path.name: "Final sealed-test strict recovery by frozen difficulty tier and controller condition. Each tier contains 24 paired scenarios; tier comparisons are exploratory.",
        safety_path.name: "Final sealed-test simulator-domain safety-violation incidence by frozen difficulty tier and controller condition. Each tier contains 24 paired scenarios; tier comparisons are exploratory.",
        taxonomy_path.name: "Recorded final sealed-test monitor-rule events by violated property for the monitored representative controller. Counts are action-level taxonomy inputs, not independent flights.",
    }
    artifacts = {}
    for path in (recovery_path, tier_path, safety_path, taxonomy_path):
        artifacts[path.name] = {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": file_hash(path), "caption": captions[path.name]}
    manifest = {
        "configuration_sha256": file_hash(config_path),
        "analysis_sha256": file_hash(source),
        "source_file_sha256": file_hash(root / "src/aircraft_recovery/analysis/v2_1b_reporting.py"),
        "artifacts": artifacts,
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
