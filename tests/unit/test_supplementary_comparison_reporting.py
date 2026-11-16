"""Publication extraction must fail closed on partial or unpaired comparisons."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("supplement_report", ROOT / "scripts/report_supplementary_architecture_comparison_20261011.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def complete_rows():
    return [{"flight_id": str(i), "split": "challenge", "controller_condition": condition,
             "difficulty_tier": i // 24, "invalid_experiment": False, "step_count": 260}
            for i in range(144) for condition in module.CONDITIONS]


def test_reporting_requires_complete_paired_six_tier_challenge():
    rows = complete_rows()
    assert len(module.validate_records(rows)) == 144
    with pytest.raises(ValueError):
        module.validate_records(rows[:-1])
    bad = deepcopy(rows)
    bad[-1] = deepcopy(bad[-2])
    with pytest.raises(ValueError):
        module.validate_records(bad)


def test_reporting_rejects_other_splits_and_changed_scenario_geometry():
    for split in ("train", "development", "ood_stress", "final_sealed_test"):
        rows = complete_rows()
        rows[0]["split"] = split
        with pytest.raises(ValueError):
            module.validate_records(rows)
    rows = complete_rows()
    rows[0]["difficulty_tier"] = 5
    with pytest.raises(ValueError):
        module.validate_records(rows)
    with pytest.raises(ValueError):
        module.validate_records(complete_rows(), {str(i) for i in range(1, 145)})


def test_reporting_keeps_completed_artifacts_immutable(tmp_path):
    path = tmp_path / "completed.json"
    module.publish(path, {"status": "COMPLETE"})
    before = path.read_bytes()
    module.publish(path, {"status": "COMPLETE"})
    with pytest.raises(FileExistsError):
        module.publish(path, {"status": "PARTIAL"})
    assert path.read_bytes() == before


def test_original_and_supplementary_summary_shapes_match_flight_evidence():
    rows = complete_rows()
    for row in rows:
        row.update(strict_recovery_success=True, strict_recovery_time_seconds=0.05,
                   safety={"flight_has_violation": False, "violation_property_step_count": 0, "violation_duration_seconds": 0.0})
    aggregate = {"strict_recovery": {"successes": 144}, "safety_violation_flights": {"successes": 0},
                 "safety_violation_property_step_count": 0, "safety_violation_duration_seconds": 0.0,
                 "strict_recovery_time_seconds": {"count": 144, "mean": 0.05, "median": 0.05}}
    summary = {"status": "COMPLETE", "paired": {"pair_count": 144},
               "overall": {condition: deepcopy(aggregate) for condition in module.CONDITIONS}}
    module.validate_summary(summary, rows)
    summary["pair_count"] = 144
    module.validate_summary(summary, rows)
    summary["overall"]["monitored"]["strict_recovery_time_seconds"]["mean"] = 0.1
    with pytest.raises(ValueError):
        module.validate_summary(summary, rows)
