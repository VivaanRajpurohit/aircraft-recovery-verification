import json
from pathlib import Path

from aircraft_recovery.data_quality.final_candidate_audit import (
    SEALED,
    _flight_statuses,
    _pairwise_overlaps,
    _realized_distributions,
)


def test_pairwise_overlap_reports_every_split_pair() -> None:
    values = {name: {name} for name in ("train", "development", "challenge", "ood_stress", SEALED)}
    overlaps = _pairwise_overlaps(values)
    assert len(overlaps) == 10
    assert set(overlaps.values()) == {0}


def test_sealed_distribution_is_design_only() -> None:
    row = json.loads(Path("datasets/candidate_policy_data/final_policy_candidate_v2_1/raw/flights.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    report = _realized_distributions([row])
    assert report["flight_count"] == 1
    assert "outcome_label" not in report
    assert "recovery_rate" not in report
    assert report["split_counts"] == {SEALED: 1}


def test_all_pass_records_imply_all_pass_flights() -> None:
    validation = {"record_count": 520, "record_status_counts": {"PASS": 520, "FAIL": 0, "REVIEW_REQUIRED": 0}}
    assert _flight_statuses(validation, [{}, {}]) == {"PASS": 2, "FAIL": 0, "REVIEW_REQUIRED": 0}
