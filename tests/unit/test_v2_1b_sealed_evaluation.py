from __future__ import annotations

import json
from pathlib import Path

import pytest

from aircraft_recovery.data_quality.common import object_hash
from aircraft_recovery.evaluation.v2_1b_sealed import (
    CANDIDATE_LOGICAL_HASH,
    EXPECTED_FLIGHTS,
    SEALED_SPLIT,
    sealed_preflight,
)


ROOT = Path(__file__).resolve().parents[2]


def test_sealed_constants_bind_frozen_candidate_geometry() -> None:
    assert SEALED_SPLIT == "final_sealed_test"
    assert EXPECTED_FLIGHTS == 144
    assert CANDIDATE_LOGICAL_HASH == "4db2d75ab341e4573019d09176507b03851d7d26dd2eb360d42ecfc2adc28706"


def test_presealed_freeze_is_self_authenticating_when_present() -> None:
    path = ROOT / "artifacts/research_stage/final_presealed_freeze_v2_1b.json"
    if not path.exists():
        pytest.skip("pre-sealed freeze is created only after the implementation gate passes")
    document = json.loads(path.read_text(encoding="utf-8"))
    observed = object_hash({k: v for k, v in document.items() if k != "freeze_logical_sha256"})
    assert observed == document["freeze_logical_sha256"]


def test_sealed_preflight_fails_closed_without_committed_freeze(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="committed pre-sealed freeze"):
        sealed_preflight(tmp_path)
