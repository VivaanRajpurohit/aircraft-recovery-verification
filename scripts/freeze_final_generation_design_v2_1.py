"""Write the proposed v2.1 freeze without activating generation."""

from __future__ import annotations

import json
from pathlib import Path

from aircraft_recovery.data_quality.common import write_json
from aircraft_recovery.research.design_amendment import build_v2_1_freeze_manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = build_v2_1_freeze_manifest(root)
    write_json(root / "artifacts/research_stage/final_generation_manifest_v2_1.json", manifest)
    registry = {
        "registry_id": "final_generation_freeze_registry",
        "historical_v1": {"freeze_hash": manifest["historical_v1_freeze_hash"], "status": "superseded_before_generation_due_to_semantic_conflict"},
        "withdrawn_v2_proposal": {"freeze_hash": manifest["superseded_v2_proposal_hash"], "status": "not_activated_due_to_holdout_semantics_collision"},
        "proposed_v2_1": {"freeze_hash": manifest["freeze_hash"], "status": "pending_explicit_activation_review"},
        "active_generation_freeze": None,
        "generation_authorized": False,
    }
    write_json(root / "artifacts/research_stage/final_generation_freeze_registry.json", registry)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
