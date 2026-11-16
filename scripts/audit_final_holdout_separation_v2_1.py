"""Write original-v2 and amended-v2.1 cross-split exposure audits."""

from __future__ import annotations

import json
from pathlib import Path

from aircraft_recovery.data_quality.common import write_json
from aircraft_recovery.research.holdout_audit import build_exposure_matrix


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    original = build_exposure_matrix(root, use_v2_1=False)
    amended = build_exposure_matrix(root, use_v2_1=True)
    write_json(root / "artifacts/research_stage/final_holdout_exposure_v2_original.json", original)
    write_json(root / "artifacts/research_stage/final_holdout_exposure_v2_1.json", amended)
    print(json.dumps({"original": {"status": original["status"], "errors": original["errors"]}, "amended": {"status": amended["status"], "errors": amended["errors"], "audit_hash": amended["audit_hash"]}}, indent=2))
    if amended["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
