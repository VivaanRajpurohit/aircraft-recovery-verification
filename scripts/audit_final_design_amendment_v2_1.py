"""Write the minimal v2.1 static amendment audit without generation."""

from __future__ import annotations

import json
from pathlib import Path

from aircraft_recovery.data_quality.common import write_json
from aircraft_recovery.research.design_amendment import audit_v2_1_amendment


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = audit_v2_1_amendment(root)
    write_json(root / "artifacts/research_stage/final_design_amendment_v2_1_audit.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
