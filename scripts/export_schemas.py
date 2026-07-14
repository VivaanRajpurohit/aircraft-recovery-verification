"""Regenerate JSON Schemas from the Pydantic boundary models."""

from __future__ import annotations

import json
from pathlib import Path

from aircraft_recovery.models import ControllerInput, ControllerOutput


def main() -> int:
    schemas = {
        "controller_input.schema.json": ControllerInput.model_json_schema(),
        "controller_output.schema.json": ControllerOutput.model_json_schema(),
    }
    output = Path("schemas")
    output.mkdir(exist_ok=True)
    for filename, schema in schemas.items():
        (output / filename).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

