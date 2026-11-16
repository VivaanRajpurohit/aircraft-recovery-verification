"""Shared serialization and integrity primitives; no simulator dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


WORKFLOW_VERSION = "candidate-data-v2"
SCHEMA_VERSION = 2
CONTROL_NAMES = ("elevator", "aileron", "rudder", "throttle_left", "throttle_right")
SEALED_DATASET_MARKERS = ("datasets/policy_smoke_v1", "policy_smoke_v1")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(record: dict[str, Any]) -> str:
    return object_hash({key: value for key, value in record.items() if key != "record_hash"})


def assert_not_sealed_path(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    normalized = candidate.as_posix().lower()
    if any(marker in normalized for marker in SEALED_DATASET_MARKERS):
        raise PermissionError(f"Sealed held-out policy data may not be opened: {candidate}")
    return candidate


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
