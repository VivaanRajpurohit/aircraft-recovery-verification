"""Structured, append-only experiment logging."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
import platform
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ExperimentLogger:
    """Write metadata, time-series JSON Lines, and summary JSON."""

    def __init__(self, run_directory: str | Path) -> None:
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self._history_path = self.run_directory / "history.jsonl"

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        payload = {
            **metadata,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": self._package_versions(),
            },
        }
        self._write_json("metadata.json", payload)

    def append_step(
        self,
        observation: BaseModel,
        proposed_action: BaseModel,
        final_action: BaseModel,
        next_observation: BaseModel,
        ground_truth_before: dict[str, Any] | None = None,
        ground_truth_after: dict[str, Any] | None = None,
        controller_latency_ms: float | None = None,
        monitor_decision: dict[str, Any] | None = None,
    ) -> None:
        """Append a complete transition, retaining safety-filter audit fields."""
        record = {
            "observation": observation.model_dump(mode="json"),
            "proposed_action": proposed_action.model_dump(mode="json"),
            "final_action": final_action.model_dump(mode="json"),
            "next_observation": next_observation.model_dump(mode="json"),
        }
        if ground_truth_before is not None:
            record["ground_truth_before"] = ground_truth_before
        if ground_truth_after is not None:
            record["ground_truth_after"] = ground_truth_after
        if controller_latency_ms is not None:
            record["controller_latency_ms"] = controller_latency_ms
        if monitor_decision is not None:
            record["monitor_decision"] = monitor_decision
        with self._history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json("summary.json", summary)

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.run_directory / filename
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")

    @staticmethod
    def _package_versions() -> dict[str, str]:
        versions: dict[str, str] = {}
        for package in ("aircraft-recovery-research", "numpy", "pydantic", "PyYAML"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not-installed"
        return versions
