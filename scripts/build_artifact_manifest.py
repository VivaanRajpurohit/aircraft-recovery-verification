"""Hash preserved local evidence without copying it into the source repository."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOTS = ("results", "datasets", "checkpoints")
EXTRA_FILES = ("docs/final_research_report.md", "docs/final_validation_report.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    paths: set[Path] = set()
    for name in EVIDENCE_ROOTS:
        directory = ROOT / name
        if directory.exists():
            paths.update(path for path in directory.rglob("*") if path.is_file())
    for name in EXTRA_FILES:
        path = ROOT / name
        if path.is_file():
            paths.add(path)

    entries = []
    for path in sorted(paths, key=lambda item: item.as_posix().casefold()):
        relative = path.relative_to(ROOT).as_posix()
        entries.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})

    output = ROOT / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    (output / "existing_artifacts.sha256").write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Existing evidence is indexed in place and excluded from the source repository.",
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "entries": entries,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Hashed {manifest['file_count']} files ({manifest['total_size_bytes']} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
