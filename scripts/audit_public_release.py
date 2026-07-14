"""Audit public Git candidates without printing possible secret values."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 10 * 1024 * 1024
SELF = "scripts/audit_public_release.py"
PATTERNS = {
    "private_windows_path": re.compile(r"(?i)C:\\Users\\|OneDrive\\Documents"),
    "credential_assignment": re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|password|client[_-]?secret|auth[_-]?token)"
        r"\s*[:=]\s*[^\s$<{]+"
    ),
    "token_prefix": re.compile(
        r"(?i)(ghp_|github_pat_|glpat-|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{16,}|xox[baprs]-)"
    ),
    "private_key": re.compile(r"-----BEGIN ([A-Z ]+ )?PRIVATE KEY-----"),
    "email_address": re.compile(r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"),
}


def candidates() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(set(result.stdout.splitlines()), key=str.casefold)


def main() -> int:
    findings: list[dict[str, object]] = []
    files = candidates()
    for relative in files:
        path = ROOT / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            findings.append({"kind": "oversized_file", "path": relative, "size_bytes": size})
        if path.suffix.lower() in {".fbx", ".obj", ".gltf", ".glb"}:
            findings.append({"kind": "unreviewed_3d_asset", "path": relative})
        if relative == SELF:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(lines, 1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append({"kind": name, "path": relative, "line": number})

    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_file_count": len(files),
        "finding_count": len(findings),
        "findings": findings,
    }
    output = ROOT / "artifacts" / "public_release_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(files)} candidate files; findings={len(findings)}")
    for finding in findings:
        location = f"{finding['path']}:{finding.get('line', '')}".rstrip(":")
        print(f"{finding['kind']}: {location}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
