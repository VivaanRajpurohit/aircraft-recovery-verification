"""Write a path-sanitized dependency lock and environment manifest."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    value = result.stdout.strip()
    return value or None


def main() -> int:
    packages = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.casefold,
    )
    (ROOT / "requirements-lock.txt").write_text(
        "\n".join(packages) + "\n", encoding="utf-8"
    )

    try:
        import numpy
        import torch
        import z3

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        versions = {
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "z3": z3.get_version_string(),
        }
        cuda_available = torch.cuda.is_available()
    except ImportError:
        gpu = None
        versions = {}
        cuda_available = False

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "gpu": gpu,
        "versions": versions,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
    }
    destination = ROOT / "artifacts" / "environment_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(destination.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
