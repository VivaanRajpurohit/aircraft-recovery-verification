# Reproducibility

Use Python 3.11 and install the locked environment where practical. Scientific
runs must record the Git commit, branch, dirty-tree status, protocol version,
configuration and artifact hashes, platform, dependency versions, seeds,
solver settings, and timestamps.

Primary-result runs must start from a clean tree. An explicit override may be
used for exploratory work only; such outputs must record
`exploratory_unfrozen_run: true` and must be excluded from primary analysis by
default.

Existing generated datasets, checkpoints, raw trajectories, and results are
kept outside the source repository. Their relative paths and SHA-256 digests
are recorded in `artifacts/existing_artifacts.sha256`.

Baseline validation:

```powershell
python scripts/write_environment_manifest.py
python scripts/build_artifact_manifest.py
python scripts/audit_public_release.py
python -m compileall -q src scripts
python -m pytest
```
