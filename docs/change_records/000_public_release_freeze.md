# Public-release pilot freeze

## Objective

Convert the validated pilot directory into a traceable, privacy-audited research
software baseline without changing existing generated evidence.

## Implementation

- added public repository policy, citation, contribution, security, licensing,
  reproducibility, provenance, and artifact guidance;
- excluded raw results, datasets, checkpoints, private-path reports, virtual
  environments, and the unlicensed Cessna model;
- added path-sanitized environment, checksum, and public-release audit tools;
- added a permanent replay-only notice and separation regression test;
- added compile validation to CI;
- increased only the offline formal-example timeout floor to 250 ms so a known
  unsafe witness does not become nondeterministically UNKNOWN under host load.

The runtime monitor retains its configured 20 ms timeout.

## Scientific rationale

Generated evidence must be attributable to immutable source and configuration
versions. Unknown solver outcomes must never be mislabeled. The offline witness
test therefore receives enough time to remain deterministic while runtime
timeout behavior is unchanged.

## Assumptions and limitations

The local Cessna FBX has no established redistribution license. Existing raw
artifacts remain local and are represented in Git by SHA-256 manifests. The
current repository license is deliberately conservative pending an explicit
copyright-holder license decision.

## Effect on earlier artifacts

None. No prior dataset, checkpoint, result, report, plot, table, screenshot, or
replay log was modified or regenerated.

## Validation

```powershell
python scripts/write_environment_manifest.py
python scripts/build_artifact_manifest.py
python scripts/audit_public_release.py
python -m compileall -q src scripts
python -m pytest
```
