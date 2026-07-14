# Change record 001: formal state specification

## Objective

Define the bounded recovery-envelope protocol before implementing or running the analyzer.

## Implementation and rationale

Added human-readable formal states, actions, failures, disturbances, uncertainty, safety, recovery, validity, and three machine-readable run levels. Values are derived from the simulator configuration and existing monitor thresholds; narrower initial regions are staged for tractable analysis.

## Assumptions and limitations

Plant residuals are not calibrated yet. Smoke settings are implementation diagnostics. Angle of attack and load factor are disabled until represented soundly.

## Tests and verification

YAML parsing and protocol validation will be added with the analyzer. Existing tests are rerun after this commit.

## Effect on earlier artifacts

None. Simulator, controller, checkpoints, datasets, prior results, and replay behavior are unchanged.

## Reproduction

`python -m pytest -q`
