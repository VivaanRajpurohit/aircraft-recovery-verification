# Change record 002: formal plant plan

## Objective

Define a traceable plant derivation and disjoint calibration/validation protocol without running new experiments.

## Implementation and rationale

Documented an interval/LPV transition directly from the numerical simulator's configured coefficients, including degraded control authority, asymmetric thrust, delay, nonlinear vertical-speed conversion, and residual inflation.

## Tests and verification results

No plant implementation or scientific result is claimed in this readiness stage. Gate C requires new versioned calibration and held-out validation data.

## Known limitations

Crosswind-to-bank coupling is absent in the current point-mass simulator. Provisional residual values cannot establish safety. Formal validity is restricted to configured bounds.

## Effect on earlier artifacts

None. No simulator trajectory was generated and no prior artifact was changed.

## Proposed reproduction command

To be added with the calibration generator after approval.
