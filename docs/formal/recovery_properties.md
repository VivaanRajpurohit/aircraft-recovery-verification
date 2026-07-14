# Recovery properties

The target stabilization region is:

- airspeed 115–160 kt;
- absolute bank at most 10 degrees;
- pitch in `[-8, 8]` degrees;
- absolute vertical speed at most 800 ft/min;
- terrain clearance at least 500 ft;
- no enabled safety violation.

Recovery requires every reachable state to enter this region and remain within it for 5.0 seconds (50 consecutive 0.1-second transitions) in the full protocol. The pilot uses a 2.0-second dwell for method evaluation. The smoke uses three transitions solely to exercise implementation; it is not evidence for the scientific recovery claim.

Valid classifications are `verified_safe_and_recovering`, `verified_safe_recovery_unknown`, `counterexample_found`, `unknown_timeout`, `unknown_overapproximation`, and `outside_model_validity`. An unknown or untested cell is never promoted to safe.
