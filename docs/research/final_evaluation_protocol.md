# Final Evaluation Protocol

## Controllers

Paired scenarios/seeds compare constant action, neutral control, deterministic expert, compatible Phase 3 historical checkpoint, smoke checkpoint, selected neural-only, and selected monitored neural. Monitor/fallback ablations run as isolated experiment configurations and never weaken the default live system.

## Flight-level metrics

Recovery success/time, post-peak strict recovery time, minimum airspeed, maximum absolute bank/pitch, minimum clearance, violation count/duration, command saturation, control oscillation, monitor modifications, fallback activations/duration, solver unknown/timeouts, neural latency, monitor latency, and full-step latency.

Report by scenario family, tier, severity, evaluation seed, controller, and training seed. Engine-only, aileron/control-only, concurrent, delay, sensor, disturbance, and boundary results remain separate.

## Statistics

- Flight is the primary experimental unit.
- Report denominator, mean/median/standard deviation where meaningful.
- Success rates use 95% Wilson intervals.
- Continuous metrics use fixed-seed flight-bootstrap 95% intervals (10,000 resamples, bootstrap seed 19001).
- Paired controller comparisons use per-flight differences.
- Timing reports p50/p95/p99/max over at least 10,000 control steps.

## Development stress search

A bounded predefined search may vary only declared simulator parameters. Discovered failures are preserved in the failure-case library. Retraining on a case creates a new generation and makes that case ineligible as unbiased evidence.

## Sealed test

The final-test runner must reject execution unless the freeze manifest is complete and an explicit authorization token is supplied. This document does not authorize that execution.

