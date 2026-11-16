# Outcome and Metric Contract

Machine source: `configs/research/final_metrics_v1.yaml`. Experimental unit: complete flight. The primary endpoint is **strict recovery success** (higher is better): 1 iff, after the timer reference, all recovery inequalities hold for 5.0 consecutive seconds; otherwise 0. Report count/flight denominator and Wilson 95% CI. Invalid numerical experiments are excluded and enumerated; all other missing binary outcomes fail.

Let samples be indexed by fixed step `dt`, `I(condition)` be 1 when true, `u[j,t]` be executed control channel `j`, and `Tref` be latest observed failure activation (zero nominal).

| Metric | Exact calculation and unit | Preference / aggregation / missing or termination handling |
|---|---|---|
| Strict recovery time | First qualifying hold start minus `Tref`, seconds | Lower; median/mean and paired flight-bootstrap CI; right-censor at horizon, exclude censored from mean and report count. |
| Post-peak recovery time | Qualifying hold start minus the time of maximum pre-recovery normalized envelope excursion after `Tref`, seconds | Lower; secondary/exploratory; same censoring. |
| Minimum airspeed | `min_t airspeed_kts`, kt | Higher; flight value then distribution/paired difference. |
| Maximum bank/pitch | `max_t abs(bank_deg)` and `max_t abs(pitch_deg)`, degrees | Lower; flight value then distribution/paired difference. |
| Minimum clearance | `min_t terrain_clearance_ft`, ft | Higher; flight value; terminating ground/clearance failures retain their observed minimum. |
| Safety violation count | `sum_t sum_property I(property violated)`, property-steps | Lower; simultaneous properties count separately. |
| Safety violation duration | `dt * sum_t I(any property violated)`, seconds | Lower; simultaneous violations count once in duration. |
| Violation flight | `I(any property violated at any step)` | Lower; rate and Wilson interval. |
| Violation severity | Per property, `max(0, raw threshold exceedance)` in native units; maximum over flight | Lower; report separately in kt/deg/g/ft; no post-hoc composite score. |
| Command saturation | Mean over channels/steps of `I(abs(surface)==1 or throttle in {0,1})`, fraction | Lower contextually; also report per-channel; use executed action. |
| Control oscillation | `sum_t>0 sum_j abs(u[j,t]-u[j,t-1])`, control-units; also divide by elapsed seconds | Lower contextually; exclude discontinuity across reset. |
| Monitor modification | `mean_t I(executed monitor selection != neural proposal)`, fraction | Descriptive; monitored controller only, N/A otherwise. |
| Fallback activation | Count false-to-true fallback transitions, count/flight | Lower contextually; monitored controller only. |
| Fallback duration | `dt * sum_t I(fallback selected)`, seconds | Lower contextually; monitored controller only. |
| Inference/monitor/full-step latency | Recorded elapsed milliseconds per control step | Lower; p50/p95/p99/max; timing failure stays recorded. |

Recovery and safety are separate. Recovery equality is inside; safety equality is safe. A recovery hold interrupted before 5.0 s resets to zero; later re-entry starts a new hold. Initial success is not revoked, but later envelope loss is a separate boolean. Supported completion reasons are classified exactly by the machine contract. Metrics apply to every controller unless marked N/A; never replace missing N/A values with zero.
