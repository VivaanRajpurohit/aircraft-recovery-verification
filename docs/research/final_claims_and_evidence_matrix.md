# Preregistered Final Claims and Evidence Matrix

Status: **FROZEN FOR DEVELOPMENT BEFORE FINAL TRAINING**. Criteria may be revised only by a dated change record before final training and never in response to sealed-test results.

All claims are restricted to the declared low-order simulator and predefined in-domain stress conditions. The trajectory/flight—not the timestep—is the primary independent experimental unit.

| Claim | Metric and baseline | Acceptance criterion fixed before training | Required evidence | Failure to support | Limitation |
|---|---|---|---|---|---|
| A — Imitation | Per-output MAE/RMSE and macro normalized MAE versus training-label mean/constant and smoke checkpoint | Macro normalized MAE ≤0.12 and ≥15% below the best trivial baseline on development; every output must improve or be explicitly reported as unsupported | Multi-seed offline development/challenge results with flight-bootstrap 95% CIs | Threshold missed, or one output materially worse and hidden by aggregation | Agreement with a limited rule/PID expert is not optimal control |
| B — Closed-loop recovery | Recovery rate/time versus constant, neutral, expert, historical, and smoke baselines on identical scenarios/seeds | Monitored neural recovery rate must exceed best trivial baseline by ≥15 percentage points on challenge validation; no Tier 4/5 family may be omitted | Paired flight-level results and Wilson/bootstrap intervals | CI and effect do not support threshold, or systematic family failure | Simulator recovery only |
| C — Runtime assurance | Violations/duration/unsafe commands for neural-only versus monitored neural | Monitored controller must not increase total violation count and must reduce violation duration by ≥10% on paired challenge flights; recovery degradation must be ≤5 percentage points | Monitor ablation with identical seeds and paired differences | Violations increase, reduction <10%, or recovery loss >5 points | Monitor model/solver assumptions remain simulator-specific |
| D — Robustness | Recovery/violations across tiers, severities, initial-state bins, delays, sensor errors, disturbances, and held-out known combinations | Challenge/OOD-style recovery drop versus development ≤20 percentage points, with every predefined family reported; no claim if any family has <20 flights | Stratified results, fixed seed lists, failure-case library | Drop >20 points, missing strata, or uncontrolled seed selection | Not real-world generalization |
| E — Real-time feasibility | Inference, monitor, and full step p50/p95/p99/max on recorded hardware | Full-step p99 ≤50 ms and maximum <100 ms over ≥10,000 development steps | Raw timing samples, environment manifest, summary | Either timing threshold fails | Not hard-real-time certification |
| F — Reproducibility | Hash/provenance verification and repeated run semantic equality | All manifests/checksums validate; fixed-seed regeneration preserves record hashes and semantic metrics; checkpoints load reproducibly | Reproduction logs, hashes, environment/config manifests | Any unexplained mismatch or missing provenance | Binary files may contain metadata; semantic and byte reproducibility are separate |

The final conclusion must state unsupported claims plainly. No result may be described as foolproof, certified, aerodynamically validated, universally safe, or safe for an actual aircraft.

## Evidence status after pilot approval

| Evidence statement | Status | Current evidence or missing requirement |
|---|---|---|
| Pilot scenario set is discriminative overall | CURRENTLY SUPPORTED | Constant/neutral strict recovery 14/36 each versus expert 36/36. |
| Tier 5 is strongly discriminative | CURRENTLY SUPPORTED | Trivial 0/6, historical 2/6, smoke 1/6, expert 6/6. |
| Smoke checkpoint outperforms trivial baselines on this pilot | CURRENTLY SUPPORTED | Smoke 31/36 versus 14/36; descriptive pilot result without final inferential claim. |
| Independent validation distinguishes intentional failure behavior from invalid data | CURRENTLY SUPPORTED | Validator v3 verified 9,350/9,350 records; adversarial corruptions fail. |
| Approved pilot is traceable and reproducible | CURRENTLY SUPPORTED | One-to-one export audit, immutable hashes, 36 flights and 9,350 transitions. |
| Final neural imitation superiority | PLANNED TO TEST | Requires approved 1,152-flight dataset and multi-seed development/challenge results. |
| Final closed-loop recovery advantage | PLANNED TO TEST | Requires paired final-policy evaluation under the frozen protocol. |
| Runtime monitor benefit | PLANNED TO TEST | Requires neural-only versus monitored paired ablation. |
| Robustness across predefined simulator stress conditions | PLANNED TO TEST | Requires challenge/OOD stratified evaluation and confidence intervals. |
| Final statistical significance or final held-out performance | NOT SUPPORTED | Final model does not exist and sealed evaluation remains prohibited. |
| Universal robustness, real-aircraft safety, certification, or aerodynamic validity | NOT SUPPORTED | Outside the declared low-order simulator domain. |
