# Final Baseline Protocol

Status: frozen before large-dataset generation.

Every baseline will run identical scenario instances and seeds. Results are reported per controller, training seed, split, tier, family, severity, and flight. The flight—not timestep—is the independent recovery unit.

| Baseline | Frozen identity | Role and limitation |
|---|---|---|
| Constant idle | `(elevator, aileron, rudder, left throttle, right throttle)=(0,0,0,0,0)` | Non-reactive lower reference; throttle saturation is expected. |
| Neutral cruise | `(0,0,0,0.5,0.5)` | Non-reactive neutral-command reference. |
| Deterministic expert | `DeterministicFallbackController` plus hashed fallback configuration | Label source and privileged upper reference, not assumed optimal. |
| Historical Phase 3 | `checkpoints/phase3_quick/best.pt`, SHA-256 `efbcc999dd43e7f9e75dc9738e67bb19f2b7fe994ecee626de10b296b8b62c25` | Historical only; provenance-incomplete and unsuitable as primary scientific evidence. |
| Smoke checkpoint | `checkpoints/policy_smoke_v1/best.pt`, SHA-256 `daecaeb37eedd943804da7cf13ed90e4dccca4a80268f8003596f63d6dfbf4e4` | Partially reproducible pipeline-validation baseline, not the final model. |
| Final neural, unmonitored | Every one of five frozen training seeds | Measures policy behavior without runtime assurance. |
| Final neural, monitored | Same checkpoints/scenarios/seeds with frozen monitor and fallback | Paired runtime-assurance comparison. |

No baseline is selected or removed after seeing sealed-test results. Neural results report all five training seeds; the best seed is not reported alone. Constant baselines are not tuned per scenario. Historical provenance limitations remain visible in every comparison table.
