# Final Pre-Generation Experimental Gate

Status: **PASS FOR DESIGN; GENERATION, TRAINING, AND SEALED EVALUATION REMAIN DISABLED.** This contract supersedes earlier 720-flight planning text. It does not change the simulator, controller, monitor, approved pilot, or prior evidence.

## Experimental unit and outcomes

One complete flight is the independent experimental unit. A transition is only a correlated training sample within a flight; no transition-level split or pseudo-replication is permitted. The confirmatory primary outcome is binary strict-recovery success per flight. Secondary outcomes are strict-recovery time, minimum airspeed, maximum absolute bank and pitch, minimum terrain clearance, safety-violation property-steps and duration, command saturation, control total variation, monitor modifications, fallback activations/duration, and timing latency. Exact machine rules are frozen in `configs/research/final_metrics_v1.yaml`.

Strict recovery requires 5.0 consecutive seconds with stall margin at least 10 kt, absolute bank at most 10°, pitch from −7.5° through +7.5°, absolute vertical speed at most 750 ft/min, and nonnegative terrain clearance. Boundaries are inside. The timer begins at the latest observed failure activation, or zero in nominal flights. An achieved recovery remains a success; subsequent loss is reported separately. Missing binary results count as failures unless a preregistered numerical/invalid-state termination makes the experiment invalid. Continuous recovery time is right-censored and omitted from arithmetic means while its censoring is reported.

Safety is distinct from recovery: stall and overspeed margins must be nonnegative, absolute bank at most 60°, pitch −20° through +25°, angle of attack at most 14°, load factor −1 through 3.5 g, and clearance nonnegative. Equality is safe. Simultaneous breaches count once per property-step; any-breach duration counts once per time step. Ground impact, clearance breach, loss of control, unrecoverable state, and unrecovered stall/overspeed are recovery failures. Numerical instability and invalid state invalidate the experiment and are separately enumerated. Controller/monitor timeouts are execution faults, not valid trajectory termination reasons.

## Prospective size decision

The old 720-flight design is rejected as under-precise: each evaluation split had only 12 flights per tier, yielding an approximately 24.6 percentage-point worst-case Wilson half-width. The accepted design has 1,152 flights: 576 train and 144 each development, challenge, OOD-style stress, and final sealed; every tier has 192 flights (96/24/24/24/24 by split). At 144 paired final flights, worst-case single-rate Wilson half-width is about 8.1 points (about 6.5 points near 80% success). Approximate 80% paired-binary minimum detectable differences are 12.8 points with 30% discordance and 16.5 points with 50% discordance. These are precision/MDE approximations, not a formal power guarantee. Six-flight family-by-evaluation cells remain exploratory and cannot support strong family-specific claims.

Expected size is about 299,000 transitions, with a 230,400–345,600 design range. Extrapolating the approved pilot serialization gives about 1.95 GB raw (roughly 1.51–2.26 GB across that range), before filesystem/compression overhead. The auditable calculation is `scripts/analyze_final_sample_size.py`.

## Tiers and Tier 1 review

The exact six-tier numeric contract is `configs/research/final_tiers_v1.yaml`; names alone never determine membership. Tier 0 is nominal. Tier 1 contains one mild impairment (20–35% reduced authority/loss) with narrower initial-state and disturbance ranges. Tier 2 contains one moderate isolated impairment. Tier 3 has two concurrent moderate factors. Tier 4 has two or three severe in-domain factors. Tier 5 exercises supported boundary values. Tier order describes intended challenge, not a claim that heterogeneous failures form one physical scalar.

Tier 1 was specifically reviewed because the pilot could make it look easier than nominal. That is plausible, not evidence of a simulator bug: mild failure declaration switches the deterministic expert into recovery/stabilization logic, while nominal trajectories can retain nonzero initial errors without an emergency declaration. The final plan therefore keeps Tier 0 and Tier 1 separate, balances initial-state and vertical-speed strata, reports recovery relative to failure activation, and does not pool them into an ordinal trend test. If Tier 1 again outperforms Tier 0, it will be reported as controller/scenario interaction rather than hidden.

## Vertical speed and sensor noise

Direct initial vertical-speed override is unsupported. Every scenario instead uses an exactly 2.0 s failure-free prelude controlled by a reset deterministic expert. Declared environment disturbances remain active and seeded; failures are prohibited until the prelude ends. Vertical speed is measured at exactly t=2.0 s and assigned to lower-inclusive/upper-exclusive bins (last upper bound inclusive): [−2500,−750), [−750,−150), [−150,150), [150,750), [750,2500]. Targets are equal within tier/split where integer counts allow; achieved counts and deviations must be audited before approval.

Gaussian sensor noise is excluded from generation v1. The present system has deterministic sensor-bias faults but lacks an audited noise provenance schema. Adding noise now would confound the frozen simulator contract. A later version must record pre-noise observation, post-noise observation, exact draw, seed, and noise-model version; it must receive a new dataset identifier.

## Expert privilege audit

The label generator and policy do not have identical information. The expert sees the structured controller input, full airport list, exact controller-known health/effectiveness, and retains PID integrals, previous errors, emergency-mode dwell/history, and selected diversion state. The 42-feature policy sees a fixed instantaneous summary and has no recurrent memory or full airport list. Neither is allowed to read the simulator ground-truth audit snapshot. Consequently, expert comparisons are a privileged label-source upper reference, not a fair optimal-agent contest. Label metadata must retain `expert_proposed_controls`; monitor-modified and simulator-executed controls remain separate.

## Monitor, fallback, and baseline fairness

`configs/research/final_monitor_v1.yaml` freezes the one-step bounded monitor, 20 ms timeout, projection blends 0.25/0.5/0.75/1.0, invalid-input rejection, disturbance bounds, and unchanged safety envelope. The monitor is evaluated on every monitored control step. It accepts a safe proposal, projects when a declared blend is safe, and invokes fallback when no admissible proposal is found or input/solver status requires rejection. It is not latched: the next step is independently reevaluated. Unknown/timeouts, modification reason, proposed action, selected action, and latency must be recorded.

`configs/research/final_fallback_v1.yaml` freezes the deterministic PID/rule fallback, 0.1 s step, 2.0 s mode dwell, targets, gains, runway/crosswind criteria, and output bounds. Fallback activation/deactivation is defined solely by each monitor decision; its internal emergency state remains stateful until controller reset at the flight boundary. No controller-specific safety envelope or scenario tuning is allowed.

All controllers receive identical scenario instances, initial state, failure parameters/timing, seed, and disturbance realization. Controller state is reset between flights. Constant policies are not tuned. The expert is explicitly privileged; historical and smoke checkpoints retain provenance limitations. Neural-only and monitored neural use the same checkpoint. The monitor/fallback are additions being evaluated, not hidden changes to the simulator.

## Pairing, splits, inference, and failure preservation

Every controller is paired by split, family, scenario-instance ID, and simulator seed. Seed ranges and five training initialization seeds are immutable in `configs/research/final_seed_registry_v1.json`. Splits are flight-level and disjoint. Challenge, OOD, and sealed severity/activation grids are disjoint; the sealed 144 flights are unusable for tuning, debugging, failure mining, threshold changes, or retraining.

The sole confirmatory comparison is monitored versus unmonitored final neural strict-recovery success at two-sided alpha 0.05. Per-controller rates receive Wilson intervals; paired differences and continuous outcomes receive 10,000-resample flight bootstrap intervals using seed 19001. Secondary/subgroup comparisons are exploratory and use Benjamini–Hochberg FDR 0.10, while raw effect sizes and intervals are also reported. No tier/family superiority claim is permitted from six-flight cells.

Discovered failures are appended to `artifacts/research_stage/failure_case_library_v1.json` with scenario/seed/controller/reason and artifact hashes. A case used for training becomes ineligible as unbiased evidence but is never deleted. Sealed failures cannot enter the library before the single authorized post-freeze evaluation.

## Balance and freeze decision

The matrix has exactly 192 flights per tier and exact split totals. Left/right complete-engine loss is balanced 152/152. Nominal, aileron, engine, partial-thrust, delay, deterministic-bias, disturbance, and rudder factor marginals are declared in the design YAML. Family cells are equal within each tier except Tier 1's intentional three-family layout. Severity grids, activation grids, vertical-speed bins, initial-state bins, engine side, and factor combinations must be reported as planned-versus-achieved before a generated dataset can be approved; unexpected imbalance is a gate failure, not a weighting adjustment made after outcomes are seen.

Final decision: **DESIGN_ACCEPTED_WITH_LIMITED_CLAIMS**. The prospective size supports overall paired simulator-domain estimates, not strong per-family claims, real-aircraft claims, formal neural verification, or certification. `artifacts/research_stage/final_generation_manifest.json` hashes the contracts and is explicitly marked generation/training/sealed-test unauthorized. Generation requires a separate explicit approval; this gate performs none.
