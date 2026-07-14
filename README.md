# Formally Verified AI Recovery for Simulated Aircraft Under Multiple Concurrent Failures

> **Simulation-only research software.** This repository must not be connected
> to or used to control any real aircraft, drone, avionics system, actuator, or
> flight-control hardware. Simulator results do not establish airworthiness.

This repository contains Phases 1 through 5 of a research prototype comparing an
unmonitored numerical policy with a bounded runtime-monitored version. It
includes validated JSON interfaces, a replaceable low-order
simulator, structured experiments, configurable failures, deterministic
fallback recovery, emergency modes, synthetic diversion planning, and Phase 2
metrics, a reproducible neural imitation baseline, named Z3 obligations, paired
experiments, uncertainty estimates, plots, tables, and replay. Formal claims
apply only to explicitly bounded properties.

## Setup

Python 3.11 is the official supported runtime. From the repository root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run and test

```powershell
pytest
python scripts/run_scenario.py
python scripts/export_schemas.py
```

Every scenario creates `metadata.json` with exact Python/package versions,
`history.jsonl` with true/observed transitions and proposed/final actions, and
`summary.json` with metrics, termination, modes, and diversion decisions.

Typical Phase 1 output is:

```text
Run complete: results\phase1_nominal_20260713_<UTC timestamp>
Steps: 200; termination: duration_complete
Minimum stall margin: 50.00 kt
```

## Phase 2 demonstrations

```powershell
python scripts/run_scenario.py --config configs/scenarios/phase2_nominal.yaml
python scripts/run_scenario.py --config configs/scenarios/phase2_single_engine.yaml
python scripts/run_scenario.py --config configs/scenarios/phase2_engine_aileron.yaml
python scripts/run_scenario.py --config configs/scenarios/phase2_sensor_actuator.yaml
python scripts/run_scenario.py --config configs/scenarios/phase2_rudder_crosswind.yaml
python scripts/run_scenario.py --config configs/scenarios/phase2_unrecoverable.yaml
```

## Phase 3 quick pipeline

```powershell
python scripts/generate_demonstrations.py --config configs/training/demonstrations_quick.yaml --force
python scripts/validate_dataset.py --dataset datasets/phase3_quick
python scripts/train_policy.py --config configs/training/imitation_quick.yaml --device cpu
python scripts/inspect_checkpoint.py checkpoints/phase3_quick/best.pt
python scripts/evaluate_policy.py --config configs/evaluation/phase3_quick.yaml --device cpu
python scripts/compare_activations.py --config configs/evaluation/activation_comparison.yaml --device cpu
python scripts/verify.py --config configs/safety/default.yaml
python scripts/evaluate_monitor.py --config configs/evaluation/phase4_quick.yaml --device cpu
python scripts/run_experiment_batch.py --config configs/experiments/phase5_quick.yaml --force
python scripts/analyze_results.py results/phase5_quick
python scripts/generate_plots.py results/phase5_quick
python scripts/export_research_tables.py results/phase5_quick
```

See [architecture.md](docs/architecture.md), [model_assumptions.md](docs/model_assumptions.md),
[phase2.md](docs/phase2.md), [phase3.md](docs/phase3.md), and
[verification_scope.md](docs/verification_scope.md), and
[final_research_report.md](docs/final_research_report.md). CUDA and diversion
follow-up findings are separated in [final_validation_report.md](docs/final_validation_report.md).
The read-only standalone viewer is documented in [replay_3d.md](docs/replay_3d.md).
for assumptions and research boundaries.

The optional Cessna FBX is not distributed because its redistribution license
could not be established. Supply a legally licensed model at
`assets/aircraft/cessna/cessna.fbx`, or use the automatic primitive fallback.
See [asset_licensing.md](docs/asset_licensing.md).

## Status

- [x] Phase 1: repository, configuration, models/schemas, low-order simulator,
  runner, JSON logging, and tests.
- [x] Phase 2: failure injection, independent deterministic fallback, emergency
  modes, diversion scoring, concurrent failures, metrics, and tests.
- [x] Phase 3: demonstrations, configurable small PyTorch policy, checkpointing,
  and baseline evaluation.
- [x] Phase 4: explicit properties, Z3 bounded abstraction, runtime monitor,
  projection/rejection, and verification reports.
- [x] Phase 5: paired experiments, uncertainty estimates, plots, tables, and replay.

## Limitations

The simulator is deliberately simplified. Formal verification applies only to
named properties under bounded abstractions and documented
assumptions. It will not prove a neural network, simulator, or real aircraft
universally safe. Real certification requires substantially different models,
hardware, validation, redundancy, testing, and regulatory review.
