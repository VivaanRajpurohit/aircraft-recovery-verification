# Bounded Runtime Assurance for Neural Aircraft Recovery

<p align="center">
  <strong>A simulation research prototype for studying neural control, concurrent failures, and formally specified recovery envelopes.</strong>
</p>

<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-neural%20policy-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Z3" src="https://img.shields.io/badge/Z3-bounded%20verification-5C2D91">
  <img alt="Tests" src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-2E7D32">
</p>

> [!CAUTION]
> **Simulation-only research software.** Do not connect this project to a real aircraft, drone, avionics system, actuator, or flight-control device. Simulator results do not establish airworthiness.

## Research question

How can a learned recovery policy be evaluated when several simulated aircraft failures occur at once—and how can a runtime monitor constrain that policy to an explicitly bounded safety envelope?

This repository explores that question through a reproducible, five-phase pipeline. It compares an unmonitored numerical policy with a monitored controller that can accept, project, or reject proposed actions before falling back to deterministic recovery logic.

## What is implemented

| Research layer | Implementation |
|---|---|
| Simulation | Replaceable low-order twin-engine aircraft model with structured state and action interfaces |
| Failure modeling | Engine, actuator, sensor, navigation, and disturbance scenarios, including concurrent failures |
| Learning | PyTorch imitation-policy training, checkpointing, dataset validation, and in-/out-of-distribution evaluation |
| Runtime assurance | Configurable bounds, action projection/rejection, deterministic fallback control, and emergency modes |
| Formal methods | Named Z3 obligations over documented bounded abstractions and exact ReLU output bounds |
| Evaluation | Paired experiment batches, uncertainty estimates, plots, tables, provenance metadata, and replay tooling |

## System flow

```text
scenario + failures
        |
        v
low-order simulator --> observation --> neural policy --> proposed action
        ^                                      |
        |                                      v
        +---- final action <-- runtime monitor + bounded properties
                               | accept
                               | project
                               ` reject --> deterministic fallback / emergency mode
```

The safety claims are deliberately narrow: formal results apply only to the named properties, bounded state/action regions, plant abstraction, and assumptions documented in this repository.

## Repository map

```text
src/aircraft_recovery/
|-- controllers/      neural, monitored, PID, fallback, and emergency control
|-- failures/         configurable failure injection
|-- simulator/        low-order simulation interface and implementation
|-- safety/           runtime monitor and safety configuration
|-- verification/     Z3 models, formal obligations, and neural bounds
|-- training/         imitation learning and checkpointing
|-- evaluation/       policy and monitor evaluation
`-- analysis/         statistics, tables, and plots

configs/              versioned scenarios, controllers, training, and experiments
scripts/              reproducible command-line entry points
schemas/              validated controller input/output contracts
tests/                unit, integration, and safety-property tests
docs/                 architecture, assumptions, formal scope, and reproducibility
artifacts/             public-release and environment manifests
```

## Reproduce the prototype

Python 3.11 is the supported runtime.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Run a single scenario and inspect its structured outputs:

```powershell
python scripts/run_scenario.py --config configs/scenarios/phase2_engine_aileron.yaml
```

Each run records environment metadata, true and observed transitions, proposed and final actions, summary metrics, termination reasons, modes, and diversion decisions.

Run the compact end-to-end research pipeline:

```powershell
python scripts/generate_demonstrations.py --config configs/training/demonstrations_quick.yaml --force
python scripts/validate_dataset.py --dataset datasets/phase3_quick
python scripts/train_policy.py --config configs/training/imitation_quick.yaml --device cpu
python scripts/evaluate_policy.py --config configs/evaluation/phase3_quick.yaml --device cpu
python scripts/verify.py --config configs/safety/default.yaml
python scripts/evaluate_monitor.py --config configs/evaluation/phase4_quick.yaml --device cpu
python scripts/run_experiment_batch.py --config configs/experiments/phase5_quick.yaml --force
python scripts/analyze_results.py results/phase5_quick
python scripts/generate_plots.py results/phase5_quick
```

## Documentation

- [Architecture](docs/architecture.md) — component boundaries and data flow
- [Research scope](docs/research_scope.md) — claims the prototype does and does not make
- [Model assumptions](docs/model_assumptions.md) — simulator and controller assumptions
- [Verification scope](docs/verification_scope.md) — bounded properties and formal limitations
- [Recovery-envelope specification](docs/formal/recovery_envelope_specification.md) — formal state, failure, and disturbance envelope
- [Reproducibility](docs/reproducibility.md) — environments, artifacts, seeds, and rerun guidance
- [3D replay](docs/replay_3d.md) — read-only visualization workflow
- [Artifact guide](docs/artifact_guide.md) — public-release manifests and provenance

## Project status

- [x] Phase 1 — repository structure, configuration, schemas, simulator, runner, and logging
- [x] Phase 2 — concurrent failures, deterministic fallback, emergency modes, diversion logic, and metrics
- [x] Phase 3 — demonstrations, PyTorch imitation policy, checkpointing, and baseline evaluation
- [x] Phase 4 — named properties, bounded Z3 abstraction, runtime monitor, and verification reports
- [x] Phase 5 — paired experiments, uncertainty estimates, plots, tables, and replay

## Limitations and ethics

The simulator is intentionally simplified. This project does **not** prove a neural network, simulator, or aircraft universally safe. It does not model the full aerodynamic, structural, sensor, actuator, human-factors, hardware, certification, or regulatory environment required for real flight. Formal verification is meaningful only inside the documented abstraction and bounds.

The optional Cessna FBX model is not distributed because its redistribution license could not be established. A legally licensed asset may be supplied locally, or the replay viewer will use its primitive fallback. See [asset licensing](docs/asset_licensing.md).

## Citation and contribution

Citation metadata is available in [CITATION.cff](CITATION.cff). Focused contributions that improve reproducibility, tests, documentation, or the fidelity of explicitly bounded models are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
