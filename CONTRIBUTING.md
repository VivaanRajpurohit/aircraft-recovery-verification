# Contributing

This repository is simulation-only research software. Contributions must not
connect it to aircraft, drones, avionics, external flight simulators, vehicles,
or physical control hardware.

Use Python 3.11, create a focused branch, add tests, and run:

```powershell
python -m compileall -q src scripts
python -m pytest
```

Scientific changes must include a change record under `docs/change_records/`
covering the objective, rationale, assumptions, affected artifacts, tests,
limitations, and reproduction command. Do not overwrite prior evidence.

Commit messages should identify scope, for example `feat(formal): ...`,
`test(simulator): ...`, or `docs(reproducibility): ...`.
