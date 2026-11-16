# Runtime-Assured Neural Recovery Under Concurrent Simulated Failures

This is the versioned v2.1B research evidence release, audited on 15 November 2026. It contains the low-order simulator, frozen training/evaluation code, configurations, provenance, and scripts for checking the recorded results. The journal manuscript and identifying submission materials are excluded. This release does not signify journal acceptance.

## Findings and interpretation

The original experiment contains 432 distinct matched scenarios across Challenge, simulator stress, and final test, each evaluated with and without monitoring (864 runs). Both final-test conditions recovered in 144/144 flights with zero recorded simulator-domain safety violations. The zero observed difference and degenerate paired bootstrap interval [0,0] do not demonstrate equivalence or a recovery/safety benefit from monitoring.

Recovery requires **50 consecutive in-envelope samples at 0.1-second intervals**. This nominal five-second hold spans **4.9 seconds between its first and last timestamps**. Onset is backdated to the first sample relative to the latest observable/detected failure activation (zero for nominal cases). The five recovery thresholds are independent of separately assessed safety properties. Continuous intersample containment and real-aircraft performance are not established. Formal interval checks concern the same low-order simulator under recorded actions, not closed-loop neural-controller proof.

The supplementary three-architecture Challenge comparison is exploratory and postdates the original final test. It adds two fits and 576 runs; original moderate-policy records are reused. Combined totals are 14 fits and 1,440 condition records on 432 distinct scenarios.

## Frozen data and checkpoints

Use tag `v2.1b-research-audit-2026-11-15` or branch `publication/jate-v2.1b-2026-11-15`. Three ZIP assets in the matching GitHub release supply candidate data, approved nonsealed data, original evaluations, 14 best checkpoints, and training records. Download size is about 358 MB; allow 6 GB for extraction and outputs.

```console
python scripts/fetch_research_release.py
```

The downloader verifies the SHA-256 values in `artifacts/publication_release/release_assets.json`, rejects unsafe paths, and refuses to replace different existing artifacts. Alternatively, verify SHA256SUMS.txt and extract all three release ZIPs in the repository root. Do not retrain into frozen checkpoint paths.

## Clean environment and reproduction

The audit used CPython 3.11.9 on Windows and PyTorch 2.13.0 CPU. The original used the CUDA variant; cross-device numerical tolerances are explicit in the scripts. Historical latency is reaggregated from stored timings, not claimed to transfer to other devices.

```console
python -m venv .venv-publication
# Activate the environment for your shell, then:
python -m pip install -r requirements-publication-cpu.txt
python -m pip install --no-deps -e .
python scripts/audit_publication.py
python scripts/audit_training_and_tables.py
python scripts/publication_figures.py
python -m pytest --ignore=tests/unit/test_generated_aircraft.py --ignore=tests/unit/test_generated_aircraft_v2.py --ignore=tests/unit/test_generated_aircraft_audit.py -k "not headless_viewer_identifies_live_simulation"
```

New audit outputs go to `results/publication_audit`; figures (PDF/SVG and 600 dpi PNG/TIFF) go to `results/publication_figures`. Frozen outputs are not overwritten. The completed audit snapshot is in `artifacts/publication_release`.

All 224,640 original action steps were replayed and all 864 recovery/safety outcomes reproduced. Eighteen stratified 260-step traces were checked against fresh CPU neural proposals. Five final checkpoints reproduced development MSE/MAE/RMSE within 1e-7 absolute tolerance. No training or complete fresh monitored batch was rerun.

The relevant suite passed **326 tests, with nine existing skips and one deselected viewer test**. Three optional generated-aircraft asset test modules were excluded. Their demonstration assets are not part of this scientific release; the unrestricted visualization suite is not claimed to pass.

## Provenance and limits

See `docs/research/PUBLICATION_AUDIT.md`. Historical manifests remain unchanged, including original source and file hashes. This public branch starts from public main `73aca720fac0769b405ddf1b868d6cc3734f5913` plus a reviewed science-only snapshot of local commit `7566df3`; private manuscript history was not merged. The new release commit identifies the public snapshot, while historical hashes identify the original runs. Existing reconciliation records correct earlier transcription of seed 17002--17005 hashes; representative seed 17001 was not changed or reselected.

The refitting pipeline is supplied but was not rerun. Independent high-fidelity validation, hardware tests, flight tests, standalone baseline comparisons and real-world generalization remain outside the evidence. Optional prototype outputs, duplicate generation batches, private diagnostics and worker logs are excluded. Historical manifests may reference private or historical artifacts outside this release.

## License

The existing LICENSE retains all rights and its stated research-inspection/review scope, with other uses requiring separate authorization. This repository is **not MIT-licensed or advertised as open source**. The inconsistent MIT badge was removed; no license rights were changed. Third-party software retains its own licenses, and optional aircraft artwork is excluded.
