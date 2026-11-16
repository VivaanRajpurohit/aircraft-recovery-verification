# Publication evidence audit, 15 November 2026

Source: local branch codex/v2-1-architecture-screening, commit 7566df3. The new publication branch starts at public main 73aca720fac0769b405ddf1b868d6cc3734f5913 and imports science files without private manuscript history.

## Material corrections

- Recovery table incorrectly required independent safety status. The code checks five envelope thresholds; safety is separate. Description corrected; frozen outcomes/code unchanged.
- Fifty samples at dt=0.1 span 4.9 seconds between endpoints. Nominal five-second hold now described as a sampled criterion; 0.05-second onset is not hold completion. No continuous intersample containment is claimed.
- Engine thrust availability features 20 and 21 are fractions in [0,1], not binary flags. Runway descriptor sources 39--40 are navigation/airport records. Feature-table labels corrected.
- The public snapshot lacked final artifacts. Versioned release supplies final synthetic datasets, checkpoints, traces and provenance.
- MIT badge conflicted with the existing all-rights-reserved LICENSE. Badge removed; license unchanged.

## Verified evidence

artifacts/publication_release/audit.json records split counts, paired bootstrap calculations, latency reaggregation, intervention events, dataset hashes, checkpoint parameters and fresh-inference comparisons. Candidate and approved data contain 299,520 and 262,080 transitions. Scenario IDs and seeds are unique across partitions. All 864 original runs (224,640 steps) were replayed using recorded selected commands; executed controls, recovery, safety, and modifications agree. This uses the same simulator, not an independent plant.

Fresh CPU neural inference was checked at the replayed states of 18 stratified trajectories, one per tier/split (4,680 steps); maximum action discrepancy is below 2e-6. This is not a complete new monitored experiment.

training_tables.json verifies all 42 overall/tier architecture summary rows against 864 condition records. All five final checkpoints reproduce development errors within 1e-7 absolute CPU tolerance. Unperturbed formal validation reproduces 100 one-step cases and four recorded traces. Five perturbations reproduce one-step failures 100,90,100,100,100 out of 100 and rollout failures 4,4,4,4,0 out of four.

The four figures use scripts/publication_figures.py: code-derived architecture schematic; split recovery with Wilson intervals; final-test tier onset medians/ranges; monitor-rule event counts. plotted_values.json records all plotted data. Range bars are extrema, not confidence intervals. Rule events are neither independent flights nor observed safety violations.

Main table evidence: T1 failure semantics/configuration; T2 tier bounds; T3 implemented metric; T4 final flight records; T5 action traces; T6 formal perturbation runs. Supplement evidence: S1 notation; S2 partition counts; S3 observation vector and model schemas; S4 overall architecture records; S5 tier records. Submission documents remain private.

## Test scope and remaining limitations

Clean CPython 3.11.9, pinned dependencies and CPU PyTorch: 326 relevant tests passed, nine existing skips, one viewer test deselected, three optional generated-aircraft asset modules excluded. Missing Panda3D was installed; scientific fixtures restored from frozen artifacts. Unrestricted optional viewer tests require assets outside this release and are not reported as passing.

Release excludes duplicate generation batches, optional prototype validation outputs, private diagnostics, worker logs/locks and submission documents. Historical manifests can reference files outside the release; this is not a claim that every historical artifact is public. Released files support the documented verification commands.

No training refits or new scientific outcomes were introduced. Historical timing is hardware/session dependent; no architecture latency ranking is claimed. No independent real-aircraft validation, certification or demonstrated final-test monitor benefit follows. Broader reuse licensing requires the copyright holder's decision. Journal originality, exclusivity, author details, accessibility and live disclosure questions require author confirmation.
