# Final pre-sealed freeze v2.1B

Date: 2026-09-20

The single v2.1B path is frozen after completion of the representative-checkpoint Challenge and OOD-style simulator stress evaluations. Challenge and OOD-style results did not change the controller, monitor, fallback, metrics, analysis, or reporting configuration.

The machine-readable authority is `artifacts/research_stage/final_presealed_freeze_v2_1b.json`. It binds the representative seed 17001 checkpoint, all five retained training checkpoints, candidate and approved dataset identities, train-only normalization, model-selection freeze, final metric contract, monitor and fallback, completed nonsealed results, sealed evaluator, statistical analysis, tables, figures, and manuscript-population implementation.

The final sealed split contains 144 scenarios and may be evaluated exactly once on the frozen representative seed 17001 checkpoint after this freeze is committed unchanged. The only authorized primary comparison is the paired representative neural controller without versus with the frozen runtime monitor. No further training, tuning, architecture comparison, seed selection, model reselection, or alternate evaluation branch is permitted.

At freeze time, the complete repository suite passed (343 passed, 1 skipped), `git diff --check` passed, the training/model-selection/Challenge/OOD loaders rejected `final_sealed_test`, and sealed outcome access remained zero.

