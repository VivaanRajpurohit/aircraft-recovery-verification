# Final Representative Checkpoint Amendment v2.1B

Date: 2026-09-20

Status: **ACTIVE IMMUTABLE PRE-NONSEALED-EVALUATION AMENDMENT**

This prospective amendment resolves the absence of a representative-checkpoint
rule after the five fixed training runs and before any Challenge, OOD-style
simulator stress, or final sealed outcome is accessed. It does not modify any
prior frozen artifact.

## Performance-independent representative rule

Seed `17001` is the primary representative neural checkpoint because it is the
first entry in the already-frozen final seed registry: `17001, 17002, 17003,
17004, 17005`. This positional rule is independent of observed DEVELOPMENT
performance. Seed 17001 is not claimed to be the best seed.

The representative checkpoint SHA-256 is
`175064a9d978edff8542e41d626ecad2e0cb04b4552ed055668043389b369ef7`.
It implements the frozen `[256, 256, 128]` SiLU architecture with 42 inputs,
five bounded control outputs, and 110,341 parameters.

The final-training configuration logical hash is
`288b96bbeed75d4ce19793e28198be25430f8dd260604fb654276c405467917a`.
The five-seed provenance logical hash is
`b3c06be0a6f519f5860dae4acd18d758a22c2ba3178d5a5d52152a4d1a58c9de`.

All five trained checkpoints remain frozen parts of the experiment and must be
retained and reported for training-seed variability analysis. Only the seed
17001 representative checkpoint is authorized for held-out evaluation. The
single path is model-selection freeze, Challenge, OOD-style simulator stress,
then pre-sealed freeze. No optional all-five held-out evaluation or alternate
model branch is authorized. Challenge and OOD-style simulator stress outcomes
cannot alter the architecture, optimizer, hyperparameters, representative
checkpoint, or five-checkpoint set. Final sealed test access remains prohibited.

The machine-readable companion is
`artifacts/research_stage/final_representative_checkpoint_amendment_v2_1b.json`.
