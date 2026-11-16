# Supplementary architecture comparison

Date: 2026-10-11, America/Los_Angeles

The author requested additional model comparisons within the existing aircraft-recovery study. This supplementary experiment is specified after the original v2.1B outcomes were observed. It is exploratory and does not amend, replace, or reopen the original confirmatory experiment.

## Fixed comparison

Compare the existing small reference [128,64] ReLU (14,085 parameters), current reference [256,128] ReLU (44,549 parameters), and moderate [256,256,128] SiLU (110,341 parameters). All use 42 inputs and five bounded controls in the original order. These comparisons vary architecture and activation jointly; they do not isolate a causal effect of network width.

Train the two smaller architectures once each with seed 17001, AdamW, learning rate 0.001, batch size 512, weight decay 0.0001, maximum 50 epochs, patience 8, improvement tolerance 1e-10, and gradient norm clipping 1.0. Optimize MSE on TRAIN only. Select each epoch checkpoint by minimum DEVELOPMENT MSE. Reuse the original moderate representative checkpoint from seed 17001, whose hash is 175064a9d978edff8542e41d626ecad2e0cb04b4552ed055668043389b369ef7. Do not search additional hyperparameters or select a new primary model.

Use only final_policy_approved_v2_1 and its immutable train-only normalization. Reconstruct the same 144 Challenge scenarios, 24 per tier, and evaluate each new model with monitoring disabled and enabled. Reuse the existing moderate-model Challenge records instead of rerunning the original experiment. The complete comparison is 432 model-scenario pairs / 864 controller-condition records, including 144 previously completed moderate-model pairs. No OOD-style stress or final sealed data are opened by this extension.

## Reporting

Report every architecture and both conditions. Outcomes are strict recovery success/rate, recovery onset delay conditional on recovery, simulator-domain safety-violation incidence/duration, monitor intervention and fallback frequencies. Describe all six tiers and isolated/concurrent/family groups. Recovery onset is the start of a five-second qualifying hold, using the existing timer reference. It is distinct from inference or monitor latency and from time to completion of the dwell. Do not assume higher tiers require longer time; measure it. Do not claim that intervention prevented violations without outcome evidence.

The two additional fits use one common seed; five-seed variability evidence remains specific to the original final model. Reused and new latency records come from different execution sessions and must not support a controlled speed ranking. No new confirmatory p values, tuning, or outcome-dependent reselection are permitted. Any observed equality is reported as equality.

## Interruption and resumption

The worker writes a new output namespace. Each completed epoch is an immutable, atomically published checkpoint containing optimizer, shuffle-generator and random states, full history, best epoch, and stopping counters. It resumes after the latest valid committed epoch. Each completed controller condition is committed with its complete trace and latencies; a pair marker is published only after both valid conditions exist. Completed conditions and pairs are verified and reused. Incomplete temporary files are retained and ignored, never substituted for completed results. An operating-system file lock prevents concurrent workers.

An append-only scientific plan and a content-addressed input snapshot are committed before fitting. Summary publication requires all 144 pairs for both new models and verification of the retained original records. Partial Challenge outcomes cannot change any model, configuration, or manuscript claim. Manuscript wording is updated from existing complete evidence first; new comparisons are added only after the entire comparison completes and is verified.

Resume command from the advanced repository:

    .venv311\Scripts\python.exe scripts\run_supplementary_architecture_comparison_20261011.py

Status and outputs reside under results/supplementary_architecture_comparison_20261011/. The machine-readable plan and protected-input snapshot reside under artifacts/research_stage/supplementary_architecture_comparison_20261011/.

## Editorial sources

The reporting separates task performance, constraint violations, and intervention activity, following the kinds of distinctions made in Hobbs et al., Run Time Assurance for Safety-Critical Systems (https://arxiv.org/abs/2110.03506), and Wagener et al., Safe Reinforcement Learning Using Advantage-Based Intervention (https://proceedings.mlr.press/v139/wagener21a.html). The manuscript uses original wording, and its claims remain specific to its own evidence.
