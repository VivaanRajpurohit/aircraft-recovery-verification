# Final Generation Freeze

Decision: **FINAL_DATASET_DESIGN_APPROVED**, revised to 1,152 flights. Status: **generation not authorized**.

The authoritative manifest is `artifacts/research_stage/final_generation_manifest.json`, produced by `scripts/freeze_final_generation_design.py`. It freezes dataset ID `final_policy_candidate_v1`, schema v2, `SimpleAircraftSimulator`, the hashed expert manifest, 20-family scenario matrix, six-tier ranges, 576/144/144/144/144 flight splits, seed registry, 2.0 s warmup/vertical-speed plan, metric/termination contract, monitor, fallback, held-out matrix, and evaluation analysis plan. Current logical freeze hash: `fcfde639acb99d85b17477dee7da5743da305b46a32cb0b4293fd9c9423a1cd4`.

Expected volume is approximately 299,000 transitions (230,400–345,600) and 1.95 GB raw (about 1.51–2.26 GB based on approved-pilot bytes/transition). A defensible generation-time estimate is unavailable because the pilot provenance does not contain a dedicated comparable wall-clock generation measurement; the generation run must record start/end UTC, elapsed seconds, hardware, worker count, and achieved flights/transitions per second.

The manifest flags `generation_authorized`, `training_authorized`, and `final_sealed_test_authorized` false. Reproducing the manifest hashes metadata only; it does not generate or open trajectories. A separately approved generation must refuse any hash mismatch and must write a new immutable output directory rather than modify the approved pilot.
