# Final pre-sealed freeze hash reconciliation v2.1B

Date: 2026-09-20
Status: complete post-sealed provenance reconciliation

The immutable pre-sealed freeze remains unchanged. Its file SHA-256 is `a9b65c919d88ddf1fc039e9930cf5df5de4d50c0ba6b14b63147a2dbdf7224d8` and its logical SHA-256 is `1f48d27cede12d0ba5053fd9fc1e0b59f3fd155d849745ccd6b48e0f781c114c`.

During final integrity QA, four secondary checkpoint hashes manually copied into the freeze's informational `five_seed_checkpoint_sha256` map were found to contain incorrectly transcribed suffixes. The authoritative five-seed provenance and all five checkpoint files agree exactly.

| Seed | Freeze value | Actual file SHA-256 | Match | Used in evaluation |
|---:|---|---|:---:|:---:|
| 17001 | `175064a9d978edff8542e41d626ecad2e0cb04b4552ed055668043389b369ef7` | `175064a9d978edff8542e41d626ecad2e0cb04b4552ed055668043389b369ef7` | Yes | Yes: Challenge, OOD-style simulator stress, sealed |
| 17002 | `30175b514859783f8063400c16602ad158c514dc253c8ce42899a04b43be2d68` | `30175b6046f7fbf67b28e5fdcff04c10e6344a1030524f7d78c37dda2f4b3b27` | No | No |
| 17003 | `662a04d225b016a0af277aa5ddc92c8aee867e1001156da1023dc4b46842fd48` | `662a04811bacae16d4f416563b2c84f42c908ab27d5d166a3c753a52b69260ad` | No | No |
| 17004 | `06016b03c6414af5fe9a0805ca2ee7448b55da3a280eff3a36f9d6e26a82fd6f` | `06016ba49fdbf443e6d8967b7f2b93a159b6f4ff956afd62c728e49b21a9670d` | No | No |
| 17005 | `8d49bf89682c1d173ed4d0069aaf11c855ed6b9994f38c56c5a0064f3c1a4f89` | `8d49bf9b975486dab43644cf664e4be45d55fc4b30c179da3e356a396e33eb8f` | No | No |

Checkpoint paths follow `checkpoints/final_policy_v1/moderate/final_training_v2_1a/seed_<seed>/best.pt`; the machine-readable companion records each full path. Seed 17001 was the performance-independent representative and the only checkpoint authorized or used for Challenge, OOD-style simulator stress, and final sealed evaluation. Its hash was correct in the immutable freeze and in all evaluation summaries.

This correction changes no evaluated model, dataset, normalization artifact, metric, monitor, fallback, controller, statistical procedure, scientific decision, or result. It adds a provenance layer; it does not rewrite history.

Machine-readable reconciliation logical SHA-256: `1958402f440ce9d78f4caa63132e33e236759574df7183949a41c97bfc7d9083`.
