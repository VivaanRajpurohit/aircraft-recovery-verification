# Final Approved Dataset Freeze

- Approved dataset ID: `final_policy_approved_v2_1`
- Approved logical hash: `bc69cbcb5bee7189c091dba51159f8938bdd661b3793007f45c7f9eb2897b43a`
- Candidate logical hash: `4db2d75ab341e4573019d09176507b03851d7d26dd2eb360d42ecfc2adc28706`
- Generation freeze hash: `c0b831ede8a5d75027dcce70ec1c116fc5ec58d479cf500939f2075e3d16d0dc`
- Approved flights/transitions: 1008 / 262080
- Split counts: `{"challenge": 144, "development": 144, "ood_stress": 144, "train": 576}`
- Excluded sealed flights/transitions: 144 / 37440
- Integrity-audit hash: `31eb68b767824b9dfedd9b1ba2a0afa99306c6670b709ac55585de40fd3824b2`
- Approved-manifest SHA-256: `4861adf9c6e49b8817d19838cb404a010d62cfaf226ff3da47a91d34f6dc0e3f`
- Export command: `.\.venv311\Scripts\python.exe scripts\export_final_approved_v2_1.py`
- Source Git revision: `d3df91ac4d92be40e029d9ab9df916c4222018f4`
- Freeze timestamp: `2026-08-15T19:07:30.076649+00:00`

The final sealed test is excluded from training, normalization, and model selection.
