# Pilot Dataset Approval Freeze

Status: **APPROVED PILOT EXPORTED; PILOT IS NOT THE FINAL TRAINING CORPUS**

## Immutable identities

- Source candidate: `final_policy_pilot_v2`
- Approved dataset: `final_policy_pilot_v2_approved_v1`
- Source Git revision: `d3df91ac4d92be40e029d9ab9df916c4222018f4`
- Schema: version 2; 42 ordered inputs; five ordered controls
- Validator: `candidate-validator-v3-independent-contracts`
- Validator logical hash: `78a9ae1c2861970ea0ac4622e13826337444195fa5db575c470b5bdaedd90680`
- Disposition audit version: 1
- Disposition audit hash: `8fb5cda7eec6d7461820d6a038cc6a2410355e61409bbd842f8a230f34e24b74`
- Dispositions: 6,234 APPROVE, 0 REJECT, 0 HUMAN_REVIEW_REQUIRED
- Export timestamp: `2026-08-15T09:34:49-07:00`

## Source hashes

- Generation manifest: `1c2d995a65ba37d72cafe931bcdf4ffffad6d466b80697981a64d85d74301bdf`
- Raw transitions: `40f3b590e3e585a6b8ab0d067cfe9fec768dbe999873bf0be9374e804ded3446`
- Raw flights: `93dc3e3149a5ae6ab8e1dd056b8a04a761d1977190d28420a2bcb01606adcb66`
- Validator report file: `aef0962b245194514f54dfeffcdfe62af76f0baec642cef51f33d9a8de0c4ad8`

## Approved hashes

- Logical approved dataset: `12ec8abc042dd37998cb36d5515bf9c29fe74a172ede5d6e5bdf460c7f2eb22b`
- Approved manifest: `3ab28e2a60648bf70611065a59f1b04205f7165ae617a5ec3983d94e50796273`
- Transitions: `40f3b590e3e585a6b8ab0d067cfe9fec768dbe999873bf0be9374e804ded3446`
- Flights: `93dc3e3149a5ae6ab8e1dd056b8a04a761d1977190d28420a2bcb01606adcb66`
- Pilot normalization provenance: `b2f03371be73d010f1db3a590f90658ad63f8c5ed61f968439980cbb97e1fd61`
- Integrity audit logical hash: `3f88351e18d076691c986aa4c2d281d30c0e08eb1ce297e7c1876c94815840cd`
- Integrity audit JSON: `e6b31ef3336426f8311c9aee0ecc4e8a946cd8be53c135d4c5957b052bbeb7e1`
- Integrity audit Markdown: `cc2941ed468abd764559524f82d29d98a97c8d0f4260fc96c791e8cab91a319f`

## Reconstruction command

```powershell
& ".\.venv311\Scripts\python.exe" scripts\export_approved_dataset.py `
  --dataset datasets\candidate_policy_data\final_policy_pilot_v2 `
  --approved-dataset-id final_policy_pilot_v2_approved_v1 `
  --disposition-manifest artifacts\research_stage\pilot_review_audit\automated_dispositions.json

& ".\.venv311\Scripts\python.exe" scripts\audit_approved_export.py `
  --candidate datasets\candidate_policy_data\final_policy_pilot_v2 `
  --approved datasets\approved_policy_data\final_policy_pilot_v2_approved_v1
```

The export contains 36 approved flights and 9,350 approved transitions, with zero rejected or omitted records. Exact mapping and complete serialized-value equality passed. The immutable candidate raw files were not modified. The sealed final test was untouched.

This freeze authorizes neither large-dataset generation nor final training. Pilot normalization statistics are provenance evidence only and are not frozen as final-training preprocessing.
