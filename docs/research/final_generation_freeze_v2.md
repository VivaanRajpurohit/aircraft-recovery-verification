# Proposed Final Generation Freeze v2

Status: **PENDING REVIEW — NOT ACTIVE — GENERATION NOT AUTHORIZED**.

V2 supersedes the historical v1 design before any dataset records were generated. V1 freeze hash: `fcfde639acb99d85b17477dee7da5743da305b46a32cb0b4293fd9c9423a1cd4`. Reason: the global held-out values conflicted with numeric tier contracts and several factor schedules were underdefined.

The proposed machine manifest is `artifacts/research_stage/final_generation_manifest_v2.json`. Proposed v2 freeze hash: `92d5a0137c7e185ce49060d0d1ada6ab631410a2914ba465d377aa43b83004bc`. Static amendment audit hash: `c20b20d0947ac9826242612e33f9bcfd0588ed8dc473fd1c60262d178a00d945`.

Amended config hashes:

- design v2: `5e51334c862edb50194179e5fdb5b855aa7e24dba25638b3ea4e409239c443fc`
- tier v2 overlay: `1584b21dcfd0feb4c165eb5d793a920ae59a0df1a3976d1cb783136173ced291`
- held-out matrix v2: `43e761732e34c9bb8e2beb5b49d8515d14990a53357e8ae6098f1ddfc6a76063`
- evaluation v2 overlay: `dc573722fbaee44e7ae9f5bd21c0eea11cae113ec8e381aae41290768b82cb6c`

Unchanged metrics, monitor, fallback, seed registry, expert manifest, v1 scenario counts, and source dependencies are individually hashed in the manifest. `artifacts/research_stage/final_generation_freeze_registry.json` marks v1 historical, v2 pending review, and the active generation freeze as null.

Review approval must be explicit. A later authorization step may mark v2 active but must not rewrite this proposed manifest or infer approval from test success.
