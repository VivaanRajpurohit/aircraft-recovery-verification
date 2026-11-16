# Minimal Holdout-Semantics Amendment v2.1

Status: **STATIC AUDIT PASS — PENDING EXPLICIT FREEZE ACTIVATION — NO GENERATION AUTHORITY**.

The original v2 proposal remains preserved at freeze hash `92d5a0137c7e185ce49060d0d1ada6ab631410a2914ba465d377aa43b83004bc`. This overlay corrects only strict sealed-value collisions and holdout terminology. No flight, split, tier, family, seed, metric, monitor, fallback, or statistical quantity changes.

## Violations in the original v2 proposal

Under the documented global strict-sealed interpretation, v2 had these contract violations:

- component severity: sealed `.34`, `.46`, and `.48` had appeared in earlier splits;
- activation: sealed `2.5`, `2.75`, `4.25`, and `6.5` seconds had appeared earlier;
- derived actuator effectiveness: `.66`, `.54`, and `.52` had appeared earlier;
- derived partial-thrust fraction: the same `.66`, `.54`, and `.52` collisions followed from `1 − severity`.

No crosswind strict-value collision existed. The reported Tier 2/Tier 3 2.5 s collision was therefore real but not the only violation.

## Minimal value changes

| Tier | Factor | V2 sealed | V2.1 sealed | Rule |
|---:|---|---|---|---|
| 2 | component severity | .48/.52/.56 | .485/.52/.56 | nearest legal midpoint avoiding all earlier exposures |
| 3 | component severity | .34/.46/.56 | .345/.465/.56 | nearest legal midpoints |
| 2 | activation (s) | 2.5/4.25/6.5 | 2.625/4.375/6.375 | midpoints between adjacent earlier exposure values |
| 3 | activation (s) | 2.5/4.25/6.5 | 2.625/4.375/6.375 | same deterministic schedule |
| 4 | activation (s) | 2.75/3.75/5.25 | 2.875/3.75/5.25 | nearest legal midpoint |

No controller result or trajectory outcome was available or consulted.

## Final holdout definitions

- `STRICT_VALUE_HOLDOUT`: component severity, failure activation, derived surface effectiveness, derived partial-thrust fraction, and Tier 4 crosswind maximum. Every sealed numeric value is absent from train, development, challenge, and OOD-style stress globally.
- `MODEL_SELECTION_HOLDOUT`: sealed actuator delay is absent from train and development but may occur in challenge/OOD. Challenge and OOD severity/timing values are model-selection holdouts when identified by their assigned tier and component semantic.
- `COMBINATION_HOLDOUT`: initial airspeed, bank, pitch, altitude, clearance, and the canonical joint scenario signature. Individual values may recur, but the exact sealed signature may not.
- `SEED_ONLY_HOLDOUT`: deterministic sensor bias, turbulence, engine side, and random seed. Numeric repetitions are allowed and disclosed.
- `NOT_A_HOLDOUT_FACTOR`: warmup-derived vertical-speed bin. It is measured and characterized, not selected for novelty.

The canonical signature is `(tier, family, component/fixed-failure severity, activation, delay, sensor kind/value, turbulence, crosswind, initial-grid fraction, engine side)`.

## Legitimate collisions

Earlier evaluation splits share some raw numeric severity and timing values across different tiers. Those are legitimate because challenge/OOD factor identity includes the assigned tier and component semantic; the within-tier train/development intersections are empty. Sealed delays `.375/.475` occur in challenge and `.72` occurs in OOD, which is allowed under `MODEL_SELECTION_HOLDOUT`. Sensor-bias and turbulence repetitions are allowed under `SEED_ONLY_HOLDOUT`. All 20 family names occur in every split by the unchanged scenario matrix; novelty is in the exact canonical signature, not the family name alone.

The complete value-by-tier/split matrix and every pairwise intersection are recorded in `artifacts/research_stage/final_holdout_exposure_v2_1.json`.

## Audit results

- strict sealed intersections: empty for all declared strict factors;
- challenge tier-conditioned model-selection intersections: empty;
- OOD tier-conditioned model-selection intersections: empty;
- seed intersections: empty;
- sealed canonical signature duplicates against train/development/challenge/OOD: 0/0/0/0;
- same family names with different numeric signatures: all 20, as intended;
- sealed outcomes accessed: false.

Holdout exposure audit hash: `1b55c5b90fd4220ba96c829e53e0899be3dab3e7aa82c661b7c39ace9909b070`.
