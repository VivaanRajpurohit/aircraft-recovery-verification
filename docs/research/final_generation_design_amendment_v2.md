# Proposed Final Generation Design Amendment v2

Status: **PROPOSED — STATIC AUDIT PASS — GENERATION NOT AUTHORIZED**.

## Reason

The v1 freeze was cryptographically intact but semantically impossible to execute without interpretation: one global held-out severity/timing list was to be crossed through tiers having incompatible numeric ranges. Train/development values for several factors were also unspecified, and the Tier 4 crosswind family lacked an explicit numeric wind bound. No records had been generated, so this amendment is prospective and uninformed by model or sealed outcomes.

Historical v1 freeze hash: `fcfde639acb99d85b17477dee7da5743da305b46a32cb0b4293fd9c9423a1cd4`.

## Complete v1 conflict table

`UNSPECIFIED` means v1 named a range/factor but assigned no split-specific values. The machine-complete row table, including every sensor dimension, is in `artifacts/research_stage/final_design_amendment_v2_audit.json`.

| Tier | Factor | Allowed v1 range | Train | Development | Challenge | OOD-style | Sealed | Result |
|---:|---|---|---|---|---|---|---|---|
| 0 | failure severity/timing | N/A | N/A | N/A | N/A | N/A | N/A | compatible nominal |
| 1 | component severity | .20–.35 | unspecified | unspecified | .40/.65/.82 | .22/.52/.85 | .39/.64/.84 | incompatible |
| 1 | activation (s) | 3–8 | unspecified | unspecified | 2.5/5/7.5 | 2.25/5.5/7.25 | 2.75/5.25/7.75 | incompatible |
| 2 | component severity | .45–.60 | unspecified | unspecified | .40/.65/.82 | .22/.52/.85 | .39/.64/.84 | incompatible |
| 2 | activation (s) | 2–7 | unspecified | unspecified | 2.5/5/7.5 | 2.25/5.5/7.25 | 2.75/5.25/7.75 | incompatible |
| 3 | component severity | .30–.60 | unspecified | unspecified | .40/.65/.82 | .22/.52/.85 | .39/.64/.84 | incompatible |
| 3 | activation (s) | 2–7 | unspecified | unspecified | 2.5/5/7.5 | 2.25/5.5/7.25 | 2.75/5.25/7.75 | incompatible |
| 4 | component severity | .70–.80 | unspecified | unspecified | .40/.65/.82 | .22/.52/.85 | .39/.64/.84 | incompatible |
| 4 | activation (s) | 2–6 | unspecified | unspecified | 2.5/5/7.5 | 2.25/5.5/7.25 | 2.75/5.25/7.75 | incompatible |
| 5 | component severity | .80–.85 | unspecified | unspecified | .40/.65/.82 | .22/.52/.85 | .39/.64/.84 | incompatible |
| 5 | activation (s) | 2–5 | unspecified | unspecified | 2.5/5/7.5 | 2.25/5.5/7.25 | 2.75/5.25/7.75 | incompatible |
| 2 | delay (s) | .30–.50 | unspecified | unspecified | unspecified | unspecified | unspecified | underdefined |
| 3 | delay (s) | .30–.60 | unspecified | unspecified | unspecified | unspecified | unspecified | underdefined |
| 4 | delay (s) | .60–.80 | unspecified | unspecified | unspecified | unspecified | unspecified | underdefined |
| 5 | delay (s) | .70–.80 | unspecified | unspecified | unspecified | unspecified | unspecified | underdefined |
| 2–5 | airspeed/altitude/heading bias | tier-specific v1 bounds | unspecified | unspecified | unspecified | unspecified | unspecified | underdefined |
| 2–5 | turbulence | tier-specific v1 bounds | unspecified | unspecified | unspecified | unspecified | unspecified | underdefined |
| 4 | crosswind maximum (kt) | undefined | unspecified | unspecified | unspecified | unspecified | unspecified | missing bound |
| 1–5 | actuator effectiveness | derived as `1 − surface severity` but global severity incompatible | unspecified | unspecified | unspecified | unspecified | unspecified | underdefined |
| 1–5 | partial-thrust remaining fraction | derived as `1 − partial-thrust severity`; complete engine loss fixed at zero | unspecified | unspecified | unspecified | unspecified | unspecified | underdefined |

## Amendment principle

Held-out numeric values are assigned **per tier**, never globally. Every split remains inside its tier's closed range. A complete engine loss stays severity 1.0/availability zero and is not incorrectly assigned a surface/partial-loss severity; its tier role is determined by surrounding initial conditions and concurrent factors. Surface effectiveness and partial-thrust remaining fraction are deterministically `1 − applicable component severity`.

## Proposed per-tier component severity

| Tier | Train | Development | Challenge | OOD-style | Sealed |
|---:|---|---|---|---|---|
| 0 | N/A | N/A | N/A | N/A | N/A |
| 1 | .20/.24/.28/.32 | .22/.30 | .26/.34 | .21/.31/.35 | .23/.29/.33 |
| 2 | .45/.50/.55/.60 | .47/.57 | .49/.59 | .46/.53/.58 | .48/.52/.56 |
| 3 | .30/.40/.50/.60 | .35/.55 | .45/.58 | .32/.48/.59 | .34/.46/.56 |
| 4 | .70/.74/.78/.80 | .72/.76 | .73/.79 | .71/.77 | .725/.755/.785 |
| 5 | .80/.825/.85 | .81/.84 | .815/.845 | .805/.835 | .8125/.83/.8475 |

These values apply only to the component named by each family mapping. Exact delay, sensor-bias, turbulence, and crosswind schedules are in `final_heldout_factor_matrix_v2.yaml`.

## Proposed per-tier activation seconds

| Tier | Train | Development | Challenge | OOD-style | Sealed |
|---:|---|---|---|---|---|
| 0 | N/A | N/A | N/A | N/A | N/A |
| 1 | 3/4/6/8 | 3.5/6.5 | 4.5/7.5 | 3.25/5.5/7.75 | 3.75/5.25/7.25 |
| 2 | 2/3.5/5.5/7 | 2.5/6 | 2.75/6.25 | 2.25/4.75/6.75 | 2.5/4.25/6.5 |
| 3 | 2/3.5/5.5/7 | 2.5/6 | 2.75/6.25 | 2.25/4.75/6.75 | 2.5/4.25/6.5 |
| 4 | 2/3/4.5/6 | 2.5/5.5 | 3.5/5 | 2.25/4.25/5.75 | 2.75/3.75/5.25 |
| 5 | 2/3/4/5 | 2.5/4.5 | 2.25/3.5/4.75 | 2.125/3.25/4.625 | 2.375/3.75/4.875 |

## Held-out semantics

| Split | Seed | Value | Timing | Combination | Severity-band |
|---|---|---|---|---|---|
| Train | assigned partition | support values | support timings | training families | no |
| Development | holdout | selection-only values | selection-only timings | known combinations | no |
| Challenge | holdout | unseen in-range values | unseen in-range timings | known hard families | no |
| OOD-style simulator stress | holdout | unseen in-range values | unseen in-range timings | supported compositions/shifts | only declared boundary-relative shifts |
| Final sealed test | holdout | unseen in-range values | unseen in-range timings | selected frozen combinations | no post-freeze tuning |

Not every split must exercise every holdout type. “OOD-style” remains a simulator-domain statement only.

## Sealed spacing

The old global .39/.64/.84 and 2.75/5.25/7.75 values are replaced because they either violate tiers or become near/invalid duplicates when globally crossed. Sealed values use deterministic interior midpoints between nonsealed support points within each tier. Initial airspeed, bank, pitch, altitude, and clearance use frozen normalized within-tier grid fractions: train 0/.3333/.6667/1, development .20/.80, challenge .10/.90, OOD-style .05/.95, and sealed .15/.50/.85. Selection used only numeric spacing; no policy has been trained and no outcome was consulted.

## Changed and unchanged components

Changed: global held-out schedule becomes per-tier; all factor split values become explicit; Tier 4 crosswind maximum receives the existing supported bound 20–35 kt; candidate split schema recognizes `final_sealed_test` while retaining legacy `test` and `validation` aliases.

Unchanged: 1,152 total flights; 576/144/144/144/144 splits; 192 per tier; all 20 joint-family counts; seed ranges and five training seeds; 152/152 engine-side balance; metric/recovery/safety/termination definitions; primary endpoint; sample-size interpretation; monitor; fallback; statistical plan. Each evaluation split still has 24 flights/tier and family-level cells remain exploratory.

## Audit result and impact on claims

Static audit: **PASS**. All values are within tier bounds; mandatory holdouts do not collide with train; sealed values are distinct from challenge/OOD where required; every family has parameter semantics; arithmetic, seeds, and engine balance pass. Audit hash: `c20b20d0947ac9826242612e33f9bcfd0588ed8dc473fd1c60262d178a00d945`.

The amendment repairs executability and provenance. It does not strengthen statistical claims, add real-world validity, or authorize generation/training/sealed evaluation.
