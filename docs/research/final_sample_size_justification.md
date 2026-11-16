# Prospective Sample-Size and Precision Justification

Decision: **SAMPLE_SIZE_REVISION_REQUIRED for 720; revised 1,152-flight design accepted.**

This is a prospective approximation, not a formal guarantee of statistical power. It uses complete flights as independent units and does not select sample size to force significance.

## Why 720 was revised

With 72 flights in an evaluation split, a 50% recovery proportion has an approximate Wilson 95% half-width of 11.3 percentage points; an 80% rate has roughly 9.2 points. Combining challenge and OOD-style stress gives 144 flights and approximately 8.1/6.5-point half-widths, but combining is legitimate only for a prespecified overall simulator-stress summary—not as a substitute for separate split results.

The decisive limitation was tier precision: 12 flights per tier per evaluation split gives an approximate 24.6-point half-width at 50%. That is too sparse for strong tier-specific conclusions.

## Revised design

The revised design has 1,152 flights: 576 train and 144 in each of development, challenge, OOD-style stress, and final sealed test. Every tier has 192 total flights: 96 train and 24 in each evaluation split.

At `n=144`, approximate Wilson half-width is 8.1 points at a 50% rate and 6.5 points near 80%. At `n=24`, tier-specific half-width remains about 19.4 points at 50%; tier results are therefore stratified uncertainty estimates, not strong confirmatory tests. Exact family-by-split cells contain six flights and are explicitly exploratory.

For a paired binary comparison with 30% discordant controller outcomes, approximate 80%-power minimum detectable differences are 18.1 points at 72 pairs and 12.8 points at 144 pairs. With 50% discordance they are about 23.3 and 16.5 points. These approximations motivate 144 paired final scenarios but do not guarantee detection. The minimum practically meaningful primary difference remains 15 percentage points.

For paired continuous outcomes, approximate standardized minimum detectable mean differences are 0.33 at 72 pairs, 0.23 at 144, and 0.17 at 288. Recovery-time analysis is secondary and reports paired bootstrap intervals, censoring, and denominators.

## Operational scale

Pilot mean length was 259.7 transitions per flight. The revised design therefore expects approximately 299,000 transitions, with a 230,400–345,600 range for 20–30 second flights at 10 Hz. At the pilot's 6,537 bytes per JSONL transition, raw transitions are estimated at roughly 1.95 GB, with a 1.51–2.26 GB range before manifests, validation reports, and review packets. Generation-time estimates remain unmeasured; a timed non-sealed generation pilot must establish throughput before authorization.

This size balances aggregate precision, tier coverage, five-seed training, and operational feasibility. It does not make exact family-level comparisons confirmatory.
