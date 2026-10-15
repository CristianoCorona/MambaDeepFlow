# Pre-registration — night 6b: replacement of DfpNet's deep path with Mamba blocks

Decided on 2026-10-15, **after the test set was opened** (test passes of 2026-10-14 and 2026-10-15), before any run
(including the control runs) of the new models. Implementation: commit `3f4f785` (`mdf/deep_replace.py`, variants `TR`
and `DRv11`; docs/lab-notebook.md, entry "Night 6b: deep-path replacement models TR and DRv11"). This file is committed
on its own, before the control runs and before the launch of night 6b. Lab-notebook entry of the same date:
"PRE-REGISTRATION: night 6b".

## Experiment

Replacement of DfpNet's deep path (layer4-6, dlayer6-4) with **11 Mamba SS2D blocks at 8x8 (DRv11)**, parameters
matched to B (2 354 691 vs 2 332 035, +0.97 %), and the control **TR** (the same truncated network without the blocks,
493 315 parameters).

**All models**: split grouped-holdout, seed 0 (`d84da2c`), 133 epochs, Adam lr 6e-4, betas (0.5, 0.999), L1 loss,
batch 10, final epoch (last checkpoint), protocol unchanged. **N = 11 is fixed now and is not modified after the
validation.**

Runs (night 6b, `~/runs/night6b/`): DRv11, B-GH (= B on grouped-holdout), TR, seeds 0-4, interleaved by seed
(DRv11 s0, B-GH s0, TR s0, DRv11 s1, ...); then MD8-GH and MDv8-16-32-GH seeds 0-2 (moved from night 6). Each run is
evaluated after training on `val` (unseen shapes) and `val_seen` (seen shapes). Labels in the analyses: `DRv11`,
`B-GH`, `TR`, `MD8`, `MDv8-16-32`.

## CONFIRMATORY (the only one)

On the official test set: **rel_all** (original definition, mean over samples of the per-sample ratio) of DRv11 seeds
0-4 against B-GH seeds 0-4. Estimate of the difference **DRv11 − B-GH** and its 95 % interval from the **two-level
bootstrap (seeds + profiles, 10 000 repetitions, seed 0)** as in the main analysis
(`scripts/final_analysis.py --confirm-dr`). Outcomes:

- **"improvement demonstrated"** if the upper bound < 0;
- **"worsening demonstrated"** if the lower bound > 0;
- otherwise **"no demonstrable difference"**, reporting the estimate and the interval.

Descriptive, without decision: rel_p, rel_vx, rel_vy (mean, aggregate, median), MAE per channel.

## TR CONTROL (descriptive, validation and test, no decision)

TR − B-GH and DRv11 − TR, same metrics and same bootstrap. Interpretation fixed now:

- if TR ≈ B-GH, the deep path is not necessary and DRv11 − B-GH says nothing about global mixing;
- if TR is worse than B-GH, DRv11 − TR measures how much of what the deep path provided the Mamba blocks recover.

## EXPLORATORY (validation only)

Difference of differences seen vs unseen shapes relative to B-GH for DRv11 and TR (as in the seen- vs unseen-shape
experiment, pre-registration `d46b279`), next to MD8 and MDv8-16-32 (`scripts/seen_unseen.py`).

## AMENDMENT to d46b279 (seen- vs unseen-shape experiment)

Moved to night 6b. The difference of differences for MD8-GH and MDv8-16-32-GH uses **B-GH seeds 0-2**, as
pre-registered; B-GH seeds 3-4 serve the DRv11 confirmation and are only descriptive there.

## VALIDATION CALIBRATION

Extended to all GH configurations (B, MD8, MDv8-16-32, DRv11, TR); descriptive, **no architectural conclusion**.

## AMENDMENT to d46b279 (MDv64)

The confirmatory test of MDv64 (content unchanged) is postponed: MDv64 seeds 3-7 trained on the night of Friday
2026-10-16, test pass on Saturday 2026-10-17 in the morning. If a run fails: it is repeated identically once;
otherwise it is declared missing.

## Operational rules (fixed with this pre-registration)

- Control runs (17 epochs, seed 0, grouped-holdout, DRv11 and TR, `~/runs/control-2026-10-15b`): fault criteria only —
  no NaN / Inf; training loss decreasing (mean of the last 3 epochs < mean of the first 3). The ratio of the final val
  L1 to the B control run of 2026-10-12 is reported (different split, declared) but is **not** a criterion: these
  networks may legitimately be worse than B.
- If the estimated queue duration exceeds 9.5 h, runs are cut in this order: MDv8-16-32-GH s2, s1, s0, then MD8-GH s2,
  s1, s0. The core (DRv11, B-GH, TR, seeds 0-4) is never cut.
- Runs failed / interrupted are reported with the reason; no run is repeated or dropped because of its result.
- Test session (single pass, `scripts/test_session_night6b.sh --go`): only the night-6b runs are evaluated on test
  (refused if any already has `eval_test.json`); then the confirmatory analysis (out `docs/results/test-confirm-dr`),
  the TR control and the calibration (out `docs/results/seen-unseen-test`).
