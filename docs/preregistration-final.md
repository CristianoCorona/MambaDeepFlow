# Pre-registration — final architecture: B + MDv64-local, single confirmatory test pass (the last of the project)

Decided by the experimenter on **2026-10-18 at 13:49**, **after all the phase-2 validation results** (nights 8a, 8b, 9;
commits `f99f3d8`, `1188c12`, `71bf4f6`) and **before any run** of the new seeds. Lab-notebook entry of the same date:
"PRE-REGISTRATION: final architecture B + MDv64-local, single confirmatory test pass". This file is committed on its
own, before the night-10 queue, before any cost measurement and before any analysis script of this pass.

## Final architecture

**B + MDv64-local**: the base U-Net B (TurbNetG, expo 5) with, after layer1 (64x64, 32 channels), the MDv block
without propagation along the scan (`--variant MDv-local --level 64`, `mdf/ablation.py` `SS2DVMambaLocal`): pre-norm,
in_proj (C -> 2E), depthwise 3x3 conv + SiLU, per-channel D scaling, output LayerNorm, gating by SiLU(z), out_proj
(zero-initialised). It is a simplified Mamba block without the SSM, in line with MambaOut (Yu & Wang, 2024). Chosen on
validation: MDv64-local − B rel_vy −0.72 [−1.13, −0.34] and MDv64-local − MDv64 +0.08 [−0.13, +0.31] (seeds 0-4,
`f99f3d8`), at 73.5 instead of 30.6 training it/s.

**This configuration has never been evaluated on the test set.** No MDv64-local run has `eval_test.json`; the explore
cache holds no test inference of MDv64-local (checked on 2026-10-18: only `__val__` entries).

## The four previous test passes (this will be the fifth and last)

1. 2026-10-14, pre-registered final analysis (plan `26b6e7f` + addendum `65aa729`), results `ca067a4`: B, MD8, SA8,
   MD32, SA32, B-large (8 seeds), CBAM8 (3 seeds). No two-level interval of the main comparisons excluded 0.
2. 2026-10-15, MDv8 second pass (plan `26b6e7f`), results `98cbb84`: MDv8 − MD8 and MDv8 − B, no interval excluded 0.
3. 2026-10-16, night-6b session (pre-registration `a986ca1`), results `ae394ec`: DRv11 − B-GH on rel_all, no
   demonstrable difference; TR control; validation calibration (seen / unseen shapes) with GH_MD8 and GH_MDv8-16-32.
4. 2026-10-16, MDv64 confirmatory test (pre-registration `d46b279`, amended in `a986ca1`), results `817c558`:
   MDv64 − B rel_vy −1.17 [−2.47, −0.43], **CONFIRMED**.

(The one-off pipeline-equivalence check of the reproduced baseline at the start of the project, rule 5 of the
notebook, is not a comparison of architectures and is not counted.)

After this pass the test set is closed: no further test pass of any model is planned or allowed.

## Runs

- **MDv64-local, seeds 0-7.** Seeds 0-4: `~/runs/night8a/MDv64-local_s0-4` (night 8a, commit `635b69c`). Seeds 5-7:
  `~/runs/night10/MDv64-local_s5-7` (night 10, `queues/night10.txt`), **same configuration and same training code**:
  `--seed k --epochs 133 --variant MDv-local --level 64`, grouped split (`splits/val_grouped_seed0.txt`), Adam lr
  6e-4, betas (0.5, 0.999), L1 loss, batch 10, final epoch; evaluated on `val` after training (`run_queue.py`).
- Training code unchanged since night 8a (`git diff 635b69c HEAD`, checked on 2026-10-18): the changes to `mdf/` and
  `scripts/train.py` / `scripts/run_queue.py` only **add** the Conv64 and ladder variants (new classes after
  `SS2DVMambaLocal`, new branches in `blocks.make_block` and `models.normalize_cfg` taken only for the new variants),
  the usage string of `train.py`, and the absolute-date cutoff of `run_queue.py` (queue scheduling only). The
  `SS2DVMambaLocal` class and the paths taken by `MDv-local` are untouched. Empirical check: for seeds 0-7 the initial
  state dict of the MDv-local network built at `635b69c` and at HEAD has the same sha256; the forward of a fixed input
  differs by ~1e-10 relative, as two runs of the same commit do (GPU non-determinism).
- **Reference: B, seeds 0-7**, already evaluated on test in pass 1: `~/runs/night1/ep133_s0-2` + `~/runs/night3/B_s3-7`.
- **Descriptive reference: MDv64, seeds 0-7**, already evaluated on test in pass 4: `~/runs/night5/MDv64_s0-2` +
  `~/runs/night7/MDv64_s3-7`.
- **A failed night-10 run** (NaN / Inf, crash) is repeated identically once (re-running the queue); otherwise the
  test pass uses the seeds available, and the missing seeds are declared in the notebook before the pass.
- **No seed is added after seeing any result.** The validation results of seeds 5-7 are reported before the test pass
  (descriptive, `scripts/analyze_final_val.sh`) and **do not change this plan**.

## Primary hypothesis (the only decision)

On the test set (90 samples, 30 profiles), **rel_vy of MDv64-local is lower than that of B**.

- rel_vy: original definition, mean over samples of the per-sample ratio sum|e_vy| / sum|t_vy| (whole grid,
  normalised fields, as `scripts/evaluate.py` and `train/runTest.py`).
- Estimate: MDv64-local − B, difference of the means (per-sample seed means, then mean over samples), in percentage
  points.
- Interval: 95 %, two-level bootstrap (seeds of each configuration + profiles), **10 000 repetitions, bootstrap seed
  0**, as in the previous confirmatory tests (`explore.paired_bootstrap`, `scripts/final_analysis.py`).
- **Rule: "CONFIRMED" only if the upper bound of the interval is < 0; otherwise "NOT CONFIRMED"**, reported with the
  estimate and the interval. No other metric or interval decides. One comparison only: no correction needed.

## Descriptive (no decision)

- **MDv64-local − MDv64** (8 vs 8 seeds), rel_vy, with its two-level interval: whether, on the test set, the SSM
  (long-range propagation along the scans) adds anything over the local block. Read as on validation: an interval
  including 0 does not demonstrate equality.
- **MDv64-local − B**: rel_all, rel_p (and rel_vx); aggregate and median rel_vy (and rel_p); fluid MAE per channel;
  vy MAE in the wall band (T3); divergence error (T4).
- Everything else (SA64, the ladder L-*, Conv64, MDv64-dec, MDv16) **stays validation-only**: no test evaluation, no
  test analysis.

## Caveats stated now

- The hypothesis was generated on the validation set, and the test set has been opened four times before, including
  for MDv64 (confirmed on rel_vy). MDv64-local shares the block (and, at a given seed, the initial weights) with MDv64,
  so this test is not independent of pass 4: a confirmation here supports "the local block suffices on the test set
  as well", not a new, independent discovery.
- 8 seeds and 30 test profiles: the interval of MDv64-local − MDv64 is expected to be wide.

## Procedure

1. This file + notebook entry, committed (no run before).
2. Night 10: `queues/night10.txt` (MDv64-local s5, s6, s7), `scripts/launch_night10.sh`, out root `~/runs/night10`,
   cutoff 2026-10-19T08:00. Expected ~1 h.
3. Cost measurements (never during a training run), `docs/results/cost-final/`.
4. Next morning: run check of night 10 (notebook), `scripts/analyze_final_val.sh` (validation only, 8 seeds,
   descriptive) -> `docs/results/val-final-mdv64local/`, commit.
5. **One test session**: `scripts/test_session_final.sh --go` (refuses on a dirty tree, if HEAD does not descend from
   the commit of this file, if any MDv64-local run already has `eval_test.json`, or if a run is missing without being
   declared): test evaluation of MDv64-local s0-7 (batch 1, one run at a time), then the confirmatory analysis
   (CONFIRMED / NOT CONFIRMED printed first) and the descriptive comparisons -> `docs/results/test-final/`.
6. Results committed immediately, without re-runs; notebook entry.
