# Pre-registration — night 8b: Conv64, a plain residual conv block at 64x64 (EXPLORATORY, validation only)

Decided on 2026-10-17 at 03:10, **after seeing the results of night 8a** (MDv64-local, MDv64-dec, MDv16; commit
`f99f3d8`, docs/lab-notebook.md entry "Night 8a run check (15/15) and interim ablation analysis"), and **before any
run** of the new model (including the control run). Implementation: commit `feb7162` (`mdf/ablation.py`,
`ConvResBlock64`, variant `Conv`; notebook entry "Night 8b: Conv64 variant"). This file is committed on its own, before
the control run and before the launch of night 8b. Lab-notebook entry of the same date: "PRE-REGISTRATION: night 8b".
The rules of docs/preregistration-night8.md (with its amendments) are unchanged; this is a separate, additional
experiment.

**Exploratory experiment, validation set only. The test set is never read** for this model (no script of this
experiment evaluates on test; no test pass is planned).

## Motivation

Night 8a: MDv64-local (the MDv64 block without long-range propagation, i.e. in_proj, 3x3 depthwise conv + SiLU,
out_norm, SiLU gating, out_proj) keeps a large share of the rel_vy gain of MDv64 (MDv64-local − B −0.72 [−1.13, −0.34];
MDv64-local − MDv64 +0.08 [−0.13, +0.31], not determined). Question: is **any** extra local processing at 64x64 enough,
or are the **gating and normalisation** of the block needed?

## Model

| label | train.py arguments | block |
|---|---|---|
| `Conv64` | `--variant Conv --level 64` | y = x + Conv3x3(12 -> 32)(LeakyReLU(0.2)(Conv3x3(32 -> 12)(x))), padding 1, bias; no normalisation, no gating; last conv zero-initialised (network = B at init); after layer1 (32 channels, 64x64), as MDv64 |

- 6956 block parameters (MDv64-local uses 7232 of its 20800).
- Declared difference besides gating / normalisation: the receptive field of the block is **5x5** (two 3x3
  convolutions) against 3x3 for the depthwise conv of MDv64-local.

Protocol unchanged: split grouped (`splits/val_grouped_seed0.txt`), 133 epochs, Adam lr 6e-4, betas (0.5, 0.999), L1
loss, batch 10, final epoch. **Seeds 0-4.** Runs in `~/runs/night8b/`, each evaluated on `val` after training
(`run_queue.py`).

References (existing runs, no new run): **B** = `~/runs/night1/ep133_s0-2` + `~/runs/night3/B_s3-7`; **MDv64** =
`~/runs/night5/MDv64_s0-2` + `~/runs/night7/MDv64_s3-7`; **MDv64-local** = `~/runs/night8a/MDv64-local_s0-4`.

## Readings

- **Main reading:** paired comparisons on seeds 0-4 of every configuration, two-level interval as in night 8a.
- **Secondary reading:** B and MDv64 with seeds 0-7 (MDv64-local exists only for seeds 0-4).

## Metrics

- **Primary:** rel_vy (original definition: mean over samples of the per-sample ratio), with its aggregate and median
  versions.
- **Descriptive:** rel_all, rel_p; vy MAE in the wall band (T3); divergence error (T4); verification 5a
  (`train_fit.py`).
- Uncertainty: two-level bootstrap (seeds + profiles, 10 000 repetitions, seed 0) as `scripts/final_analysis.py`; the
  one-level (profiles) interval is reported as well.

## Comparisons

- Conv64 − B;
- Conv64 − MDv64-local;
- Conv64 − MDv64.

No correction for multiple comparisons: exploratory, every comparison reported with its estimate and interval.

## Interpretation rules (fixed now)

- If **Conv64 − B is similar to MDv64-local − B** on rel_vy, any extra local processing at 64x64 is enough.
- If **Conv64 is close to B and worse than MDv64-local**, the gating and normalisation of the block matter.
- "Similar", "close" and "worse" are read from the two-level intervals. **With 5 seeds an interval that includes zero
  does NOT demonstrate equality.** Operationally:
  - "worse than MDv64-local" only if the interval of Conv64 − MDv64-local lies above 0;
  - "better than B" only if the interval of Conv64 − B lies below 0;
  - "similar" or "close" is never stated from an interval that includes 0 alone. Such a result is reported as
    "not determined", with the estimate and interval (for example, Conv64 better than B with Conv64 − MDv64-local
    including 0 is compatible with the first reading but does not demonstrate it).
- The 5x5 vs 3x3 receptive field is a confound in the direction of Conv64 and is mentioned in every reading.

## Runs and reporting

- All runs are reported. **No seed is added after seeing the results.**
- A failed run (NaN / Inf, crash) is repeated identically once (re-running the queue), otherwise declared missing.
- Runs not done because of the queue cutoff (12:00 of 2026-10-17) are declared missing; the analysis is run on what
  exists, with the seed counts printed.
- Control run (17 epochs, seed 0, `~/runs/control-2026-10-17`): fault check only (no NaN / Inf, train L1 decreasing);
  its ratio to B is reported, not a criterion. Not part of the analysis.
