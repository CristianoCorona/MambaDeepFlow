# Pre-registration — night 9: ladder of one-ingredient ablations of MDv64-local (EXPLORATORY, validation only)

Decided on 2026-10-17 at 09:29, **after seeing the results of nights 8a and 8b** (MDv64-local, MDv64-dec, MDv16:
commit `f99f3d8`; Conv64: commit `1188c12`), and **before any run** of the new models (including the control runs).
Implementation: commit `1e3a3a8` (`mdf/ablation.py`, `SS2DVMambaLocalLadder`; docs/lab-notebook.md entry "Night 9:
ladder of one-ingredient ablations of MDv64-local"). This file is committed on its own, before the control runs and
before the launch. Lab-notebook entry of the same date: "PRE-REGISTRATION: night 9 (ladder)". The rules of
docs/preregistration-night8.md and docs/preregistration-night8b.md (with their amendments) are unchanged; this is a
separate, additional experiment. SA64 (pre-registered in docs/preregistration-night8.md) runs in the same queue right
after the ladder: see the amendment of the same date in that file.

**Exploratory experiment, validation set only. The test set is never read** for these models (no script of this
experiment evaluates on test; no test pass is planned).

## Motivation

MDv64-local (the MDv64 block without long-range propagation) keeps a large share of the rel_vy gain of MDv64
(MDv64-local − B −0.72 [−1.13, −0.34]); Conv64, a plain residual conv block with a similar number of parameters, does
not (Conv64 − MDv64-local +0.54 [+0.24, +0.89]). Conv64 and MDv64-local differ in several ingredients at once. Question:
**which ingredient of MDv64-local matters?** Each model below removes one ingredient from MDv64-local.

## Models (each differs from MDv64-local in one thing only; all at 64x64, after layer1, 32 channels)

| label | train.py arguments | difference from MDv64-local | block params total / used |
|---|---|---|---|
| `L-nogate` | `--variant L-nogate --level 64` | output not multiplied by SiLU(z) (z still computed, unused) | 20 800 / 5 184 |
| `L-nonorm` | `--variant L-nonorm --level 64` | out_norm (output LayerNorm of SS2D) -> identity | 20 672 / 7 104 |
| `L-noprenorm` | `--variant L-noprenorm --level 64` | block input LayerNorm -> identity: y = x + Mixer(x) | 20 736 / 7 168 |
| `L-noexp` | `--variant L-noexp --level 64` | no channel expansion: inner channels 32 instead of 64 (expand = 1) | 10 432 / 3 648 |
| `L-dense` | `--variant L-dense --level 64` | dense 3x3 conv (E -> E) instead of the depthwise one | 57 088 / 43 520 |

MDv64-local: 20 800 / 7 232. **L-noexp and L-dense also change the number of (used) parameters** (−3 584 and
+36 288); the other three change it only by the removed ingredient itself. All five: out_proj zero-initialised
(network = B at init), 3x3 receptive field, same initial weights as MDv64-local at a given seed wherever the removed
ingredient is not involved (L-noexp: different shapes, so no shared weights; tested in `tests/test_ladder64.py`).

Protocol unchanged: split grouped (`splits/val_grouped_seed0.txt`), 133 epochs, Adam lr 6e-4, betas (0.5, 0.999), L1
loss, batch 10, final epoch. **Seeds 0-4.** Runs in `~/runs/night9/`, each evaluated on `val` after training
(`run_queue.py`).

References (existing runs, no new run): **B** = `~/runs/night1/ep133_s0-2` + `~/runs/night3/B_s3-7`; **MDv64-local** =
`~/runs/night8a/MDv64-local_s0-4`; for the figure also MDv64 (`~/runs/night5` + `~/runs/night7`), MDv64-dec, MDv16
(`~/runs/night8a`), Conv64 (`~/runs/night8b`).

## Readings

- **Main reading:** paired comparisons on seeds 0-4 (references restricted to seeds 0-4), two-level interval as in
  nights 8a and 8b.
- **Secondary reading:** B with seeds 0-7 (MDv64-local exists only for seeds 0-4).

## Metrics

- **Primary:** rel_vy (original definition: mean over samples of the per-sample ratio sum|e_vy| / sum|t_vy|), with
  its aggregate and median versions.
- **Descriptive:** rel_all, rel_p; vy MAE in the wall band (T3); divergence error (T4); verification 5a
  (`train_fit.py`).
- Uncertainty: two-level bootstrap (seeds + profiles, 10 000 repetitions, seed 0) as `scripts/final_analysis.py`; the
  one-level (profiles) interval is reported as well.

## Comparisons

- every variant − MDv64-local (5 comparisons);
- every variant − B (5 comparisons).

No correction for multiple comparisons: exploratory, every comparison reported with its estimate and interval.

## Interpretation rules (fixed now)

- An ingredient **"matters"** if removing it makes rel_vy worse than MDv64-local with an interval that excludes zero
  (the interval of variant − MDv64-local lies entirely above 0).
- If that interval includes zero, the result is **"not demonstrated"** — NOT "does not matter". With 5 seeds an
  interval that includes zero does not demonstrate equality.
- An interval entirely below 0 (variant better than MDv64-local) is reported as such, with no further conclusion.
- Variant − B is read with the same rule ("better than B" only if the interval lies below 0; otherwise "not
  demonstrated").
- **L-noexp and L-dense** also change the number of parameters (L-noexp fewer, L-dense many more). Their reading
  always states this: a worse L-noexp may be due to fewer parameters as well as to the missing expansion; a result of
  L-dense mixes the dense/depthwise difference with a ~6x larger block.

## Runs and reporting

- All runs are reported. **No seed is added after seeing the results.**
- A failed run (NaN / Inf, crash) is repeated identically once (re-running the queue), otherwise declared missing.
- Single queue `queues/night9.txt`: the ladder first in seed order (s0 of the five variants, then s1, ..., s4), then
  SA64 s0-s4. Cutoff: no new run started after 13:00 of Sunday 2026-10-18. Runs not done by the cutoff are completed
  the following night (same queue and out root, finished runs skipped), no seed added; if a ladder run is still
  missing after that, it is declared missing and the analysis runs on what exists, with the seed counts printed.
- Control runs (17 epochs, seed 0, `~/runs/control-2026-10-17`): fault check only (no NaN / Inf, train L1
  decreasing); their ratio to B is reported, not a criterion. Not part of the analysis.
