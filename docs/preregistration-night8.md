# Pre-registration — night 8: ablations of MDv64 at 64x64 (EXPLORATORY, validation only)

Decided on 2026-10-16, after the results of phase 1 (including the confirmatory test of MDv64, commit `817c558`), and
**before any run** (including the control runs) of the new models. Implementation: commit `b3a85c2`
(`mdf/ablation.py`, variants `MDv-local` and `MDv-dec`; docs/lab-notebook.md, entry "Night 8: ablation variants of
MDv64"). This file is committed on its own, before the control runs and before the launch of night 8. Lab-notebook
entry of the same date: "PRE-REGISTRATION: night 8".

**Exploratory experiment, validation set only. The test set is never read** for these models (no script of this
experiment evaluates on test; no test pass is planned).

## Motivation

MDv64 (VMamba SS2D block after layer1, 32 channels, 64x64) reduces rel_vy against B (validation and test). Three
explanations of the gain:

1. **global mixing** (long-range propagation along the scans);
2. **more local processing at high resolution** (the block's projections, depthwise 3x3 conv and gating);
3. **position in the network** (encoder, and the skip from out1 to the last decoder stage).

## Models (each differs from MDv64 in one thing only)

| label | train.py arguments | difference from MDv64 |
|---|---|---|
| `MDv64-local` | `--variant MDv-local --level 64` | scan output replaced by its D·x term in all 4 directions (no propagation); everything else identical, same initial weights per seed |
| `SA64` | `--variant SA --level 64` | the existing SA block (4 heads, SDPA, 2D sin/cos PE) instead of SS2D, same place (after layer1, 4 096 tokens) |
| `MDv64-dec` | `--variant MDv-dec --level 64` | the same MDv block in the decoder: on the output of dlayer2, before the concatenation with out1 that feeds dlayer1; same initial weights per seed |
| `MDv16` | `--variant MDv --level 16` | the MDv block at 16x16 (after layer2b, 64 channels) |

**All models**: split grouped (`splits/val_grouped_seed0.txt`, the split of B and MDv64), 133 epochs, Adam lr 6e-4,
betas (0.5, 0.999), L1 loss, batch 10, final epoch (last checkpoint), protocol unchanged. **Seeds 0-4.** Runs in
`~/runs/night8/`, each evaluated on `val` after training (`run_queue.py`).

References (existing runs, no new run): **B** = `~/runs/night1/ep133_s0-2` + `~/runs/night3/B_s3-7`; **MDv64** =
`~/runs/night5/MDv64_s0-2` + `~/runs/night7/MDv64_s3-7`; for the curve also MDv8 (`~/runs/night4`, s0-7) and MDv32
(`~/runs/night5`, s0-2).

## Readings

- **Main reading:** paired comparison on seeds 0-4 of every configuration (references restricted to seeds 0-4).
- **Secondary reading:** references with all 8 seeds (0-7), new variants with their seeds 0-4.

## Metrics

- **Primary:** rel_vy (original definition: mean over samples of the per-sample ratio sum|e_vy| / sum|t_vy|), with
  its aggregate (sum|e| / sum|t| over all samples) and median versions.
- **Descriptive:** rel_all, rel_p; vy MAE in the wall band (T3 of `final_analysis.py`); divergence error (T4);
  verification 5a (`train_fit.py`: L1 and rel_* on training in eval mode vs validation).
- Uncertainty: two-level bootstrap (seeds + profiles, 10 000 repetitions, seed 0), as in the main analysis
  (`scripts/final_analysis.py`, group `ablation64`); the one-level (profiles) interval is reported as well.

## Comparisons

- MDv64-local − MDv64 and MDv64-local − B;
- SA64 − MDv64 and SA64 − B;
- MDv64-dec − MDv64 and MDv64-dec − B;
- MDv16 − B (and MDv16 − MDv64, descriptive);
- curve of rel_vy against the block level for MDv8, MDv16, MDv32, MDv64 (descriptive; different seed counts per
  level, reported as such).

No correction for multiple comparisons: exploratory, every comparison reported with its estimate and interval.

## Interpretation rules (fixed now)

- If **MDv64-local** is close to B and worse than MDv64 on rel_vy, the gain needs long-range propagation; if it is
  close to MDv64, the local processing of the block is enough.
- If **SA64** is close to MDv64, the gain comes from global mixing at high resolution in general; if SA64 is close to
  B, it is specific to Mamba (or to its configuration).
- **MDv64-dec** tells whether the point of the network matters.
- "Close" and "worse" are read from the two-level intervals. **With 5 seeds an interval that includes zero does NOT
  demonstrate equality**: estimates and uncertainty are reported without forcing a conclusion; a conclusion of the
  kinds above is stated only when the intervals support it (e.g. "worse than MDv64" only if the interval of
  variant − MDv64 lies above 0), otherwise "not determined".

## Runs and reporting

- All runs are reported. **No seed is added after seeing the results.**
- A failed run (NaN / Inf, crash) is repeated identically once (re-running the queue), otherwise declared missing.
- Runs not done because of the queue cutoff are declared missing; the analysis is run on what exists
  (`scripts/analyze_night8.sh`), with the seed counts printed.
- Control runs (17 epochs, seed 0, `~/runs/control-2026-10-16`) only look for faults (no NaN / Inf, train L1
  decreasing); their ratio to B is reported, not a criterion. They are not part of the analysis.

## Amendment 2026-10-16 (after the control runs, before the launch): queue version and launch rule

Control runs (17 epochs, seed 0, `~/runs/control-2026-10-16`, commit `ccdd0f1`): all four pass (no NaN / Inf, train L1
decreasing). Estimated durations at 133 epochs (linear extrapolation of the per-epoch times + start-up; nominal 0.98 x,
worst case 1.05 x + 1 min; `queues/night8-*.txt` header): MDv64-local 18.5 / 21 min, **SA64 195.6 / 211 min**,
MDv64-dec 44.3 / 49 min, MDv16 15.3 / 18 min (nominal / worst case).

- `queues/night8-full.txt`: 20 runs, nominal 22.8 h, worst case 24.9 h.
- `queues/night8-cut.txt`: without MDv16 s4, s3, MDv64-dec s4, s3, SA64 s4, s3 (14 runs; MDv64-local s0-4, the
  others s0-2): nominal 14.3 h, worst case 15.7 h. This exceeds the ~13 h window: **not cut further** (decision left to
  the experimenter before the launch; any change is a new dated amendment, committed before the launch).
- Cutoff 12:00 (2026-10-17) for both. **T_full** (latest launch for which the full queue ends by 12:00 in the worst
  case) = 2026-10-16 **11:05**, already past: tonight the full queue cannot end by the cutoff.
- **Rule:** launch at or before T_full -> `scripts/launch_night8.sh full`; launch after T_full ->
  `scripts/launch_night8.sh cut`. The cut queue ends by 12:00 in the worst case if launched by **20:21** (nominal by
  21:42); with a later launch `run_queue.py` stops before the cutoff and the runs not started (the last of the queue:
  MDv64-local s4, s3, ...) are declared missing as stated above.

## Amendment 2026-10-16 (later the same afternoon, before any 133-epoch run): SA64 in a separate night

Decision of the experimenter, taken only on the durations measured by the control runs (SA64 ~3.3 h per run, 6.8 it/s;
previous amendment, commit `56d3792`) and before any long run. **This amendment supersedes the full / cut rule of the
previous amendment**; `queues/night8-full.txt`, `queues/night8-cut.txt` and `scripts/launch_night8.sh` stay in the
repository but are not used.

- **Night 8a** (Friday 2026-10-16 night): MDv64-local, MDv64-dec, MDv16, **seeds 0-4** (5 seeds as pre-registered),
  in seed order (s0 of the three, then s1, ...): `queues/night8a.txt`, `scripts/launch_night8a.sh`, out root
  `~/runs/night8a`, cutoff 12:00 of 2026-10-17. Estimate: nominal 6.5 h, worst case 7.3 h.
- **Night 9** (Saturday 2026-10-17 evening): SA64 **seeds 0-4**: `queues/night9.txt`, `scripts/launch_night9.sh`, out
  root `~/runs/night9`, cutoff 13:00 of Sunday 2026-10-18. Estimate: nominal 16.3 h, worst case 17.6 h. If part of it
  does not fit, the remaining seeds are completed the following night (same launcher, queue and out root: finished
  runs are skipped); **no seed is added**.
- Everything else unchanged (models, seeds, comparisons, metrics, readings, interpretation rules, reporting). The
  analysis (`scripts/analyze_night8.sh`) reads MDv64-local / MDv64-dec / MDv16 from `~/runs/night8a` and SA64 from
  `~/runs/night9`; until SA64 exists, the SA64 comparisons are absent and listed as missing.
