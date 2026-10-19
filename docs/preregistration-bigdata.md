# Pre-registration - big-dataset experiment (exploratory, validation only)

Decided on **2026-10-19**, **after the test set was closed** (final test pass `4859a04`, tag `final-results-2026-10-18`)
and **before any run on the full data set**. Lab-notebook entry of the same date: "PRE-REGISTRATION: big-dataset
experiment". This file is committed on its own, after the preparation commit `8bbefbb` (split grouped-full, cloud
package, analysis script, all tried on fake data only) and before the cloud VM exists. `scripts/cloud/launch_bigdata.sh`
refuses to start unless HEAD descends from the commit that added this file and the file is unchanged since.

## Status

**Exploratory.** Validation only; **the test set is never read** (it is not even unpacked on the VM). No result of this
experiment is confirmatory, none changes the report's confirmed findings, and none is used to select anything that is
then tested.

## Question

Does the rel_vy gain of MDv64-local (and, budget permitting, MDv64) over B, observed on the reduced 6.4k set
(validation, 8 seeds: MDv64-local − B −0.667 [−0.977, −0.373], MDv64 − B −0.747 [−1.027, −0.496]; confirmed on test),
remain when the networks are trained on the full 53.8k set?

## Data and split

- Full set `data_full.tar.gz` (README of thunil/Deep-Flow-Prediction; 10 961 766 491 bytes). Used subdirectories:
  `reg` and `shear` (`MDF_FULL_SUBDIRS`, default `reg,shear`); `sup` is not used. **Declared fallback:** if names in
  `shear/` do not parse with `mdf.data.profile_full`, only `reg` is used (`MDF_FULL_SUBDIRS=reg`), stated in the
  notebook before the launch.
- Split mode `grouped-full` (`mdf/data.py`, commit `8bbefbb`): validation = every file of the **85 profiles** of the
  grouped validation (`splits/val_grouped_seed0.txt`), from all used subdirectories (a sheared airfoil belongs to its
  base profile); every file of the **30 test profiles** (`splits/test_profiles.txt`) excluded; training = all the rest.
  The validation and training file lists are fixed by these rules; their sha256 are recorded in every run and checked
  by every later step.
- Go / no-go checks before the launch (`scripts/cloud/inspect_full.py`, run by `smoke_test.sh`): all 85 validation
  profiles present; no test profile in training or validation; all names of the used subdirectories parse; peak RAM of
  the loader fits the VM. If one fails: no launch; back to the laptop.

## Configurations, seeds, protocol

| Label | train.py arguments (plus `--seed k --epochs 133 --split-mode grouped-full`) |
|---|---|
| B | (none) |
| MDv64-local | `--variant MDv-local --level 64` |
| MDv64 | `--variant MDv --level 64` |

- Seeds **0, 1, 2** for every configuration (same seeds across models).
- Protocol unchanged from the reduced-set runs: **133 epochs** (an epoch is now a pass over the full training part,
  i.e. ~8x more optimiser steps than on the 6.4k set), Adam lr 6e-4, betas (0.5, 0.999), L1 loss, batch 10, original LR
  decay, final-epoch checkpoint (no selection on validation), float32 loader, evaluation on validation right after
  training (`run_queue.py`).
- Code: the commit that contains this file (or a descendant whose only changes do not touch training, evaluation or
  the split; declared in the notebook). Each run records the commit, the GPU and the environment.

## Queue order and budget rule

`queues/bigdata.txt`, priority order: **B s0, MDv64-local s0, B s1, MDv64-local s1, B s2, MDv64-local s2, then MDv64
s0, s1, s2.**

- Before the launch, `scripts/cloud/smoke_test.sh` measures it/s and memory on the VM (short runs, not results) and
  `estimate_budget.py` estimates each 133-epoch run (x 1.15 margin) and the cumulative cost in queue order from the
  hourly price read in the console. **MDv64 runs are launched only if their estimate fits in the remaining budget**
  (50 USD minus the credit already spent minus a 5 USD reserve); a run that does not fit is not started.
- The launch cutoff (absolute date-time) is the earlier of the budget limit and the time the results are needed;
  `run_queue.py` starts no run whose estimated end is after it.
- During the queue: if the spent credit approaches the limit, the queue is stopped (`STOP` file; or the VM is stopped,
  losing the current run).
- A run that fails or is interrupted (e.g. VM preemption) may be repeated identically once if the budget allows;
  otherwise it is declared missing.
- **All runs are reported**; runs not done are declared with the reason (budget, cutoff, failure).

## Analysis (`scripts/bigdata_analysis.py`, commit `8bbefbb`; validation only)

- **Main metric: rel_vy** (mean over validation samples of the per-sample ratio, original definition).
- Descriptive: aggregate and median rel_vy; rel_all; rel_p; wall-band vy MAE; divergence error (wall band and the
  other fluid cells); verification 5a (train and validation L1 in eval mode, difference / SE_Welch), all as defined in
  the earlier analyses (`mdf/explore.py`, `scripts/train_fit.py`).
- Comparisons: **MDv64-local − B**, **MDv64 − B**, **MDv64-local − MDv64**; paired differences of per-sample seed
  means; 95 % interval from the **two-level bootstrap** (seeds of each configuration + validation profiles), 10 000
  repetitions, bootstrap seed 0; on the seeds common to both labels of a pair.
- Reference: the same comparisons on the reduced set with the same seeds 0-2 (B `night1/ep133_s0-2`, MDv64-local
  `night8a/MDv64-local_s0-2`, MDv64 `night5/MDv64_s0-2`; same 85 validation profiles, 402 files), next to the 8-seed
  reduced-set values above.
- Figure: IEEE one column, rel_vy of B and MDv64-local per seed and mean, reduced vs full set side by side.
- **Reading: qualitative.** The full-set effect is compared with the reduced-set one (same sign? similar size?); the
  interval is reported, but **no confirmatory conclusion** is drawn: three seeds, one validation set, decided after the
  test set was used. Wording: "on validation, with the full data set, the difference MDv64-local − B is ... (reduced
  set: ...)". An interval that includes 0 is "not determined", never "no effect".
- Not planned: any test-set evaluation, any other configuration, hyper-parameter changes, more seeds (unless the
  budget is left over after all nine runs: then a new notebook entry before running them).

## Known limits (stated in advance)

- Three seeds; the validation set of the full data (all samples of 85 profiles, incl. sheared shapes) differs from the
  402-file reduced validation, so absolute errors are not directly comparable between the two sets; differences
  between configurations within a set are.
- The full set has ~8x more samples, and with 133 epochs ~8x more optimiser steps: data and compute grow together.
- Throughput and therefore the run time on the L4 are only known after the short check; the number of MDv64 runs
  depends on the budget, not on any result.
