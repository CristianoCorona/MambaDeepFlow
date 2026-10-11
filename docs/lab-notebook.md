# Lab notebook

Chronological log of decisions, experiments and findings (newest entry last).
Rules: see "Lab notebook (standing rule)" in `CLAUDE.md`. Environment record: [`setup-log.md`](setup-log.md).

---

## 2026-10-09 — Environment setup and original-baseline smoke test (summary)

**Context / why.** Before writing any new code we needed a working local environment with
torch + mamba-ssm CUDA kernels, the 6.4k dataset, and evidence that the original
Deep-Flow-Prediction scripts run unmodified.

**Decision / action.** Full record in [`setup-log.md`](setup-log.md). In short:
- conda env `mdf`: Python 3.12.15, torch 2.10.0+cu128, causal-conv1d 1.7.0, mamba-ssm 2.3.2.post1,
  installed from prebuilt wheels (no source build; system GCC 16 is incompatible with nvcc 12.6).
- GPU: RTX 4060 Laptop, 8 GB. `scripts/check_env.py` confirms the fused CUDA kernels are used.
- Dataset: `data_6k.tar.gz`, 6400 train / 90 test `.npz` files, kept outside the repo, symlinked into `data/`.
- Original `train/runTrain.py` + `train/runTest.py`, unmodified, one run.

**Evidence.** From `setup-log.md` section 6 (commit `1fd9822`):
17 epochs x 600 batches = 10200 iterations, 2332035 trainable parameters, 2:01.81 total wall clock,
peak VRAM 442 MiB (nvidia-smi), peak RSS 7908624 kB. Test (90 samples): relative error
p / v / combined = 17.429916 % / 2.761411 % / 3.365578 %.

**Caveat.** Single run with a random seed (1162998757) drawn by the original script: the numbers are
indicative only and not suitable for model comparison.

**Open questions.** None for the setup itself; a controlled training/evaluation pipeline is needed (next entries).

---

## 2026-10-11 — Protocol decisions for all experiments

**Context / why.** The original script draws a random seed and re-shuffles the train/validation split with it,
so every run uses a different validation set and runs are not comparable. Model comparisons in this
project (baseline vs. Mamba-augmented U-Nets) need a fixed, declared protocol.

**Decisions.**
1. **Fixed train/validation split.** The split is derived from the *sorted* list of training files,
   shuffled once with a constant split seed (`SPLIT_SEED = 0`, Python `random.Random(0).shuffle`);
   the last 400 files are validation, the first 6000 training (same sizes as the original:
   `min(20 %, 400)` validation files). The validation file list is committed as `splits/val_seed0.txt`
   and its sha256 is recorded in every run's `config.json`.
   *Alternative considered:* keep the original per-seed split. Rejected: validation numbers would not be
   comparable across seeds and models, and different seeds would also train on different data.
2. **`--seed` controls only weight initialisation and batch order** (it seeds `random`, `numpy`,
   `torch`, `torch.cuda` and the DataLoader generator). It does not affect the split.
3. **Same script and protocol for baseline and every architecture**; only the model changes
   (`--model`, default `dfpnet` = the unmodified original `TurbNetG`).
4. **Protocol choices (e.g. number of iterations: 10k / 40k / 80k) are made on validation L1**, never on the test set.
5. **Test-set rule.** The 90-sample test set is used only (a) for the one-off equivalence check of the new
   pipeline against the original scripts and (b) for final results. No protocol or design choice is
   based on test numbers. `scripts/train.py` never touches the test set; there is no checkpoint selection
   (the final checkpoint is saved, as in the original).

**Evidence.** Original behaviour: `train/runTrain.py` calls `dataset.TurbDataset(prop, shuffle=1)`, which
shuffles the sorted file list with the run's random seed before the 6000/400 split (`train/dataset.py`).

**Open questions.** Number of iterations for the main experiments (calibration not yet run).

---

## 2026-10-11 — Controlled training/evaluation pipeline: design and declared deviations

**Context / why.** The original scripts cannot take an explicit seed, use a per-seed split, hold the whole
dataset in float64 (peak RSS ~7.9 GB of 14 GiB RAM) and do not log per-epoch losses or metadata.
CLAUDE.md forbids modifying `train/*.py`, so the new pipeline lives in new files and imports the
original model.

**Decision / action.** New files:

```
mdf/data.py          float32 loader, fixed split, normalisation (re-implemented), denormalisation
mdf/models.py        model registry; "dfpnet" = TurbNetG + weights_init imported from train/
mdf/runinfo.py       git hash / dirty flag, library + GPU versions, timers, peak RAM / VRAM
scripts/train.py     --seed --iterations --lr --batch-size --expo --out-dir --model
scripts/evaluate.py  evaluates a checkpoint on val or test; per-sample CSV + JSON summary
scripts/run_original.py   runs the unmodified train/runTrain.py with an explicit seed
scripts/verify_loader.py  new loader vs. original loader
scripts/compare_runs.py   compares loss curves / evaluation summaries across runs
splits/val_seed0.txt      the 400 validation file names
```

The original modules are imported by appending `train/` at the **end** of `sys.path`, so they can never
shadow other modules. Outputs go to `~/runs/<run-id>/` (outside the repo).

Kept identical to the original: pressure offset removal (`p -= mean(p)`, `p -= p * mask`), dimensionless
targets (`p / |v|^2`, `v / |v|` with `|v|` from the input free-stream channels), fixed maxima
(inputs 100 / 38.12 / 1.0, targets 4.65 / 2.04 / 2.37), Adam(lr, betas=(0.5, 0.999), weight_decay=0),
`utils.computeLR` decay (imported, applied per epoch with the same `if currLr < lrG` guard), `nn.L1Loss`,
`weights_init`, dropout 0, batch size 10, `epochs = int(iterations / len(trainLoader) + 0.5)`.

Declared deviations:

| #  | Item | Original | New |
|----|------|----------|-----|
| D1 | Train/val split | sorted list shuffled with the run seed; last 400 = validation | sorted list shuffled with constant `SPLIT_SEED = 0`; last 400 = validation |
| D2 | Data precision | whole dataset in float64, normalised in float64, cast to float32 per batch | each file cast to float32 on load into a preallocated float32 array; normalisation in float32 (pressure mean with a float64 scalar accumulator); differences measured in 6a |
| D3 | Batch order | sampler uses the global torch RNG | dedicated `torch.Generator` seeded with `--seed`; shuffle, drop_last, 0 workers kept |
| D4 | Seeding | random seed drawn and printed | `--seed` seeds random, numpy, torch, CUDA and the DataLoader; cuDNN left non-deterministic as in the original (`--deterministic` opt-in) |
| D5 | Validation pass | `eval()`, no `no_grad`, images every epoch | `eval()` + `torch.no_grad()`, no images (same values) |
| D6 | Loss logging | final-batch L1 printed; `saveL1 = False` | `metrics.csv` per epoch: `train_L1` = mean batch L1 over the epoch, `val_L1` = mean over the 40 validation batches (same formulas as the original) |
| D7 | Iterations | `epochs = int(it / 600 + 0.5)` | identical; requested and effective iterations both recorded (10k -> 10200) |
| D8 | Test evaluation | loads 10 unused train files, writes images; p / v / combined | loads only `data/test` with the same fixed maxima; adds vx, vy and fluid-only secondary metrics; per-sample CSV |
| D9 | Original baseline | no seed argument | `scripts/run_original.py` seeds `random` before running the unmodified `runTrain.py` (runpy); all outputs go to the run directory |

Metric definitions in `scripts/evaluate.py` (all on normalised fields, per sample, then averaged over samples,
exactly as `train/runTest.py`):
- primary (original, whole 128x128 grid incl. the airfoil interior):
  `rel_X = sum|o_X - t_X| / sum|t_X|` for X in p, vx, vy; `rel_v` = vx and vy pooled (original "v");
  `rel_all` = all three channels pooled (original "combined"); plus original `L1` and denormalised L1.
- secondary, labelled `fluid_*`: the same ratios restricted to fluid cells (input channel 2 == 0;
  the channel is 1 inside the airfoil, where all targets are 0 after offset removal).

Verification plan:
- **6a** new vs. original loader on a sample of files: max abs / rel difference per channel.
- **6b** the same `--seed` twice (10k iterations): difference between loss curves; no bitwise determinism assumed.
- **6c** 3 runs of `scripts/train.py` vs. 3 runs of the original `runTrain.py` (seeds 0, 1, 2, 10k iterations,
  run sequentially), all evaluated with `scripts/evaluate.py` on the test set (validation cannot be used here:
  the original's validation set changes with the seed and overlaps our training set).
- **6d** `git diff --stat master -- train/` must be empty.
- An independent code review (fresh-context agent) of the new code against the original, limited to
  correctness-relevant differences, before the long runs.

**Open questions.** None yet; results in the following entries.

---

## 2026-10-11 — Verification 6a: float32 loader vs. original loader

**Context / why.** The new loader re-implements the original normalisation in float32 (D2); it must
produce the same normalised tensors within float32 rounding.

**Action.** `python scripts/verify_loader.py` (defaults: 200 training files sampled with sample seed 0,
all 90 test files; tolerance 1e-6 on max |new - original float64|). The original `TurbDataset`
(train mode with `shuffle=0`, and test mode) is run on a temporary directory of symlinks to the same files.
The script also checks all 6490 raw files for the mask convention.

**Evidence.** Max |new - original float64| per channel (relative to the channel's max |value| in brackets):

| Set   | vx_in    | vy_in    | mask | p                   | vx                  | vy                  |
|-------|----------|----------|------|---------------------|---------------------|---------------------|
| train | 6.67e-08 | 5.48e-08 | 0    | 1.59e-07 (1.83e-07) | 1.48e-07 (1.55e-07) | 1.09e-07 (1.18e-07) |
| test  | 8.38e-08 | 5.63e-08 | 0    | 2.18e-07 (2.57e-07) | 1.41e-07 (1.47e-07) | 1.08e-07 (1.26e-07) |

Against the original cast to float32 (what the original network sees) the max differences are
5.96e-08 to 2.38e-07, i.e. at most 2 float32 ulps for values below 1; a large fraction of elements
differ by one ulp (e.g. 2294028 of 3276800 training pressure values), as expected from rounding in a
different precision. Denormalisation (`mdf.data.denormalize` vs. `TurbDataset.denormalize`): identical (0.0).
Mask check: 6490 files, 0 violations: input channel 2 is exactly 0 (fluid) or 1 (airfoil interior), and
p, vx, vy are exactly 0 inside the airfoil (raw and after normalisation). This is the basis of the
`fluid_*` secondary metrics (mask == 0). Result: PASS.

**Open questions.** None. Note: the original scripts compute `v_norm` for denormalisation from the
*normalised* input channels (runTrain.py, runTest.py); `scripts/evaluate.py` reproduces this for
`L1_denorm` so that the number is comparable with the original "Denormalized error", but that metric is
therefore not in physical units.

---

## 2026-10-11 — Independent code review (fresh-context agent) and corrections

**Context / why.** Before the long runs, a fresh-context agent compared the new code with `train/*.py`,
limited to correctness (normalisation, loss, optimiser, LR decay, split, denormalisation, metrics,
model, seeding, `run_original.py`), and cross-checked the notebook numbers against `setup-log.md` and
a re-run of `scripts/verify_loader.py`.

**Result.** No correctness issue found; D1-D9 implemented as declared. Confirmed among others: same
normalisation order and maxima; `nn.L1Loss` mean; Adam created after `.cuda()` with betas (0.5, 0.999);
`computeLR` imported from `train/utils.py`, 17 epochs for 10k iterations (6e-4 for epochs 0-8, then
4.83e-4 ... 7.38e-5); split 6000/400 with no overlap, equal to `splits/val_seed0.txt`; model has
2332035 parameters; `run_original.py` derived seed for seed 0 = 3626764237, equal to the seed printed by
the unmodified `runTrain.py` in a mechanics test (scratch run, not part of 6c), and nothing written to `train/`.
All notebook numbers of the setup and 6a entries matched their sources.

**Minor findings, fixed in this commit:**
1. `mdf/data.py`: the split-file check was silently skipped if the file was missing; now raises.
2. `scripts/evaluate.py --split val` read the split file without verifying it; it now recomputes the split,
   compares it with the file and with `val_files_sha256` in the run's `config.json`.
3. `scripts/compare_runs.py groups`: with a single run per group the sample std is NaN and the verdict
   was misleading; now requires >= 2 runs per group.

**Corrections to earlier entries (not rewritten):**
- 6a entry: "at most 2 float32 ulps for values below 1" is wrong. Measured by the reviewer: inputs <= 1 ulp,
  vx/vy targets <= 4 ulps, pressure up to ~1.2e6 ulps at values near zero after mean subtraction (rounding
  of the subtraction dominates). The correct statement is the absolute bound: max |new - original cast to
  float32| <= 2.4e-7 for all channels. "2294028 of 3276800" is the number of pressure values that differ
  at all (1306013 of them by exactly 1 ulp).
- Pipeline-design entry: "peak RSS ~7.9 GB" should read ~7.5 GiB (7908624 kB).

**Open questions.** None.

---

## 2026-10-11 — Pre-registered compatibility criterion for verification 6c

**Context / why.** 6c compares 3 runs of `scripts/train.py` with 3 runs of the unmodified `train/runTrain.py`
(via `scripts/run_original.py`). The criterion is fixed here, before any 6c result exists, and will not be
changed after seeing the results.

**Design.**
- Group A (new): `scripts/train.py --seed {0,1,2} --iterations 10000` (the seed-0 run is the first 6b run).
- Group B (original): `scripts/run_original.py --seed {0,1,2}` (10k requested iterations, 10200 effective, as A).
- Runs executed sequentially (never in parallel), same machine, AC power, `performance` platform profile.
- All six checkpoints evaluated with `scripts/evaluate.py --split test` (90 samples). The test set is used
  here only because the original's validation set changes with the seed and overlaps our training set.
  No decision other than the compatibility verdict is taken from these numbers.

**Criterion.** For each primary metric m in {rel_p, rel_v, rel_all} (the original runTest.py
"p", "v", "combined" relative errors, mean over the 90 test samples):
compatible if |mean_A(m) - mean_B(m)| <= 2 * max(std_A(m), std_B(m)), where std is the sample standard
deviation across the 3 runs of a group (ddof = 1). The pipelines are called *compatible* only if all three
metrics pass; otherwise *not compatible*, and the cause must be investigated before any further experiment.
vx, vy, fluid-only metrics and L1 are reported for information only. Computed by
`python scripts/compare_runs.py groups --a <A runs> --b <B runs> --split test --k 2`.

**Limitations (declared in advance).** With n = 3 per group the standard deviations are themselves very
uncertain (a std estimated from 3 values can easily be off by a factor of 2), so this test can only exclude
gross differences (e.g. a normalisation or loss bug shifting the error by several run-to-run standard
deviations); it cannot establish equivalence. Group B has extra variance by design (its training set changes
with the seed, D1). The primary evidence for equivalence is verification 6a (identical tensors within
float32 rounding) plus the independent code review; 6c is a coarse end-to-end sanity check.

**Open questions.** None.

---

## 2026-10-11 — Verification 6b: same seed twice

**Context / why.** Measure run-to-run variation at a fixed `--seed` (bitwise determinism not assumed;
cuDNN is non-deterministic, as in the original).

**Action.** `python scripts/train.py --seed 0 --iterations 10000` twice (run IDs `new_s0_a`, `new_s0_b`,
sequential), then `python scripts/compare_runs.py curves`. Commits recorded in config.json: `c7ee905` (a) and
`706fcd2` (b; the only change in between is a notebook entry); working tree clean (`dirty: false`) for both.

**Evidence.**
- 17 epochs, 10200 iterations; LR schedule identical in all 17 epochs.
- train L1 (mean of batches): no epoch identical; max |a-b| 5.34e-4 (max rel 2.6 %, epoch 1);
  final 0.0066620 vs 0.0067779 (rel 1.7 %).
- val L1 (mean of 40 batches): no epoch identical; max |a-b| 1.67e-3 (max rel 15.6 %, epoch 5);
  final 0.0056389 vs 0.0057735 (rel 2.4 %).
- Validation metrics (`evaluate.py --split val`), combined relative error: 2.888 % vs 2.956 %; p: 16.35 % vs 16.60 %.
- Diagnosis: with `seed_everything(0)` twice, initial weights are identical and the first three batches
  are identical (first batch indices [2933, 4210, ...]); the same forward/backward on the GPU twice with
  identical input gives gradients differing by up to 2.9e-6. The divergence therefore comes from
  non-deterministic CUDA kernels (atomic accumulation in backward), amplified over 10200 steps, not from seeding.

**Conclusion.** A fixed `--seed` fixes initialisation and batch order but not the trajectory: at the end of
training, two runs with the same seed differ by roughly as much as runs with different seeds (cf. 6c,
std of combined test error across seeds ~0.08-0.17 percentage points). Consequence for the project: model
comparisons need several seeds and must be reported with dispersion; single-run differences of a few
percent are noise.

**Open questions.** Whether `--deterministic` (cudnn.deterministic) or `torch.use_deterministic_algorithms`
gives bitwise-reproducible runs and at what speed cost was not tested (bilinear upsampling backward may have
no deterministic CUDA implementation).

---

## 2026-10-11 — Verification 6c: new pipeline vs. original runTrain.py (3 + 3 runs) and 6d

**Action.** Sequential runs, AC power, `performance` platform profile, GPU idle at 40 C before the runs:
`scripts/train.py --seed {1,2}` (seed 0 = `new_s0_a` from 6b) and `scripts/run_original.py --seed {0,1,2}`
(derived seeds 3626764237, 3280387012, 242886303; each equal to the seed printed by the unmodified
`runTrain.py`). All at commit `706fcd2` (except `new_s0_a`, `c7ee905`), clean tree. Each checkpoint
evaluated with `scripts/evaluate.py --split test`; criterion as pre-registered in the previous entry:
`scripts/compare_runs.py groups --a new_s0_a new_s1 new_s2 --b orig_s0 orig_s1 orig_s2 --split test --k 2`.

**Evidence.** Test relative errors in %, mean (sample std) over 3 runs; per-run values in run order (seed 0, 1, 2):

| Metric        | New (A)          | Original (B)     | abs diff | 2 * max std | Verdict    |
|---------------|------------------|------------------|----------|-------------|------------|
| rel_p         | 17.869 (0.719)   | 17.714 (0.796)   | 0.155    | 1.591       | compatible |
| rel_v         | 2.867 (0.088)    | 2.919 (0.123)    | 0.052    | 0.247       | compatible |
| rel_all       | 3.446 (0.083)    | 3.519 (0.166)    | 0.073    | 0.332       | compatible |
| rel_vx (info) | 2.180 (0.062)    | 2.222 (0.146)    | 0.041    |             |            |
| rel_vy (info) | 8.430 (0.177)    | 8.458 (0.102)    | 0.027    |             |            |
| fluid_rel_all | 3.301 (0.040)    | 3.386 (0.167)    | 0.085    |             |            |
| L1 (info)     | 0.0069 (0.0002)  | 0.0070 (0.0004)  | 0.0002   |             |            |

Per run, rel_all: A 3.4756, 3.5100, 3.3530; B 3.6912, 3.5060, 3.3605. rel_p: A 18.6909, 17.5630, 17.3532;
B 18.5257, 17.6814, 16.9357. The single setup-log run of the original (random seed) had p / v / combined
17.43 / 2.76 / 3.37 %, inside the range of both groups.

**Verdict.** By the pre-registered criterion the two pipelines are **compatible** on rel_p, rel_v and rel_all.
This is a weak statement: n = 3 per group only excludes gross differences; the evidence for equivalence is
mainly 6a and the code review. Group B shows a slightly higher mean on 5 of 7 metrics, by far less than one
run-to-run std; not interpretable with n = 3.

**Resources (per run; time = wall clock).**
- New: total 1:44.45-1:45.17 (`/usr/bin/time`), in-script 102.2-102.9 s, of which data load 10.2-11.7 s,
  training loop 87.7-89.4 s, validation 2.5 s, checkpoint < 0.01 s; 114.1-116.3 it/s (training loop only,
  definition in `scripts/train.py`), 111.0-113.0 it/s including validation. Peak RSS 3909004-3909460 kB
  (`/usr/bin/time`; 3817-3818 MiB from `ru_maxrss` inside the script); peak VRAM 147 MiB allocated /
  188 MiB reserved (`torch.cuda`, excludes the CUDA context).
- Original (via `run_original.py`): total 2:02.78-2:05.92, runTrain.py 119.7-122.6 s; peak RSS 7736928-7938784 kB;
  peak VRAM 223 MiB allocated / 306 MiB reserved. Its training-loop throughput was not measured separately
  (no per-phase timers in the unmodified script; setup-log measured 106.5 it/s from log timestamps).
- The float32 loader halves peak RAM (~3.7 GiB vs ~7.4 GiB).

**6d.** `git diff --quiet master -- train/` exits 0 and `git diff --stat master -- train/` is empty: no file in
`train/` modified. `find train -newer docs/setup-log.md` (excluding `__pycache__`) lists nothing: the original
baseline runs wrote nothing into `train/`.

**Not verified.** Bitwise reproducibility (see 6b); the original's training-loop it/s; compatibility at
40k/80k iterations or with other `--lr`/`--batch-size` values (only defaults tested); the `--no-lr-decay` and
`--deterministic` options were not exercised in any run.

**Open questions.** Calibration of the number of iterations (10k / 40k / 80k) on validation L1: next step,
not started.

---

## 2026-10-11 — Correction to the 6b/6c entry

- "Group B shows a slightly higher mean on 5 of 7 metrics" is wrong: in the table B is higher on 6 of 7
  (all except rel_p, where A is higher). The conclusion (differences far below one run-to-run std,
  not interpretable with n = 3) is unchanged.
- "GPU idle at 40 C before the runs": that reading (40 C, P8, 12 MiB used) was taken before the 6b runs,
  not immediately before the 6c runs, which followed the 6b runs after a pause of a few minutes.
- Peak RSS of the original runs: 7736928-7938784 kB = 7.38-7.57 GiB (written as "~7.4 GiB").

---

## 2026-10-11 — Airfoil profiles in the training, validation and test files (read-only check)

**Context / why.** Before calibrating the training budget on validation, we need to know what the validation
set measures: generalisation to unseen airfoil shapes, or to unseen flow conditions on shapes seen in training.

**Action.** `data/dataGen.py` names each sample `<airfoil>_<int(fsX*100)>_<int(fsY*100)>.npz`, where `<airfoil>`
is the basename of the airfoil coordinate file; so the profile is `name.rsplit("_", 2)[0]` (6 training files have
underscores in the airfoil name: `davis_corrected`, `hybrid_1`). `data/shearAirfoils.py` would append a
`_<shear*1000>` suffix (950-1050) to sheared variants; no file name has that extra field, so the 6.4k set contains
no sheared variants identifiable by name. New read-only script `scripts/airfoil_overlap.py` (file names only; it
recomputes the fixed split and checks it against `splits/val_seed0.txt`; the split is not changed).

**Evidence.** `python scripts/airfoil_overlap.py`:

| Set              | Files | Distinct profiles | Files per profile (min / median / max) |
|------------------|-------|-------------------|----------------------------------------|
| train + val      | 6400  | 1456              | 1 / 4 / 12                             |
| train            | 6000  | 1448              | 1 / 4 / 12                             |
| val              | 400   | 341               | 1 / 1 / 4                              |
| test             | 90    | 30                | 3 / 3 / 3                              |

- Validation: 333 of 341 profiles also occur in train; **390 of 400 validation files share their profile with
  at least one training file** (only 8 profiles / 10 files are unseen: ag46c03, e855, goe590, naca65210, s6061,
  s8035, s9026, sd7034).
- Test: **0 of 30 profiles occur in train or val**; the test set is profile-disjoint from the 6.4k set.

**Consequence.** Validation (as in the original, which also splits by file) mostly measures interpolation to
new flow conditions (velocity, angle) on known shapes; test measures generalisation to unseen shapes. Validation
numbers are therefore lower than test numbers and may not reveal overfitting to shapes. The calibration below
still uses validation only (test-set rule); this is a declared limitation of it.

**Limitation of the check.** Equal names imply the same coordinate file; different names may still be
near-identical shapes (variants of one family), which a name-based check cannot detect.

**Open questions.** Whether a profile-disjoint validation split should be used for later model selection
(not changed now; to be decided).

---

## 2026-10-11 — Independent recomputation of 6c (ddof convention)

**Context / why.** Check the 6c numbers from the raw per-sample CSVs with a separate script, and make the
standard-deviation convention explicit.

**Action.** New `scripts/recheck_groups.py` (does not import `compare_runs.py`): recomputes each run's means from
`eval_test_samples.csv`, compares them with `eval_test.json`, then group means and std with ddof=1 and ddof=0.
`scripts/compare_runs.py groups` uses `np.std(ddof=1)` (sample std), and the pre-registered 6c criterion
specifies ddof=1.

**Evidence.** `python scripts/recheck_groups.py --a new_s0_a new_s1 new_s2 --b orig_s0 orig_s1 orig_s2 --split test --k 2`
(run dirs in `~/runs/tinfra/`). Per-run CSV means equal the JSON means within 4.9e-12. In %:

| Metric  | mean A  | std A (1 / 0)   | mean B  | std B (1 / 0)   | abs diff | 2*max std (1 / 0) |
|---------|---------|-----------------|---------|-----------------|----------|-------------------|
| rel_p   | 17.8690 | 0.7194 / 0.5874 | 17.7143 | 0.7955 / 0.6495 | 0.1548   | 1.5910 / 1.2990   |
| rel_vx  | 2.1804  | 0.0616 / 0.0503 | 2.2215  | 0.1455 / 0.1188 | 0.0411   | (info)            |
| rel_vy  | 8.4304  | 0.1774 / 0.1448 | 8.4577  | 0.1016 / 0.0829 | 0.0273   | (info)            |
| rel_v   | 2.8672  | 0.0875 / 0.0715 | 2.9194  | 0.1233 / 0.1007 | 0.0522   | 0.2466 / 0.2014   |
| rel_all | 3.4462  | 0.0825 / 0.0674 | 3.5192  | 0.1657 / 0.1353 | 0.0730   | 0.3315 / 0.2707   |

All values with ddof=1 equal the 6c entry. Verdict: compatible with ddof=1 (the registered convention) and
also with ddof=0; the verdict does not depend on the convention.

**Open questions.** None.
