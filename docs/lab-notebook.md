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

---

## 2026-10-11 — Pre-registered criterion for training-budget calibration

**Context / why.** The number of training epochs for all further experiments is chosen here, on validation only.
Candidates: 17 epochs (10k requested iterations, 10200 effective), 67 epochs (40k, 40200) and 133 epochs
(80k, 79800); from now on the protocol is named by epochs. The LR decay (`computeLR`) depends on the total number
of epochs, so a shorter run is not a prefix of a longer one: each budget needs its own runs. This entry is
written before any 67- or 133-epoch run exists and will not be changed afterwards.

**Runs.**
- 17 epochs: existing `new_s0_a` (seed 0) and `new_s1` (seed 1) in `~/runs/tinfra/`.
- 67 and 133 epochs: seeds 0 and 1, `scripts/train.py --seed S --iterations {40000,80000}` (unchanged code path;
  `--iterations` gives 67 / 133 epochs with the existing rule), launched sequentially, one process at a time,
  by `scripts/run_calibration.sh` (refuses a dirty tree; logs AC state, `platform_profile`, available RAM, swap
  used and GPU state before each run; each run followed by `scripts/evaluate.py --split val`). Run dirs
  `~/runs/calib/ep{67,133}_s{0,1}`.

**Metric.** Every run is evaluated on the fixed validation split (`evaluate.py --split val`, 400 samples).
Primary metric: val `rel_all` (original "combined" definition, whole grid). Secondary, reported but not part of
the criterion: `rel_p`, `rel_vx`, `rel_vy` and the `fluid_*` versions.

**Reference noise.** sigma_ref = sample std (ddof=1) of val rel_all over the four existing 17-epoch runs
(`new_s0_a`, `new_s0_b`, `new_s1`, `new_s2`). Measured before the calibration runs (`new_s1`, `new_s2` evaluated
on val now, at commit `c623758`): val rel_all = 2.8884, 2.9564, 3.0440, 2.8586 %, mean 2.9368 %,
**sigma_ref = 0.0823 percentage points**.

**Decision rule.** With means over seeds 0 and 1 of val rel_all:
Delta1 = mean(17 ep.) - mean(67 ep.), Delta2 = mean(67 ep.) - mean(133 ep.).
If Delta1 < sigma_ref choose 17 epochs; otherwise, if Delta2 < sigma_ref choose 67 epochs; otherwise 133 epochs.
A negative Delta counts as "less than sigma_ref". No compute budget enters the rule; the cost is reported and
decided separately.

**Overfitting check (reported, does not change the rule).** For each run of the chosen budget, with
k = max(3, round(0.15 * epochs)) (17 -> 3, 67 -> 10, 133 -> 20) and moving averages of window k over epochs:
flag "possible overfitting" if (a) the mean val_L1 over the last k epochs exceeds by more than 5 % the minimum of
the val_L1 moving average over the whole run, AND (b) the mean train_L1 over the last k epochs is lower than the
mean over the k epochs before. The 5 % threshold is about twice the final val_L1 difference between two runs with
the same seed in 6b (2.4 %). Also reported per run: the raw ratio final val_L1 / minimum of the val_L1 moving
average, plus a visual check of the curves.

**Test set.** Not used in this phase.

**Limitation (declared in advance).** 390 of 400 validation files share their airfoil profile with training files
(entry "Airfoil profiles ..."): validation mostly measures generalisation to new flow conditions on known shapes.
With 2 seeds per budget the means carry noise of the order of sigma_ref / sqrt(2).

**Open questions.** None before the runs.

---

## 2026-10-11 — Training-budget calibration: results (rule outcome: 133 epochs)

**Context / why.** Application of the pre-registered criterion (previous entry, commit `06f4bc1`).

**Action.** `nohup scripts/run_calibration.sh` (runs `ep67_s0`, `ep67_s1`, `ep133_s0`, `ep133_s1` in
`~/runs/calib/`, sequential, 14:44-15:21 UTC, all at commit `06f4bc1`, clean tree, val split sha256 `7f6bc466...`).
Machine state logged before each run: AC online, `platform_profile` performance, available RAM 10808-11123 MB,
swap used 514 MB; GPU 43 C / P3 before the first run, 58 C / P0 before the later ones (runs back to back).
Summary: `python scripts/calibration_summary.py` (writes `docs/results/calibration.csv`,
`docs/figures/calibration.png`). Test set not used.

**Evidence.** Per run, val, %:

| Run      | Epochs | Seed | rel_all | rel_p  | rel_vx | rel_vy | it/s  | Wall (time -v) |
|----------|--------|------|---------|--------|--------|--------|-------|----------------|
| new_s0_a | 17     | 0    | 2.8884  | 16.354 | 1.756  | 7.957  | 116.3 | 1:45.2         |
| new_s1   | 17     | 1    | 3.0440  | 16.194 | 1.935  | 7.701  | 115.1 | 1:44.7         |
| ep67_s0  | 67     | 0    | 1.9955  | 10.555 | 1.226  | 5.132  | 115.7 | 6:11.7         |
| ep67_s1  | 67     | 1    | 2.0483  | 10.681 | 1.251  | 5.557  | 116.0 | 6:08.3         |
| ep133_s0 | 133    | 0    | 1.8700  | 10.755 | 1.124  | 4.831  | 116.2 | 11:58.1        |
| ep133_s1 | 133    | 1    | 1.8303  | 9.631  | 1.113  | 4.786  | 116.4 | 11:56.8        |

(`new_s0_b` 2.9564 and `new_s2` 2.8586 enter only sigma_ref.) Per budget, mean over seeds 0 and 1 (|s0 - s1|):

| Epochs | rel_all         | rel_p            | rel_vx          | rel_vy          | fluid_rel_all |
|--------|-----------------|------------------|-----------------|-----------------|---------------|
| 17     | 2.9662 (0.1556) | 16.2741 (0.1594) | 1.8455 (0.1790) | 7.8288 (0.2556) | 2.8035        |
| 67     | 2.0219 (0.0528) | 10.6181 (0.1259) | 1.2385 (0.0258) | 5.3444 (0.4255) | 1.9266        |
| 133    | 1.8502 (0.0397) | 10.1930 (1.1237) | 1.1184 (0.0108) | 4.8084 (0.0443) | 1.7823        |

Resources per run (identical across budgets apart from time): 115.7 / 115.8 / 116.3 it/s (training loop only),
peak RSS 3.73 GiB (`/usr/bin/time`), peak VRAM 147 MiB allocated / 188 MiB reserved. Wall clock per run
1:44.9 / 6:10.0 / 11:57.5 (17 / 67 / 133 epochs); validation evaluation adds ~3 s (log timestamps).

**Rule.** sigma_ref = 0.0823; Delta1 = 2.9662 - 2.0219 = 0.9443 >= sigma_ref; Delta2 = 2.0219 - 1.8502 =
0.1717 >= sigma_ref. **Outcome: 133 epochs** (`--iterations 80000`, 79800 effective iterations).

**Overfitting check (pre-registered, not part of the rule).** No run flagged. For the chosen budget (k = 20):
mean val_L1 of the last 20 epochs / min of the 20-epoch moving average = 1.0000 (seed 0), 1.0005 (seed 1),
threshold 1.05; train_L1 still decreasing in both; raw ratio final val_L1 / min moving average = 1.0133 and
1.0002. The minimum of the moving average is at the end of the run (epoch index 132 and 131). Visually
(figure, left) the 133-epoch curves flatten after ~100 epochs without rising.

**Independent check.** A fresh-context agent recomputed all numbers from `eval_val_samples.csv`, `metrics.csv`,
`config.json` and the time logs (without the summary scripts) and applied the rule from the criterion text:
no discrepancy; same sigma_ref, means, Deltas, outcome and overfitting result; all 8 runs finished with the
expected epochs, seeds, clean tree and validation split.

**Cost estimate for the chosen protocol (scenarios, not measurements).** One baseline run at 133 epochs:
~12.0 min including validation evaluation (720 s). For 7 configurations x 3 seeds = 21 runs, sequential:

| Variant cost vs. baseline | All 21 runs at that cost | Baseline 3 runs at 1x + 18 variant runs |
|---------------------------|--------------------------|-----------------------------------------|
| 1x                        | 4.2 GPU-h                | 4.2 GPU-h                               |
| 2x                        | 8.4 GPU-h                | 7.8 GPU-h                               |
| 3x                        | 12.6 GPU-h               | 11.4 GPU-h                              |

**Additional considerations (not part of the rule).**
- Delta2 is ~2.1 sigma_ref; the noise on a difference of two 2-seed means is of order sigma_ref, so the
  67 -> 133 step clears the threshold, but not by a wide margin. The 67 -> 133 gain (0.17 points, ~8 %) is much
  smaller than the 17 -> 67 gain (0.94 points, ~32 %), at about twice the cost.
- At 133 epochs the seeds differ by 1.12 points in rel_p (10.75 vs 9.63), against 0.13 at 67 epochs; rel_p is
  secondary. Whether pressure error is more seed-sensitive at long budgets is not established with 2 seeds.
- Validation shares airfoil profiles with training (390 / 400 files); generalisation to unseen shapes (the test
  setting) may benefit less, or more, from longer training. Not measured (test set not used).

**Open questions.** Final choice of protocol given its cost (to be decided with the team; the rule gives 133
epochs). Whether 3 seeds per configuration suffice at this noise level.

---

## 2026-10-11 — Parallelism test: design

**Context / why.** Can two training processes share the GPU to cut total time for multi-seed experiments?
Run only after the calibration runs, so it cannot disturb their timings.

**Design.** New `scripts/run_parallel_test.sh`: two `scripts/train.py --iterations 10000` (17 epochs) processes at the
same time, seeds 0 and 1, run dirs `~/runs/par/par_s{0,1}`, each under `/usr/bin/time -v`; MemAvailable and
SwapFree sampled every second; both evaluated on val. Compared with the sequential runs `new_s0_a` / `new_s1`:
total wall time, it/s (training loop), combined peak RAM, val metrics (expected within noise, i.e. within the
17-epoch spread, sigma_ref = 0.0823 points). Guard: if MemAvailable < 9000 MB at the start, the test is skipped
and recorded as not run. Model-cost timings will still come from isolated runs, whatever the outcome.

---

## 2026-10-11 — Parallelism test: results

**Action.** `scripts/run_parallel_test.sh` at commit `ca1a4d9` (clean tree), 17:33 UTC, after the calibration runs.
Start state: AC online, `platform_profile` performance, MemAvailable 10922 MB (>= 9000, test run), swap used
514 MB, GPU 38 C / P3. Runs `~/runs/par/par_s0`, `par_s1` (17 epochs, 10200 iterations each).

**Evidence.** Parallel vs. the sequential runs with the same seeds (`new_s0_a`, `new_s1`):

| Quantity                          | Sequential (2 runs)       | Parallel (2 processes)    |
|-----------------------------------|---------------------------|---------------------------|
| Total wall clock for both runs    | 209.8 s (1:45.2 + 1:44.7) | 179.7 s                   |
| it/s per process (training loop)  | 116.3 / 115.1             | 62.8 / 62.8               |
| Combined training throughput      | ~115.7 it/s               | 125.6 it/s                |
| Training loop per run             | 87.7 / 88.6 s             | 162.3 / 162.5 s           |
| Peak RSS                          | 3.73 GiB per run          | 3.72 + 3.70 GiB (max RSS) |
| MemAvailable drop / swap growth   | n/a                       | 5076 MB / 24 MB           |
| Val rel_all (%) seed 0 / seed 1   | 2.8884 / 3.0440           | 2.9280 / 3.1150           |
| Val rel_p (%) seed 0 / seed 1     | 16.354 / 16.194           | 17.195 / 16.041           |

Total time -14 % (the data-loading phases, ~11 s each, also overlap); training-loop throughput +8.5 %.
Mean val rel_all 3.0215 (parallel) vs 2.9662 (sequential): difference 0.055, below sigma_ref (0.0823);
`par_s1` (3.115) is 0.07 above the highest of the four sequential 17-epoch runs. Within noise.

**Conclusion / recommendation.** Two processes share the GPU without errors or swapping, but the gain is small:
~8.5 % throughput, so for 133-epoch runs (where data loading is ~1.5 % of the time) the saving is expected to be
roughly 8-10 % (extrapolation, not measured), at the cost of double RAM (~7.4 GiB of 15 GiB) and per-process
timings that are not usable as model-cost measurements. Not worth using by default; acceptable for batches of
seed repetitions when RAM is free. Model-cost timings must still come from isolated sequential runs.

**Open questions.** Gain at 133 epochs and with heavier (Mamba) models not measured; with more GPU-bound models the
gain may be smaller.

---

## 2026-10-11 — Plan decisions: epoch-defined protocol, profile-disjoint validation, night queue, scope cuts

**Context / why.** The calibration (entry above) gave 133 epochs on the random validation split, which shares the
airfoil profile with training for 390 / 400 files, so it says little about generalisation to unseen shapes.
The PC is available for long runs only at night. Project deadline: 2026-10-18.

**Decisions (taken by the team).**
- **Protocol defined in epochs**: 133 epochs for final runs, 67 for pilots / comparisons. New `--epochs` option of
  `scripts/train.py`, which takes precedence over `--iterations` (kept to reproduce earlier runs). With the
  profile-disjoint split there are 599 batches per epoch (5998 training files), i.e. 79 667 / 40 133 effective
  iterations, 0.17 % fewer than in the calibration (79 800 / 40 200). `--iterations 80000` would instead give
  134 epochs (int(80000 / 599 + 0.5)); alternatives considered: keep `--iterations 80000` (134 epochs) or force
  exactly 400 validation files (would break the "at least 400" rule). `config.json` records epochs, batches per
  epoch and effective iterations.
- **D1 amended: profile-disjoint validation split** (default from now on, `--split-mode grouped`). Profile =
  `name.rsplit("_", 2)[0]`. With split seed 0 the sorted unique profiles are shuffled once and whole profiles are
  taken until validation holds at least 400 files; no profile in both sets. The random split stays reproducible
  (`--split-mode random`, `splits/val_seed0.txt`). Runs from before this entry have no `split_mode` in
  `config.json`; `scripts/evaluate.py` treats them as random.
- **Night queue** (`scripts/run_queue.py`): one run at a time, validation evaluation right after each run,
  completed runs skipped, interrupted runs redone from scratch, no run started if its estimated end is after a
  cutoff (default 07:00). No resuming in the middle of a run.
- **Scope.** No position study: final configurations B, B-large (parameter-matched), CBAM, SA at 8x8,
  M-D (Mamba SS2D at 8x8), 3 seeds each. M-E (Mamba at 64x64) only as a stretch goal. Cut: data-efficiency study,
  scan-order ablation, OOD on free-stream conditions.

**Open questions.** Whether 133 epochs still beats 67 on the profile-disjoint split (pre-registered criterion in a
later entry, before the night runs).

---

## 2026-10-11 — Profile-disjoint validation split: implementation and check

**Action.** `mdf/data.py`: `profile()`, `split_files_grouped()`, `split_file_path()`; `load_train_val` / `load_val`
take `split_mode` (default stays `random` at library level, so old callers are unchanged); `split_info` gains
`split_mode`, `n_train_profiles`, `n_val_profiles`. `scripts/train.py`: `--split-mode {grouped,random}` (default
grouped), `--epochs`; run IDs now contain the budget and split mode. `scripts/evaluate.py --split val` uses the
split recorded in the run's `config.json`. New `scripts/make_split.py` writes `splits/val_grouped_seed0.txt` and
checks overlaps.

**Evidence.** `python scripts/make_split.py --split-mode grouped --write`:
- 6400 files, 1456 profiles; train 5998 files / 1371 profiles; val 402 files / 85 profiles.
- Files in both sets: 0; train + val = all files: True; **profiles shared by train and val: 0**.
- val sha256 `3b26466ac7ccc1ad6db6f9a0f2cbaffee01d11784e7d537d5e139e5af085e2e6`,
  train sha256 `351af04002f93ec247b049967c82f3053fabe1881eb89855909d437fff1e5daf`.

`python scripts/make_split.py --split-mode random`: recomputed random split identical to `splits/val_seed0.txt`
(val sha256 `7f6bc466...` as in the calibration runs); it shares 333 of its 341 validation profiles with training.
`scripts/verify_loader.py`: OK (unchanged loader).

Validation loss curve in `metrics.csv` uses 40 batches of 10 (drop_last), i.e. 400 of the 402 files;
`scripts/evaluate.py` uses all 402.

**Open questions.** Profiles with different names may still be near-identical shapes (same limitation as
`scripts/airfoil_overlap.py`).

---

## 2026-10-11 — Night queue: design

**Context / why.** Runs happen only at night, unattended; a run must not run past the morning cutoff, and a crash
or power loss must not leave a half-finished run counted as done.

**Design.** `scripts/run_queue.py QUEUE_FILE --out-root DIR [--cutoff HH:MM]` (stdlib only; calls `train.py` and
`evaluate.py` with the same interpreter). Queue file: `<name> [est=13m] <train.py args>` per line. Per line:
`finished` + `eval_val.json` -> skipped; `finished` without evaluation -> evaluation only; `running` / `interrupted`
/ no config -> interrupted: directory moved to `<name>.interrupted-<UTC>` (kept as evidence, not deleted) and the run
redone from scratch. Estimated duration: maximum wall time (train + eval) of finished runs in the same out root with
the same arguments apart from `--seed` (`queue_timing.json`), otherwise the line's `est=`; if
`now + estimate > cutoff` (next occurrence of HH:MM after the queue starts) the queue **stops** (later runs are not
tried, to keep the order; team choice over "skip and try the next one"). Non-zero exit of training -> status
`interrupted` and the queue continues. `touch DIR/STOP` stops after the current run; SIGINT / SIGTERM stop the
current run (status interrupted) and the queue. Refuses to start on a dirty tree. `queue.log` with local
timestamps, start/end of each run, estimates, val metrics and machine state.

The night session runs under `systemd-inhibit` so the laptop does not suspend (see the launch entry).

---

## 2026-10-11 — Night queue: smoke test

**Action.** `scripts/run_queue.py` at commit `fe8b22c` (clean tree), out root `~/runs/smoke-queue`, 1-epoch runs
(`--epochs 1`, grouped split: 599 batches, 599 effective iterations, 402 validation samples, 85 profiles).

**Evidence (queue.log).**
- (a) Queue with `s1`, `s2`; while `s2` was training, the queue process and its `train.py` were killed with
  SIGKILL (simulated crash / power loss). `s1` status `finished`; `s2` left with status `running`.
- (b) Same queue restarted: "s1: already finished and evaluated - skipped"; "s2: found with status 'running' -
  treated as interrupted, moved to s2.interrupted-20261011T212121Z, redone from scratch"; `s2` then finished
  (18 s training, 3 s evaluation). The estimate for `s2` came from `s1` (0.4 min), not from `est=2m`.
- (c) Queue `s3` (no `est=`, same configuration as `s1`), `s4 est=10m --epochs 2`, `s5`, cutoff 23:24 at 23:21:49:
  `s3` ran (estimate 0.4 min from 2 finished runs); "s4: not started - estimated 10.0 min ... would end at
  23:32:10, after the cutoff 23:24:00; queue stopped"; `s5` not tried.
- (d) SIGINT to the queue during `s6`: training terminated (exit -15), `s6` status `interrupted`, "queue stopped by
  SIGINT", no `train.py` left running. `touch STOP` then restart: "STOP file found before s6: queue stopped",
  file renamed to `STOP.consumed-*`.

1-epoch val rel_all (grouped split): 9.61 / 9.62 / 10.87 % (seeds 0 / 1 / 2) - smoke only, not results.

---

## 2026-10-11 — Pre-registered criterion: protocol confirmation on the profile-disjoint split

**Context / why.** The 133-epoch protocol was calibrated on the random validation split (profile overlap). Written
and committed before the night runs.

**Criterion.** On the profile-disjoint validation split (`splits/val_grouped_seed0.txt`), if the mean of val
`rel_all` at 133 epochs (seeds 0 and 1) is lower than the mean at 67 epochs (seeds 0 and 1), the protocol stays at
133 epochs; otherwise this is reported and the decision is taken together by the team (no automatic change).
`rel_all` = mean over validation samples of the original "combined" relative error (`scripts/evaluate.py`,
`eval_val.json`). Seed 2 at 133 epochs is not part of the criterion.

---

## 2026-10-11 — Night 1: queue and launch

**Queue** (`queues/night1.txt`, order fixed): `ep133_s0`, `ep67_s0`, `ep133_s1`, `ep67_s1`, `ep133_s2`, all
`--epochs`, grouped split, out root `~/runs/night1`, cutoff 07:00. Expected ~51 min sequential (13 + 7 + 13 + 7 +
13 min from the calibration timings).

**Launch** (detached tmux session, TMPDIR as in CLAUDE.md, sleep / idle / lid switch inhibited):
`tmux new -d -s night "export TMPDIR=\$HOME/.cache/pip-tmp; systemd-inhibit --what=sleep:idle:handle-lid-switch
--why=night-queue <mdf python> scripts/run_queue.py queues/night1.txt --out-root ~/runs/night1 --cutoff 07:00"`.
Monitor: `tail -f ~/runs/night1/queue.log`. Stop after the current run: `touch ~/runs/night1/STOP`; stop now:
`tmux send-keys -t night C-c` (current run marked interrupted, redone at the next launch).

---

## 2026-10-12 — Night 1: results (criterion outcome: protocol stays at 133 epochs)

**Action.** Queue `queues/night1.txt` in tmux session `night` under `systemd-inhibit`, commit `3e69402` (clean tree),
2026-10-11 23:23 to 2026-10-12 00:11 local time; all 5 runs finished and evaluated, none interrupted, cutoff not
reached. Validation split: profile-disjoint (sha256 `3b26466a...`, 402 samples). Summary:
`python scripts/night_summary.py` (writes `docs/results/night1.csv`). Test set not used.

**Evidence.** Val, %; training wall time from `queue_timing.json` (evaluation 3 s each):
- `ep133_s0`: rel_all 2.4453, rel_p 12.6203, rel_vx 1.5783, rel_vy 5.9287; 12.01 min, 115.6 it/s
- `ep67_s0`: rel_all 2.5689, rel_p 13.7390, rel_vx 1.6461, rel_vy 6.3566; 6.13 min, 116.0 it/s
- `ep133_s1`: rel_all 2.4071, rel_p 12.4800, rel_vx 1.5337, rel_vy 6.0938; 11.94 min, 116.2 it/s
- `ep67_s1`: rel_all 2.5518, rel_p 13.3516, rel_vx 1.6492, rel_vy 6.2313; 6.11 min, 116.3 it/s
- `ep133_s2`: rel_all 2.4110, rel_p 12.1626, rel_vx 1.5420, rel_vy 5.9535; 11.94 min, 116.3 it/s

Mean of seeds 0 and 1 (|s0 - s1|): 67 epochs rel_all 2.5603 (0.0171), rel_p 13.5453 (0.3874), rel_vx 1.6476
(0.0031), rel_vy 6.2939 (0.1253); 133 epochs rel_all 2.4262 (0.0381), rel_p 12.5501 (0.1403), rel_vx 1.5560
(0.0445), rel_vy 6.0112 (0.1651). 133 epochs, seeds 0-2: rel_all 2.4212 (sd 0.0210), rel_p 12.4210 (sd 0.2345).

**Criterion (pre-registered).** 2.4262 (133) < 2.5603 (67): **protocol stays at 133 epochs.**

**Additional observations (not part of the criterion).**
- Error level, profile-disjoint vs random validation (calibration runs, mean of seeds 0 and 1): rel_all x1.27
  (67 ep) and x1.31 (133 ep); rel_p x1.28 / x1.23; rel_vx x1.33 / x1.39; rel_vy x1.18 / x1.25. The two sets
  contain different samples, so the ratio mixes profile novelty with sample difficulty.
- Gain 67 -> 133 epochs in rel_all: 0.134 points (5.2 %) on the profile-disjoint split vs 0.172 (8.5 %) on the
  random one; larger than both seed differences (0.017, 0.038).
- Seed spread of rel_p at 133 epochs is 0.14 (s0 vs s1), 0.46 over 3 seeds, smaller than the 1.12 seen in the
  calibration on the random split.

**Open questions.** Pressure analysis follows (next entries).

---

## 2026-10-12 — Secondary evaluation metrics: median and aggregate ratio

**Context / why.** The primary metrics (mean over samples of per-sample ratios, as `train/runTest.py`) weight every
sample equally, so a few samples with small sum|t| can dominate (notably rel_p). Two secondary views are added; the
primary metrics, their definitions and their role in decisions are unchanged.

**Change.** `scripts/evaluate.py`: `eval_<split>.json` gains `secondary` with `median_rel_<c>` /
`median_fluid_rel_<c>` (median over samples of the per-sample ratio) and `agg_rel_<c>` / `agg_fluid_rel_<c>`
(sum over all samples of sum|o-t| / sum over all samples of sum|t|, float64), for c in p, vx, vy, all (all = the
three channels pooled, as the original "combined"). The per-sample CSV gains the sums it is built from
(`abs_err_<c>`, `abs_tgt_<c>`, `fluid_abs_err_<c>`, `fluid_abs_tgt_<c>`). Recomputed on the existing runs
(validation only) in the next entry; previous outputs kept as `eval_val.v1.json` / `eval_val_samples.v1.csv`.

---

## 2026-10-12 — Secondary metrics recomputed on existing runs

**Action.** `scripts/reevaluate.sh` (new; keeps the previous outputs as `eval_val.v1.*`) with `scripts/evaluate.py`
at commit `38a8a88`, validation only, on `calib/ep{67,133}_s{0,1}`, `night1/*` (5), `tinfra/new_{s0_a,s0_b,s1,s2}`,
`par/par_s{0,1}` (15 runs). Each run is evaluated on the split recorded in its `config.json` (random for all runs
before night 1). Test set not used.

**Evidence.** Primary metrics unchanged: max |new - old| over all means and runs 7.5e-10 (CSV rounding).
Secondary metrics, mean of seeds 0 and 1, %:
- 67 ep, random val: median rel_all 1.4709, rel_p 9.0535; aggregate rel_all 2.0939, rel_p 9.9898 (fluid 1.9976 / 9.5095)
- 133 ep, random val: median rel_all 1.2554, rel_p 8.4883; aggregate rel_all 1.9220, rel_p 9.5384 (fluid 1.8532 / 9.1443)
- 67 ep, grouped val: median rel_all 1.5328, rel_p 10.2070; aggregate rel_all 2.6938, rel_p 12.9676 (fluid 2.5988 / 12.4473)
- 133 ep, grouped val: median rel_all 1.4003, rel_p 9.1963; aggregate rel_all 2.5570, rel_p 12.3622 (fluid 2.4937 / 12.0080)

**Observations.** 133 < 67 epochs on every secondary metric, on both splits. Grouped / random ratio at 133 epochs:
1.12 for the median rel_all, 1.31 for the mean, 1.33 for the aggregate: the cost of unseen profiles sits mostly in
the tail of hard samples, not in the typical sample.

---

## 2026-10-12 — Pressure error: per-sample analysis (no new runs)

**Context / why.** At 133 epochs on the random split, the two seeds differed by 1.12 points in rel_p (calibration
entry). Question: is the rel_p level and its seed variability driven by a few samples?

**Action.** `python scripts/pressure_analysis.py` (new; reads only `eval_val_samples.csv`; writes
`docs/results/pressure_top10.csv`). Groups: calibration 133 ep (random val, seeds 0-1), night 1 133 ep (grouped,
seeds 0-2), night 1 67 ep (grouped, seeds 0-1).

**Evidence.**
- Distribution right-skewed: grouped 133 ep, per-sample rel_p median 9.2-9.3 %, 90th pct 20.6-24.0, 99th pct
  62-75, max 182-186 (mean 12.2-12.6). Random 133 ep: median 8.2-8.8, 99th pct 29-35, max 54-59.
- Top 10 samples by seed-mean rel_p (2.5 % of samples) carry 8.1 % (random) / 14.8 % (grouped 133) / 14.5 %
  (grouped 67) of the summed rel_p; without them the grouped 133 mean drops from ~12.4 to ~10.8.
- Grouped top 10: two kinds. (i) small sum|p target| (ranks 1-26 of 402: `hh02` x3, `ht05` x2, `goe314`,
  `sc20706`), absolute error near the median; (ii) large sum|p target| with large absolute error
  (`pfcm_6703_-2724` rel_p 182-186 % in all 5 grouped runs, `pfcm_7006_-2167`, `rae5214_8305_-3198`; high
  fsY / fsX, i.e. high angle of attack). `pfcm` and `rae5214` are profiles absent from training.
- rel_p vs sum|p target|: Spearman -0.21 to -0.30; by quintile of sum|p target| the mean rel_p is U-shaped
  (grouped 133: Q1 17.9, Q3 8.7, Q5 15.0); sum|p o-t| grows with sum|p target| (Spearman +0.75 to +0.77).
- Seed consistency: Spearman of per-sample rel_p between seeds 0.87 (random 133), 0.91-0.92 (grouped 133), 0.94
  (grouped 67); median per-sample |diff| 1.2-1.5 points.
- Seed variability: the top 10 samples by |diff| carry 37-64 % of the summed squared per-sample differences
  (48 % of the per-sample variance over 3 seeds, grouped 133). Effect on the difference of the means depends on
  the pair: calibration s0 vs s1 +1.12 -> +0.85 without them, s0 worse on 66 % of samples, median diff +0.77
  (a broad shift); grouped 133 s0 vs s1 +0.14 -> +0.04, 50 % of samples, median +0.006 (a few samples);
  grouped 133 sd of the 3 run means 0.23 -> 0.19 without the top 10.

**Conclusion.** The mean rel_p is inflated by a tail of hard samples (small pressure signal or high-angle,
unseen-profile cases), which are consistently hard across seeds. Per-sample noise between seeds is concentrated
in a few samples, but the 1.12-point calibration gap was a broad shift over most samples, not a few outliers.
Median and aggregate rel_p (previous entry) are less sensitive to the tail and are worth reporting alongside.

**Open questions.** Whether the high-angle `pfcm` / `rae5214` cases are physically unsteady (RANS targets
less reliable). Not examined.

---

## 2026-10-12 — Independent check of night 1; correction to the pressure-analysis entry

**Action.** Fresh-context agent, read-only, recomputed everything from the raw files (`ls data/train`, split file,
`config.json`, `queue_timing.json`, `eval_val*.json`, `eval_val_samples*.csv`, `queue.log`) with its own scripts,
without the repo's summary scripts or the `mdf` package; test data not read.

**Result.** Split: 402 / 5998 files, 85 / 1371 profiles, 0 shared profiles, both sha256 match, the rule re-derived
independently gives the same list, all night-1 runs use it. Night-1 per-run and per-budget numbers, the criterion
outcome (2.42621 < 2.56033: 133 epochs), the grouped / random ratios, the secondary metrics (max |diff| vs
`eval_val.json` 2e-11; old vs new primary 7.45e-10 over 15 runs), `docs/results/night1.csv` and
`docs/results/pressure_top10.csv`: no discrepancy.

**Corrections to "Pressure error: per-sample analysis"** (text not rewritten):
- `pfcm_6703_-2724`: rel_p 180-186 % in all 5 grouped runs (not 182-186; `ep67_s1` 180.23).
- The Spearman ranges (-0.21 to -0.30; +0.75 to +0.77) are on seed-mean per-sample values; per run they are
  -0.18 to -0.33 and +0.73 to +0.78.
- "High angle of attack" (from fsX / fsY in the file name) holds for `pfcm_6703` (-22.1 deg, 98.8th percentile of
  |angle| in validation) and `rae5214_8305_-3198` (-21.1 deg, 94.5th); `pfcm_7006_-2167` is at -17.2 deg (76th
  percentile), so it is not a high-angle case relative to the set.
- Grouped 133 seed-pair Spearman: 0.905-0.915 (stated as 0.91-0.92).

---

## 2026-10-12 — Global-mixing blocks: implementation, deviations D10-D13, change of plan

**Context / why.** B (DfpNet, 133 epochs, grouped split, 3 seeds) is done: val rel_all 2.421 +- 0.021 %.
The variants with a global-mixing block are implemented here; all are built by one factory so that training,
evaluation and the queue use exactly the same model.

**Implementation.**
- `mdf/models.py`: `build_from_cfg({variant, level, channels, mamba_impl})`, variant in {B, B-large, CBAM, SA, MD}
  (`identity` for tests only). B and B-large go through the unchanged `build_dfpnet` (original `TurbNetG` +
  `weights_init`): B is bit-for-bit the model of the earlier runs (2 332 035 params, same initial weights at the
  same seed, same outputs; unit tests 2-3). The other variants use `TurbNetGMixer`, a subclass of `TurbNetG` with
  the forward copied; the block output replaces the output of the encoder layer of the chosen level and feeds
  both the next layer and the skip connection. Level -> layer (channels with 32 base channels): 64 -> `layer1`
  (32), 32 -> `layer2` (64), 16 -> `layer2b` (64), 8 -> `layer3` (128); default level 8.
- **Initialisation.** `TurbNetGMixer.__init__` applies `weights_init` to the skeleton *before* the block exists,
  then creates the block. So `weights_init` never touches a block (unit test 8), and with the same `--seed` the
  skeleton gets exactly the same initial weights as B (the block draws from the RNG afterwards). Blocks keep
  PyTorch defaults, Mamba's internal init and the zero output projections.
- **Base channels as an integer** (`--channels`, B-large = 33). The original constructor only takes an exponent
  (`int(2**e + 0.5)` channels), so internally `e = log2(channels)` is passed and checked to give exactly
  `channels` (and `layer1` is checked to have 33 output channels). B-large: **2 479 755 params (+147 720 vs B)**.
- `mdf/blocks.py`:
  - **SA** `y = x + Wo(SDPA(qkv(LN(x) + PE)))`: LayerNorm over channels on `(B, HW, C)`; fixed 2D sin/cos PE
    (first C/2 channels row index, last C/2 column index; frequencies 1/10000^(2i/(C/2))), a non-persistent
    buffer (not a parameter, not in the checkpoint); qkv Linear C -> 3C; 4 heads; `F.scaled_dot_product_attention`;
    Wo Linear C -> C, weight and bias zero. No MLP, no dropout. Params: 66 304 at 8x8, 16 768 at 32x32.
  - **MD** (simplified SS2D) `y = x + sum_k unscan_k(Mamba(scan_k(LN(x))))`, k in rows, rows reversed, columns,
    columns reversed; one shared `mamba_ssm.Mamba(d_model=C, d_state=16, d_conv=4, expand=2)`, the 4 sequences
    stacked in the batch `(4B, HW, C)`, one call; outputs mapped back to 2D and summed. `out_proj.weight` zeroed
    (Mamba's out_proj has no bias with the default `bias=False`); no other Mamba init touched. Params: 116 736 at
    8x8, 32 768 at 32x32.
  - **CBAM** (Woo et al. 2018, multiplicative, no residual): channel attention (spatial avg- and max-pool, shared
    MLP Linear C -> max(C//16, 1) -> C with biases and ReLU, sum, sigmoid), then spatial attention (channel mean
    and max, Conv2d 2 -> 1, 7x7, padding 3, no bias, sigmoid). Choices: MLP biases and bias-free conv as in the
    official CBAM code; the BatchNorm of the official spatial branch is omitted (not in the specification).
    Params: 2 282 at 8x8. Not an identity at initialisation (multiplicative).
  - `mdf/mamba_ref.py` (`--mamba-impl ref`): the non-fused path of `Mamba.forward` with the module's own parameters,
    PyTorch `nn.Conv1d` and `selective_scan_ref` copied from mamba_ssm 2.3.2 (Apache-2.0, attribution in the file;
    copied because `mamba_ssm.ops.selective_scan_interface` imports the CUDA extension at load time). Runs on
    CPU tensors. Limitation: constructing the block still imports `mamba_ssm.Mamba`, so a machine without the
    mamba_ssm wheel cannot use it.
- `scripts/train.py`: `--variant` (default B), `--level` (default 8), `--channels`, `--mamba-impl` (default cuda);
  `--model dfpnet` / `--expo` kept as legacy aliases of B. `config.json` adds `variant`, `level` (null for B /
  B-large), `channels`, `mamba_impl`, `block_params`, `params_B_reference`, `params_added_vs_B`,
  `evaluated_checkpoint`. Training loop unchanged.
- `scripts/evaluate.py` rebuilds the model from `config.json` (`models.model_cfg_from_run_config`); runs without a
  `variant` field (all earlier runs) are read as B with `2**expo` channels. `scripts/run_queue.py` unchanged: it
  calls `train.py` and `evaluate.py`.
- **Protocol unchanged for all variants**: Adam lr 6e-4, betas (0.5, 0.999), L1, original `computeLR`, batch 10,
  dropout 0, `--split-mode grouped` (split seed 0). **The evaluated model is the final-epoch model** (`model.pt`);
  validation is only for monitoring, no checkpoint selection (as stated on 2026-10-11, decision 5).

**Deviations from the planning report** (numbering continues D1-D9):

| #   | Item | Planning report | Now |
|-----|------|-----------------|-----|
| D10 | Mamba 2D block | SS2D as in VMamba | simplified SS2D: one Mamba module shared by the 4 scan directions (in_proj, gating, conv1d, SSM, out_proj all shared), outputs summed; no separate per-direction parameters, no extra output norm / gating of VMamba's SS2D |
| D11 | Parameter matching | B-large parameter-matched to the Mamba variant | partial: B-large (33 base channels, +147 720) is a "more parameters, no global mixing" control; added params per variant: MD8 +116 736, SA8 +66 304, CBAM8 +2 282, MD32 +32 768, SA32 +16 768. No matching at 32x32 |
| D12 | M-E (Mamba at 64x64) | stretch goal | replaced by a block-level micro-benchmark (time / memory vs number of tokens, 64 to 16 384) |
| D13 | Self-attention variant | not a main configuration | SA added as the natural global-mixing comparison for Mamba (same position, same residual form, same zero-initialised output projection); it was already listed among the final configurations on 2026-10-11, the reason is recorded here |

**Change of plan.** The 2026-10-11 entry said "no position study". Two 32x32 variants (MD32, SA32) are added at the end of
the queue as exploratory runs (where the sequence is 1024 tokens instead of 64).

**Inaccuracy in the planning report.** CBAM was planned on "skip connections / bottleneck"; at the 1x1 bottleneck
spatial attention is meaningless (one pixel). CBAM is placed at 8x8, the same point as the other variants.

**Declared limitation.** Same optimiser and learning rate for all variants (tuned for the U-Net): fair as a
protocol, but it may disadvantage SA and Mamba, which are often trained with other optimisers / LR schedules.

---

## 2026-10-12 — Unit tests of the blocks and the factory: all pass

**Action.** `pytest` 9.1.1 added to the env (`setup-log.md` section 7). `tests/test_blocks.py`, run with
`conda run -n mdf python -m pytest -s -q tests/test_blocks.py`: **43 passed, 6 skipped** (B / B-large at levels other
than 8: no block, level irrelevant), 13 s.

**Tests and evidence.**
1. Shapes: every variant at levels 64/32/16/8, batch 2, train mode -> `(2, 3, 128, 128)`, finite.
2. B from the factory vs original `TurbNetG`: 2 332 035 params both, same initial weights at seed 0, identical
   output (`torch.equal`, CPU) with the same weights. B-large: 33 channels, 2 479 755 params.
3. `nn.Identity` as the block at each level, original weights: identical output (eval and train mode).
4. SA and MD (levels 8 and 32) at initialisation with B's skeleton weights vs B: max |diff| 0 (SA, MD-ref on CPU),
   1.49e-8 (MD-cuda on GPU; the two GPU forwards are not bitwise identical), tolerance 1e-6.
5. `unscan_k(scan_k(x)) == x` for the 4 orders on a non-square 5x7 grid; the 4 orders differ pairwise; explicit
   order checked on a 2x3 grid.
6. `selective_scan_fn` (CUDA) vs reference (b=4, d=32, n=16, L=256, random inputs, z, D, delta bias, softplus):
   output max |diff| 2.9e-4; all gradients within rtol 6e-4 / atol 2e-3 (the float32 tolerances of the official
   mamba_ssm tests; largest absolute gradient diff 0.18 on dA with values up to 4.3e4). The vendored reference
   equals the upstream `selective_scan_ref` exactly. Full MD block cuda vs ref with out_proj randomised
   (std 0.05): mixing term max |diff| 2.4e-7 (8x8, C=128) and 4.8e-7 (32x32, C=64) for terms of order 1; input and
   parameter gradients within the same tolerances (atol scaled by the gradient magnitude), observed <= 3.2e-5.
7. After two Adam steps (lr 6e-4, betas (0.5, 0.999), L1) every parameter of the SA, MD, CBAM blocks (levels 8,
   32) has a non-zero gradient. At the first step only the output projection of SA / MD has a gradient (asserted):
   expected, since the zero projection blocks the gradient to everything before it.
8. `weights_init` does not touch the blocks: the block rebuilt from the same RNG state (seed -> skeleton +
   weights_init -> block) equals the block inside the model, and the skeleton equals B at the same seed;
   output projections are exactly zero. Negative control: applying `weights_init` to the MD / CBAM block does
   change it (conv1d / conv), so the test can detect it. (SA has no Conv / BatchNorm module.)
9. SA's PE is not a parameter and not in the state_dict. Block params vs estimates: SA 66 304 / 16 768 (~66k /
   ~17k), MD 116 736 / 32 768 (~117k / ~33k), CBAM 2 282 (~2k; 678 at 32x32).

`git diff --stat master -- train/` empty.

---

## 2026-10-12 — Pre-registered control runs (17 epochs): purpose and pass criterion

**Context / why.** Before the long queue, each configuration is trained once briefly to catch faults (divergence,
NaN, a block that breaks training). Written and committed before any control run.

**Runs.** 7 configurations, seed 0, 17 epochs (`--epochs 17`), profile-disjoint split, in this order:
B, MD8, SA8, CBAM8, B-large, MD32, SA32 (`queues/control-17ep.txt`, `scripts/run_queue.py`, out root
`~/runs/control-2026-10-12`, one run at a time, validation evaluation after each run). **Purpose: only to find
faults; the results are not used for comparisons or decisions.**

**Pass criterion per variant** (`scripts/control_gates.py`):
(a) no NaN / Inf in the loss (per-epoch train and val L1 means in `metrics.csv`; one NaN / Inf batch makes its
epoch mean NaN / Inf);
(b) training loss decreasing: mean train L1 of the last 3 epochs < mean of the first 3;
(c) final val L1 (`val_L1_mean_of_batches`, last epoch) <= 1.15 x the final val L1 of B at 17 epochs (B runs first).

The mean training it/s of each run (`it_per_s_train` in `config.json`) is the official measure of training cost
and is used to estimate the durations in the final queue. A variant that fails is diagnosed; only evident bugs
are fixed (recorded here), then re-run; if it still fails it is excluded from the queue.

---

## 2026-10-12 — Control runs (17 epochs): all 7 configurations pass

**Action.** `scripts/run_queue.py queues/control-17ep.txt --out-root ~/runs/control-2026-10-12 --cutoff 08:30` in a
detached tmux session under `systemd-inhibit`, commit `c370a1d` (clean tree), 23:29-23:45 local time, one run at a time,
no interruption. Gates: `python scripts/control_gates.py --root ~/runs/control-2026-10-12 --csv docs/results/control.csv`.
Machine state before B: AC online, `platform_profile` performance, GPU 42 C / P3. Test set not used.

**Evidence.** 17 epochs, 10 183 iterations each, seed 0. it/s = `it_per_s_train` (training loop only).

| Run     | Params    | Added vs B | it/s  | Train L1 first 3 / last 3 | Val L1 final | / B   | a | b | c | Result |
|---------|-----------|------------|-------|---------------------------|--------------|-------|---|---|---|--------|
| B       | 2 332 035 | 0          | 115.0 | 0.023897 / 0.006655       | 0.006324     | 1.000 | Y | Y | Y | pass   |
| MD8     | 2 448 771 | +116 736   | 92.8  | 0.023364 / 0.006425       | 0.006229     | 0.985 | Y | Y | Y | pass   |
| SA8     | 2 398 339 | +66 304    | 110.0 | 0.024270 / 0.006861       | 0.006499     | 1.028 | Y | Y | Y | pass   |
| CBAM8   | 2 334 317 | +2 282     | 110.0 | 0.024491 / 0.006526       | 0.006371     | 1.007 | Y | Y | Y | pass   |
| B-large | 2 479 755 | +147 720   | 86.2  | 0.023635 / 0.006588       | 0.006266     | 0.991 | Y | Y | Y | pass   |
| MD32    | 2 364 803 | +32 768    | 64.7  | 0.023670 / 0.006709       | 0.006349     | 1.004 | Y | Y | Y | pass   |
| SA32    | 2 348 803 | +16 768    | 55.7  | 0.023845 / 0.006626       | 0.006371     | 1.007 | Y | Y | Y | pass   |

Peak VRAM allocated (training): B 147, MD8 179, SA8 167, CBAM8 165, B-large 156, MD32 349, SA32 182 MiB.
B's it/s (115.0) matches the earlier B runs (115.6-116.3), so the factory adds no overhead to B.

As pre-registered, the val numbers are not used for comparisons (one seed, 17 epochs).

**Observation (cost, not part of the gates).** B-large (33 base channels) is the slowest of the 8x8 variants
(86.2 it/s, -25 % vs B for +6 % parameters), slower than MD8. A plausible cause is channel counts that are not
multiples of 8 (33, 66, 132, 264), which are less efficient for cuDNN kernels; not verified. Its training-time cost
is therefore not representative of its FLOPs (cost measurement in isolation pending).

---

## 2026-10-12 — Night 2: pre-registered queue, schedule and draft analysis plan

**Queue** (`queues/night2.txt`, order fixed; out root `~/runs/night2`; 133 epochs, grouped split; B not re-run, its
three 133-epoch runs are `~/runs/night1/ep133_s{0,1,2}`). All 7 configurations passed the gates, none excluded:
seed 0: MD8, SA8, B-large, CBAM8; seed 1: the same; seed 2: the same; then MD32 s0, SA32 s0, MD32 s1, SA32 s1,
MD32 s2, SA32 s2.

**Durations** estimated from the control runs: (training loop + validation of the 17-epoch run) x 133/17 + fixed
overhead (data loading, setup; 12-18 s) + validation evaluation (3-8 s): MD8 15.1, SA8 12.7, B-large 16.1, CBAM8 12.7,
MD32 21.4, SA32 24.6 min. The `est=` values in the queue file add ~10 % margin (17 / 14 / 18 / 14 / 24 / 27 min) and
are used only for the cutoff check. Total ~308 min. With a start at ~23:55 the expected ends are: 8x8 block of
seed 0 00:51, seed 1 01:48, seed 2 02:44; MD32 / SA32 s0 03:06 / 03:30, s1 03:52 / 04:16, s2 04:38 / 05:02.
Cutoff 08:30 (PC available until then): **all 18 runs are expected to finish tonight**. If not, the queue stops
before a run that would end after the cutoff and the rest resumes the next night (completed runs are skipped).

**Draft analysis plan (to be confirmed before the test set is opened).**
- Primary metric: `rel_all`; key secondary: `rel_p`; also reported: `rel_vx`, `rel_vy`.
- Primary summary: mean +- standard deviation (ddof=1) over the 3 seeds per configuration. Secondary: paired
  per-sample comparison with a bootstrap over airfoil profiles (resampling profiles, not samples, because samples of
  one profile are correlated). For pressure also the median and the aggregate ratio (secondary metrics of
  `evaluate.py`).
- Main comparisons (8x8): M-D vs B, M-D vs SA, M-D vs B-large. Exploratory: M-D32 vs M-D8, SA32 vs SA8,
  M-D32 vs SA32, CBAM vs B.
- With 3 seeds: no formal correction for multiple comparisons; results are descriptive.

**Declared limitation** (repeated from the implementation entry): same optimiser and LR for all variants.

**Launch** as night 1: `tmux new -d -s night2 "export TMPDIR=\$HOME/.cache/pip-tmp; systemd-inhibit
--what=sleep:idle:handle-lid-switch --why=night2-queue <mdf python> scripts/run_queue.py queues/night2.txt --out-root
~/runs/night2 --cutoff 08:30"`. Monitor: `tail -f ~/runs/night2/queue.log`; stop after the current run:
`touch ~/runs/night2/STOP`.

**Not yet done (never in parallel with a training run):** cost in isolation (`scripts/cost.py`) and block
micro-benchmark (`scripts/microbench.py`); both scripts are committed here but have not been run.

---

## 2026-10-13 — Night 2: results (validation only, descriptive)

**Action.** Queue `queues/night2.txt` in tmux session `night2` under `systemd-inhibit`, commit `ee7ee46` (clean tree),
2026-10-12 23:46 to 2026-10-13 04:52 local time; **all 18 runs finished and evaluated**, none interrupted, cutoff (08:30) not
reached. All runs: 133 epochs, grouped split (402 validation samples). Test set not used. Actual training wall times:
MD8 14.7, SA8 12.5-12.6, B-large 16.1-16.2, CBAM8 12.5-12.6, MD32 21.3-21.4, SA32 24.3 min (estimates 15.1 / 12.7 /
16.1 / 12.7 / 21.4 / 24.6).

**Evidence.** Val, %, mean (sample std, ddof=1) over seeds 0-2; B = `night1/ep133_s{0,1,2}` (commit `3e69402`; same
model, see the unit tests); median / aggregate rel_p = mean over seeds of the secondary metrics; it/s per seed.

| Config  | rel_all         | rel_p            | rel_vx          | rel_vy          | median rel_p | agg rel_p | it/s               |
|---------|-----------------|------------------|-----------------|-----------------|--------------|-----------|--------------------|
| B       | 2.4212 (0.0210) | 12.4210 (0.2345) | 1.5513 (0.0237) | 5.9920 (0.0890) | 9.239        | 12.308    | 115.6 / 116.2 / 116.3 |
| MD8     | 2.4318 (0.0500) | 12.0921 (0.4137) | 1.5722 (0.0354) | 5.8059 (0.1655) | 8.814        | 12.409    | 94.4 / 94.6 / 94.6 |
| SA8     | 2.4855 (0.0084) | 12.7703 (0.4147) | 1.5936 (0.0110) | 6.2296 (0.0696) | 9.407        | 12.544    | 110.9 / 109.9 / 110.2 |
| B-large | 2.4786 (0.0594) | 12.5396 (0.3307) | 1.5980 (0.0347) | 6.0051 (0.1194) | 9.089        | 12.660    | 85.7 / 85.9 / 85.8 |
| CBAM8   | 2.4963 (0.0386) | 12.5244 (0.1238) | 1.6120 (0.0300) | 6.0032 (0.2109) | 9.232        | 12.662    | 110.4 / 110.6 / 110.4 |
| MD32    | 2.4481 (0.0244) | 12.2080 (0.2093) | 1.5850 (0.0122) | 5.8070 (0.2099) | 9.026        | 12.401    | 64.3 / 64.6 / 64.5 |
| SA32    | 2.4850 (0.0630) | 12.4155 (0.4063) | 1.6118 (0.0560) | 5.9639 (0.1724) | 9.048        | 12.626    | 56.3 / 56.4 / 56.4 |

Per run (rel_all, seeds 0 / 1 / 2): MD8 2.4849 / 2.3856 / 2.4250; SA8 2.4833 / 2.4947 / 2.4784; B-large 2.4413 / 2.4474 /
2.5472; CBAM8 2.5363 / 2.4934 / 2.4593; MD32 2.4376 / 2.4760 / 2.4307; SA32 2.4123 / 2.5239 / 2.5187.

**Observations (descriptive; not the final analysis, which is pre-registered as a draft and will be confirmed
before the test set is opened).** On validation no variant has a lower mean rel_all than B; all means lie within
0.08 points of B (B 2.421, range 2.432-2.496), of the order of 1-4 seed standard deviations. MD8 and MD32 have the lowest
mean rel_p and rel_vy and the lowest median rel_p, but not the lowest aggregate rel_p. With 3 seeds these differences
are not established.

**Open questions.** Final analysis plan to be confirmed by the team; bootstrap over profiles not computed yet.

---

## 2026-10-13 — Cost in isolation

**Action.** `python scripts/cost.py` at commit `ee7ee46` (clean tree), after the queue had ended, GPU idle (no other
process). Output `docs/results/cost.csv`. FLOP counting and the analytic Mamba formula are documented in the script
docstring: the counter (`torch.utils.flop_counter`) counts matmul / conv / attention at 2 FLOPs per multiply-add;
elementwise ops are not counted for any model. For MD the block is counted analytically per sequence of length L
(C = d_model, E = 2C, N = 16, R = ceil(C/16), K = 4):
`2LC(2E) + 2LEK + 2LE(R+2N) + 2LRE + 2LEC + 7LEN + 3LE`, times 4 scan directions. The counter itself saw only the
four Mamba linear layers (MD8 0.0566 GFLOPs = exactly 4 x (in_proj + x_proj + dt_proj + out_proj)), not the
conv1d and the selective scan, which confirms the need for the analytic term.

**Evidence.** Latency: eval, no_grad, median of 100 after 20 warm-up; VRAM: `max_memory_allocated`, batch 10, 3
iterations of forward + backward + Adam step (excludes the CUDA context); it/s: control runs (17 epochs).

| Config  | Params    | Added vs B | GFLOPs fwd (b=1) | Block GFLOPs | Peak train VRAM b10 (MiB) | Latency b1 / b10 (ms) | Train it/s |
|---------|-----------|------------|------------------|--------------|---------------------------|-----------------------|------------|
| B       | 2 332 035 | 0          | 0.7833           | -            | 145.0                     | 0.82 / 2.93           | 115.0      |
| B-large | 2 479 755 | +147 720   | 0.8318           | -            | 152.8                     | 0.88 / 4.40           | 86.2       |
| CBAM8   | 2 334 317 | +2 282     | 0.7833           | 0.00002      | 164.0                     | 0.91 / 2.99           | 110.0      |
| SA8     | 2 398 339 | +66 304    | 0.7938           | 0.0105       | 165.0                     | 0.81 / 2.97           | 110.0      |
| MD8     | 2 448 771 | +116 736   | 0.8480           | 0.0647 (an.) | 177.4                     | 1.25 / 3.36           | 92.8       |
| SA32    | 2 348 803 | +16 768    | 1.0853           | 0.3020       | 178.2                     | 0.90 / 5.13           | 55.7       |
| MD32    | 2 364 803 | +32 768    | 1.0910           | 0.3078 (an.) | 346.0                     | 1.10 / 4.73           | 64.7       |

**Observations.** B-large has +6 % FLOPs but -25 % training it/s and +50 % latency at batch 10: consistent with the
non-multiple-of-8 channel hypothesis (control-run entry), still not verified. SA32 and MD32 have nearly the same FLOPs
(+39 % vs B) but SA32 trains slower; MD32 needs about twice the training memory of any other configuration.

---

## 2026-10-13 — Block micro-benchmark (replaces M-E, D12)

**Action.** `python scripts/microbench.py` at commit `ee7ee46` (clean tree), right after `cost.py`, GPU idle. d = 64,
batch 10, float32, square grids 8 / 16 / 32 / 64 / 128 (64 to 16 384 tokens); forward + backward time (median of 20
after 5 warm-up, synchronised) and peak extra memory of one forward + backward. Output `docs/results/microbench.csv`,
figure `docs/figures/microbench.png`. SDPA backend selected by PyTorch (`torch._fused_sdp_choice`) at every size:
`EFFICIENT_ATTENTION` (flash attention needs fp16 / bf16).

**Evidence.**

| Tokens | MD ms / MiB    | SA-SDPA ms / MiB | SA-naive ms / MiB |
|--------|----------------|------------------|-------------------|
| 64     | 1.31 / 18.0    | 0.45 / 2.2       | 0.54 / 3.3        |
| 256    | 1.97 / 68.2    | 1.10 / 9.5       | 1.02 / 43.2       |
| 1 024  | 6.57 / 267.0   | 9.27 / 35.4      | 15.74 / 652.6     |
| 4 096  | 52.18 / 1066.9 | 140.90 / 141.6   | OOM               |
| 16 384 | 235.68 / 4266.3 | 2219.17 / 566.8 | OOM               |

**Observations (neutral).** Time: SA (both forms) is faster than MD up to 256 tokens; the MD and SA-SDPA curves
cross between 256 and 1 024 tokens, and above that MD is faster (2.7x at 4 096, 9.4x at 16 384). From 4 096 to
16 384 tokens MD time grows 4.5x and SA-SDPA 15.7x (close to linear and to quadratic). Memory: the curves do not
cross for MD vs SA-SDPA; SA-SDPA uses the least memory at every size (MD ~7.5-8x more at the larger sizes); SA-naive
crosses MD between 256 and 1 024 tokens and runs out of memory (8 GB GPU) from 4 096 tokens. The 8x8 and 32x32
positions used in the experiments (64 and 1 024 tokens) lie on either side of the time crossing, consistent with the
training it/s (SA8 faster than MD8, SA32 slower than MD32). One GPU, one precision; not a general statement.

---

## 2026-10-13 — CBAM downgraded to an exploratory control (kept in all tables)

**Context / why.** CBAM was listed among the final configurations (2026-10-11) and in the planning report. On
reflection it is not the natural comparison for Mamba: it re-weights channels and positions but does not move
information between positions, so it is not a global mixer. Its form is also not symmetric with SA and MD: it is
multiplicative with no residual, so at initialisation it does not reproduce B, while SA and MD do (zero output
projection, unit test 4). Measured here: with seed 0, at initialisation, on 50 validation samples (eval mode, CPU),
sum|CBAM(x)| / sum|x| at the 8x8 level is 0.154 (min 0.152, max 0.155), i.e. the features start scaled by ~0.15
(the rough "two sigmoids of ~0.5" estimate of ~1/4 is an over-estimate: the channel gate is below 0.5 on average).
At 8x8 the 7x7 conv of the spatial branch covers almost the whole map (central positions see 7x7 of the 8x8
cells), so the spatial gate is in practice global, but it is still only a re-weighting.

**Decision.** CBAM is **downgraded, not removed**: the three night-2 runs stay in every table (they were in the
planning report; dropping them after seeing the results would be selective reporting). No further seeds for CBAM.
CBAM vs B is reported as exploratory only. Alternatives considered: remove CBAM (rejected, selective reporting);
add a residual CBAM variant (rejected: new configuration after seeing results, no time).

**Evidence.** Measurement script run from the scratchpad (not committed; it builds `build_from_cfg({"variant":
"CBAM", "level": 8})` at `torch.manual_seed(0)` and hooks `layer3` and the block).

---

## 2026-10-13 — Note on D10 (simplified SS2D): what is shared, and the conv1d wrap-around

**Context / why.** The D10 row in "Global-mixing blocks: implementation, deviations D10-D13" lists the
simplification but not its motivation, nor one side effect of flattening.

**Motivation of D10 (recorded now).** Simplicity, time (deadline 2026-10-18) and the use of a single official,
tested module (`mamba_ssm.Mamba`, CUDA kernels checked against the reference in unit test 6) instead of a custom
re-implementation of VMamba's SS2D.

**Differences from VMamba's SS2D.** In VMamba each of the 4 directions has its own `x_proj`, `dt_proj`, `A` and
`D`; there is a depthwise 2D 3x3 conv before the scans and an output norm. Here everything is shared by the 4
directions (one `Mamba` module), there is no 2D conv, and Mamba's causal **conv1d (d_conv = 4) acts on the already
flattened sequence**. So in each scan direction the first 3 tokens of every row (row scans) or column (column
scans), except the first line, have in their conv window tokens from the far end of the previous line, i.e. from
the opposite edge of the grid. Counted per direction (each of the 4 orders gives the same count): **21 / 64 tokens
at 8x8 and 93 / 1024 at 32x32** (= 3 x (n - 1)), as in the request. The first line sees zero padding instead.

**Effect.** Not measured. Declared as a limitation of the MD variants; the recurrence (SSM) crossing line ends is
intended (that is the global scan), the conv wrap-around is a by-product.

**Evidence.** Count script in the scratchpad (not committed): for each scan order, a token is counted if any of
the previous 3 tokens in scan order lies on a different row (row scans) or column (column scans); output
`8: 21 21 21 21 / 64`, `32: 93 93 93 93 / 1024`.

---

## 2026-10-13 — Night 3: pre-registered seed extension (queue prepared, not launched)

**Context / why.** With 3 seeds per configuration the validation differences between variants (night-2 entry) are
of the order of 1-4 seed standard deviations and not established. B's three runs come from night 1 (commit
`3e69402`), so every comparison with B also mixes two nights / commits (same model, unit test 2). **This extension
was decided after seeing the validation results of seeds 0-2** (declared here).

**Decision (fixed now, before any run).**
- Seeds per configuration, final: **8 (seeds 0-7) for B, MD8, SA8, MD32, SA32**; **3 for CBAM8** (downgraded, see
  "CBAM downgraded"); **B-large: 3 + those the queue completes** among seeds 3-7 (low priority, at the end).
- **All runs will be reported, whatever their result.**
- Purposes: more power for MD vs B and MD vs SA (at 8x8 and 32x32); remove the cross-night / cross-commit
  confound for B (B seeds 3-7 run with the current code, in the same night as the variants).
- B seeds 0-2 (night 1) and B seeds 3-7 (night 3) will be compared with each other as a consistency check
  (same model and protocol; a large difference would point to a code or machine effect).
- Training protocol unchanged: 133 epochs, `--split-mode grouped` (split seed 0), Adam lr 6e-4, batch 10, final
  checkpoint evaluated on validation right after training.

**Queue** `queues/night3.txt` (order fixed): for seed 3, 4, 5, 6, 7: B, MD8, SA8, MD32, SA32; then B-large seeds
3-7. B runs use explicit `--variant B` and are named `B_s<k>` under `~/runs/night3`, so they cannot be confused with
night-1 `ep133_s<k>`. Durations: longest real wall time (train + eval) per configuration in nights 1-2: B 12.06,
MD8 14.82, SA8 12.67, MD32 21.51, SA32 24.37, B-large 16.22 min; one seed block = 85.4 min. `est=` adds ~5 %
(13 / 16 / 14 / 23 / 26 / 17 min) and is used only for the cutoff check until a run of the same configuration
has finished in `~/runs/night3`. Night 2 had no measurable gaps between runs (306 min of runs in 306 min).
**Expected ends with a start at 22:30:** seed 3 block 23:55, seed 4 01:20, seed 5 02:46, seed 6 04:11, seed 7
05:37; B-large s3-s7 05:53 / 06:09 / 06:25 / 06:42 / **06:58**. Cutoff 08:30: all 30 runs are expected to fit
(even with every run at its `est=`: 545 min, end 07:35).

**Launch.** `scripts/launch_night3.sh` (new): refuses to start on a dirty tree (`git status --porcelain`, tracked
or untracked) or if the tmux session `night3` exists; otherwise
`tmux new -d -s night3 "export TMPDIR=...; systemd-inhibit --what=sleep:idle:handle-lid-switch --why=night3-queue
<mdf python> scripts/run_queue.py queues/night3.txt --out-root ~/runs/night3 --cutoff 08:30"`. Checked today: on
the dirty tree it exits 1 with the list of changed files; `run_queue.parse_queue` reads 30 runs.

---

## 2026-10-13 — Exploratory validation analysis (A1-A7): method and tooling

**Context / why.** Validation has already been looked at (night-2 entry), so it can be explored freely; **all
results of this analysis are EXPLORATORY** and do not change the pre-registered plan. The test set stays closed.
Goal: understand where (channels, flow conditions, regions of the field) MD / SA differ from B before deciding on
cross-validation (separate entry).

**Tooling.** `mdf/explore.py` (reusable, also for the test split later) and `scripts/analyze_val.py`.
- **Test guard.** `explore.load_split("val")` asserts that the directory is `data/train`, that no resolved file
  path lies in `data/test`, and that the computed list equals `splits/val_grouped_seed0.txt`; the script also checks
  the val sha256 recorded in every run. `--split test` exits unless `--allow-test` is given. Not run on test.
- **Runs from the command line**: `--config LABEL=GLOB` (repeatable, so B from nights 1 and 3 can be merged) and
  `--seeds 0-7`; default = the night-1 / night-2 runs. Nothing seed- or split-specific is hard-coded.
- **Inference** of each run (GPU, batch 1, ~1-2 s per run, 45 s in total for 21 runs including data loading);
  only aggregates are kept, cached in `~/runs/explore-cache` (keyed by the checkpoint sha256): per-sample sums
  of |o - t| and |t| per channel (whole grid and per region), the mean |error| map per channel, and the
  predictions of the 6 A7 samples. MD on `--device cpu` would use `mamba_impl=ref`.
- **Regions** (disjoint; rows = streamwise x, flow towards increasing row index, verified on the data: the
  column of the wake minimum 20 px behind the trailing edge follows the free-stream angle, correlation 0.96 over
  101 validation samples): interior (mask); wall band = fluid cells at Euclidean distance 1-4 px from the nearest
  airfoil cell (boundary-layer proxy); wake = other fluid cells downstream of the trailing edge (last mask row,
  mean column) along the free-stream direction, within 5 px of that line; rest = other fluid cells.
- **Sample properties** (model independent): |alpha| = |atan2(vy_inf, vx_inf)| from the input channels;
  sum |p target| (normalised); relative thickness = max over rows of the airfoil cells in that row / number of
  rows with airfoil cells (chord ~60 px, so quantised to ~1/60); external difficulty = seed-mean rel_p of
  B-large and CBAM8 (not involved in the MD / SA vs B comparisons, to avoid regression to the mean).
- **Bootstrap** (`explore.paired_bootstrap`, 10 000 repetitions, seed 0): per-sample seed means, per-sample
  difference, profiles (85) resampled with replacement, the statistic being the mean over the samples of the
  resampled profiles (sample-weighted, as the metric). Two-level version: seeds of each configuration resampled
  with replacement at each repetition as well (coarse with 3 seeds). Also median per-sample difference and the
  fraction of profiles (and samples) where the variant has the lower error.
- **A7 selection** (data only): the known hard high-angle cases `pfcm_6703_-2724` and `rae5214_8305_-3198`
  (pressure-analysis entry) are present in validation but are **not** among the 3 largest |alpha| (-22.1 and
  -21.1 deg vs -22.45 deg max), so they are included explicitly (`--qual-files`) and the third slot is the
  largest |alpha|; plus the 3 smallest sum |p target|.

**Finding during the consistency check (evaluation detail).** With batch 20 the recomputed per-sample ratios
differed from `eval_val_samples.csv` by up to 1.6e-3 (0.85 % relative, e.g. rel_vy of `n63210_3827_58`). Cause:
cuDNN TF32 convolutions (PyTorch default `cudnn.allow_tf32 = True`) depend on the batch size; with TF32 off, batch 20
vs batch 1 differ by <= 2e-7. Effect on the run means (`night1/ep133_s0`): <= 0.0002 points for batch 20 vs 1, <= 0.001
points for TF32 off vs on, i.e. far below the seed spread (>= 0.02). The analysis therefore uses batch 1, as
`evaluate.py`. Per-sample values are reproducible to ~1e-7 only at the same batch size.

---

## 2026-10-13 — Exploratory validation analysis (A1-A7): results (EXPLORATORY)

**Action.** `python scripts/analyze_val.py --seeds 0-2` at commit `e521ed3` (clean tree); 7 configurations x 3 seeds
(B = `night1/ep133_s*`, others `night2/*`), validation (402 samples, 85 profiles). GPU inference only (~45 s), no
training. Outputs `docs/results/val-explore/` (`summary.md` has every table) and `docs/figures/val-explore/`.
Test set not used. Everything below is exploratory and descriptive; with 9 comparisons x 4 metrics in A3 and 200
stratum cells in A4, some intervals excluding 0 are expected by chance.

**Consistency.** Recomputed per-sample rel_p / rel_vx / rel_vy / rel_all vs each run's `eval_val_samples.csv`: max
|diff| 1.6e-7 over 21 runs (batch 1; see the method entry for batch 20). Identity rel_all_s = sum_c w_cs rel_cs:
max |lhs - rhs| 1.1e-16.

**A1 (channel weights).** Mean w (min-max over samples): p 0.046 (0.004-0.123), **vx 0.825 (0.632-0.989)**, vy 0.129
(0.008-0.270). rel_all is mostly rel_vx. MD8 - B: d rel_all +0.011 points = p +0.003, vx +0.008, vy -0.000; the
-0.33 in rel_p and -0.19 in rel_vy contribute almost nothing. MD32 - B: +0.027 = +0.003 / +0.023 / +0.000.

**A2 (MAE, normalised, x1e-3, fluid cells).** B p 3.33, vx 6.80, vy 4.27; mean |target| p 27.8, vx 473.5, vy 77.6.
vy's absolute error is of the same order as p's and below vx's; its high relative error comes from the small
denominator, not from a "hard" channel. MD8 has slightly *higher* fluid MAE than B on all three channels (3.36 /
6.92 / 4.28) although its mean rel_p and rel_vy are lower: the relative gain sits in samples with small targets
(A4). Interior (target exactly 0): B 5.0 / 10.5 / 3.6, i.e. the networks predict non-zero values inside the
airfoil, of the same order as fluid errors (MD32 vx 14.3, CBAM8 8.8); ~3 % of the whole-grid vx error.

**A3 (paired, points; CI one level / two levels).**
- MD8 - B: rel_all +0.011 [-0.063, +0.101] / [-0.085, +0.141]; rel_p -0.33 [-0.69, +0.01] / [-0.99, +0.29];
  rel_vy -0.19 [-0.35, -0.03] / [-0.47, +0.06]; median diff of rel_all -0.040, MD8 better on 59 % of profiles.
- SA8 - B: rel_all +0.064 [+0.001, +0.130] / [-0.023, +0.151]; rel_vy +0.24 [+0.10, +0.39] / [+0.03, +0.47].
- MD8 - SA8: rel_p -0.68 [-1.10, -0.28] / [-1.43, +0.09]; rel_vy -0.42 [-0.60, -0.24] / **[-0.70, -0.14]**;
  rel_all -0.054 [-0.137, +0.045]; MD8 better on 72-73 % of profiles for p and vy.
- MD32 - B, SA32 - B, B-large - B, CBAM8 - B, MD32 - SA32, MD32 - MD8: no two-level interval excludes 0 for any metric.
- The only two-level intervals excluding 0 in A3: SA8 - B rel_vy (worse) and MD8 - SA8 rel_vy (MD8 better).

**A4 (strata).** Spearman |alpha| vs sum |p target| = 0.92 (same axis in practice); thickness and external
difficulty almost uncorrelated with them (|rho| <= 0.20). Pattern: in the low-|alpha| / small-pressure-signal
quintiles MD8 is lower than B (e.g. |alpha| Q1 rel_all -0.07 [-0.11, -0.03], rel_p -1.03 [-1.93, -0.27]; Q2
rel_all -0.11) and SA8 higher (Q1 rel_all +0.06 [+0.03, +0.09]); MD8 - SA8 Q1 rel_p -1.99 [-3.07, -1.06]. In the
highest |alpha| quintile (18-22 deg) all four variants are *above* B in point estimate (rel_all MD8 +0.26
[-0.05, +0.64], MD32 +0.16, SA8 +0.15, SA32 +0.30 [+0.06, +0.58]), with wide intervals. Because all four share the same
three B runs, a common shift in one stratum may come from B's runs themselves (B seeds 3-7 will show this).
External-difficulty Q5: MD8 rel_p -0.54 [-2.03, +0.96]: no signal. Out of 200 cells, 59 exclude 0 with the one-level
CI and 22 with the two-level CI; of the latter, none is an MD - B cell in the hardest quintile of any criterion.

**A5 (regions; MAE x1e-3; mean cells per sample: interior 323, wall 544, wake 317, rest 15 200).** Errors are
largest in the wall band (B p 11.2, vx 31.2, vy 10.9) and at the leading edge (maps). MD8 - B: wall vy -0.14
[-0.25, -0.04] (-1.3 %), all other regions / channels CI include 0; MD32 - B: wall vy -0.38 [-0.49, -0.25] (-3.5 %).
SA8 - B: wall p +2.4 %, vx +1.7 %, wake vx +5.7 % [+0.45, +1.38]. MD8 - SA8: wall vx -2.7 %, vy -1.9 %. CBAM8 - B:
wall +5 % on p and vx. One-level CIs only.

**A6 (train / val L1, last epoch, x1e-3, mean over seeds).** Gap val - train: B 1.87 (sd 0.10), MD8 2.16 (0.10), SA8
2.19 (0.13), MD32 1.96, SA32 2.14, B-large 2.01, CBAM8 2.14. MD8 and SA8 reach a lower training L1 (2.71, 2.76 vs
B 2.95) with a similar or higher val L1 (4.86, 4.95 vs 4.82): the 8x8 mixers fit the training set more without
generalising better. Caveat: train L1 is the epoch mean in train mode (BatchNorm batch statistics, weights changing).

**A7 (qualitative, seed 0).** `pfcm_6703_-2724` (-22.1 deg): the target has 1071 reverse-flow (vx < 0) fluid
cells downstream of the trailing edge and a low-pressure vortex detached behind the airfoil; B / MD8 / SA8 predict
205-229 such cells and a vortex attached to the suction side, rel_p 1.72-1.83 for all three. `rae5214_8305_-3198`:
620 target cells vs 257-268 predicted. `e836_3623_-1496` (-22.45 deg): separation bubble close to the wall (440
reverse-flow cells within 8 px), predicted well (rel_p 0.04-0.06). So the high-angle samples are separated flows;
the worst ones look like a detached-vortex state that none of the models reproduces. Whether these RANS targets are
converged steady solutions is not known (open question of the pressure entry). Small-pressure-signal samples
(`ht05`, `hh02`, |alpha| < 0.5 deg): rel_p 0.44-0.83 for all three models, absolute errors close to the median.

**Open questions.** Whether the low-angle advantage of MD8 holds with 8 seeds (night 3); whether the common
high-angle shift is a property of B's three runs.
