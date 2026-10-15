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

---

## 2026-10-13 — Cross-validation: prepared, decision on 2026-10-14 after the exploratory analysis

**Context / why.** The exploratory analysis has 85 validation profiles and few hard cases (e.g. the top |alpha|
quintile has 81 samples / 50 profiles, and the worst separated-flow cases are a handful). A profile-disjoint
k-fold over the 6400 train + validation files would give predictions on unseen profiles for all 1456 profiles, i.e.
many more hard cases, to study where MD / SA differ from B. **Only prepared today (code, tests, queue); whether to
run it is decided on 2026-10-14.**

**Implementation.**
- `mdf/data.py`: `kfold_profile_folds(files, n_folds, split_seed)`: sorted unique profiles shuffled once with split
  seed 0; each profile, in that order, goes to the fold with the fewest files so far (ties: lowest index), so folds
  are balanced in files. `split_files_kfold`, `split_files_for` (dispatch over all modes), `split_file_path` for
  kfold (`splits/kfold5_seed0_fold<k>.txt`); `load_train_val` / `load_val` take `n_folds`, `fold`; `split_info`
  records them. Existing modes untouched (`SPLIT_MODES` unchanged; grouped val sha256 still `3b26466a...`).
  Only `data/train` is ever listed: the test set is not part of any fold.
- `scripts/train.py`: `--split-mode kfold --n-folds 5 --fold k` (both required with kfold and refused without it);
  run ID contains `kfold5f<k>`. `scripts/evaluate.py --split val` reads `n_folds` / `fold` from `config.json`.
  `scripts/make_split.py --split-mode kfold --n-folds 5 --write` writes and checks the 5 fold files.
- Alternatives considered: round-robin assignment of profiles (fold sizes would vary with profile sizes); stratifying
  folds by angle (rejected: more complexity, and fold-level balance in |alpha| is not needed for out-of-fold
  predictions).

**Evidence.** `make_split.py --split-mode kfold --n-folds 5 --write`: 6400 files, 1456 profiles; validation folds
1277 / 1278 / 1276 / 1286 / 1283 files, 293 / 291 / 284 / 289 / 299 profiles; 0 shared profiles in every fold; every
file in exactly one validation fold. `pytest tests/test_kfold.py`: **8 passed** (cover every profile exactly once,
each file in exactly one fold, train / val disjoint and complete, balance <= 20 files, no test file (paths resolved
under `data/train`; test names compared via directory listing only, no test file opened), determinism and committed
files, grouped / random unchanged, bad arguments). `tests/test_blocks.py`: 43 passed, 6 skipped (unchanged).
`load_train_val(kfold, fold 0)` on CPU (data loading only, no training): 5123 / 1277 files, 1163 / 293 profiles,
`load_val` gives the same list and sha256. train.py without `--fold` or without `--split-mode kfold`: argparse error,
no run directory created.

**Queue** `queues/cv.txt` (not launched): for fold 0..4: B, MD8, SA8; seed 0, 133 epochs, out root `~/runs/cv`.
Durations from the night-1 / night-2 timings rescaled to ~512 training batches (vs 599) and ~128 validation batches
per epoch (vs 40), plus a 3.2x larger final evaluation: B 11.2, MD8 13.9, SA8 11.8 min per fold (only ~7 % shorter
than the main runs, because the per-epoch validation is 3.2x larger); **total ~184 min** (est= with margin: 200 min).

**Limits (if run).** One seed per fold; training sets differ from the main runs (5114-5124 files instead of 5998,
and different profiles), so the results are **exploratory only** and cannot be pooled with the main validation
numbers. Out-of-fold merging and the per-stratum analysis on CV predictions are not implemented yet
(`scripts/analyze_val.py` handles one split per call).

---

## 2026-10-14 — Decisions taken on 2026-10-13 evening and confirmed on 2026-10-14 morning, before looking at any metric of seeds 3-7

**Context / why.** Night 3 (seeds 3-7 of B, MD8, SA8, MD32, SA32; B-large seeds 3-7) has run. Before reading any of
its metrics (no CSV, no evaluation JSON, no evaluation line of `queue.log` has been read; only the run directory
names have been listed), the decisions below are fixed and committed. This commit is the **pre-registration commit**
for the final test analysis. The test set has never been opened.

### 1. Cross-validation abandoned; exploratory data-quantity experiment instead (conditional)

**Decision.** The profile-disjoint 5-fold cross-validation (entry "Cross-validation: prepared ...") is **not run**:
cost in human time (merging, analysis), one seed per fold, and results that cannot be pooled with the main runs
(different training sets). The code stays in the repository, unused. Alternatives considered: run it as prepared
(rejected for the reasons above); run it with fewer folds (same problems).

In its place, an **EXPLORATORY experiment on the quantity of training data**, run only if the condition below holds.
- Configurations B, MD8, SA8; seeds 0-2; fractions ~25 % and ~50 % of the training part of the grouped split
  (5998 files, 1371 profiles).
- **Reduction per profile.** All 1371 training profiles are kept; for a profile with n training samples,
  k = max(1, floor(f * n + 0.5)) samples are kept (round half up, declared explicitly because Python's `round` rounds
  half to even). This varies the density of flow conditions per profile, which is the direction in which the full
  dataset grows (~36 samples per profile in the full set vs ~4.4 here); geometric diversity is not varied.
- **Selection.** For each profile, its training files (sorted by name) are permuted with an RNG derived from the run
  seed and the profile name; the subset is the first k files of that permutation. Hence for the same seed B, MD8 and
  SA8 see the same samples, and the 25 % subset is contained in the 50 % subset (k is non-decreasing in f).
- **Same number of optimisation steps as the full runs**: 133 x 599 = 79 667 steps (epochs over the smaller set
  adapt). Learning rate at step s = `computeLR(floor(s / 599), 133, 0.1 lr, lr)`, i.e. identical step by step to the
  full runs (the original schedule halves the LR from mid-training on). Validation and one row of `metrics.csv`
  every 599 steps (133 rows, curves comparable with the full runs).
- **Main number: final checkpoint**, as always. Descriptive diagnostics: val_L1 curves and position of their
  minimum, to see late overfitting. The minimum is NOT used as a result (it would select the checkpoint on the same
  validation).
- Evaluated on validation only, never on the test set.
- Question: do the differences MD8 - B and SA8 - B (in either direction) and the train / validation gap change with
  the quantity of data?
- Limits: geometric diversity not varied; extrapolation to the full dataset remains a hypothesis; with 3 seeds the
  power is low.
- **Condition (operational, fixed now).** The experiment runs in the night 14 -> 15 only if verification 5a
  (training error in eval mode, all seeds available) gives, **for MD8 or for SA8, both**:
  (i) training L1 in eval mode lower than B with (mean_variant - mean_B) / SE_Welch <= -2;
  (ii) validation L1 NOT lower than B beyond noise: (mean_variant - mean_B) / SE_Welch >= -1;
  SE_Welch = sqrt(s_var^2 / n_var + s_B^2 / n_B), s = sample std over seeds (ddof = 1). Training L1 = mean over the
  5998 training samples of the per-sample L1 (final model, eval mode, batch 10); validation L1 = the same on the 402
  validation samples (from the same kind of inference, batch 1). The rule is applied literally and the four ratios
  (MD8 / SA8, train / val) are reported.

### 2. Final analysis plan for the test set (confirms the draft of "Night 2: pre-registered queue")

Additions decided AFTER the exploratory look at validation are marked (added).
- Primary metric `rel_all`; key secondary `rel_p`; also `rel_vx`, `rel_vy`.
- `rel_p` always reported with the aggregate `rel_p` (sum|e| / sum|t| over all samples) and the median, and (added)
  the absolute MAE in the fluid per channel.
- Mean +- std (ddof = 1) over seeds: 8 for B, MD8, SA8, MD32, SA32; B-large the completed ones; 3 for CBAM8.
  Main interval: (added) **two-level bootstrap (seeds + profiles), 10 000 repetitions** (`explore.paired_bootstrap`).
- Main comparisons: MD8 vs B, MD8 vs SA8, MD8 vs B-large. Exploratory: MD32 vs MD8, SA32 vs SA8, MD32 vs SA32,
  CBAM8 vs B. No formal correction for multiple comparisons; results are descriptive.
- (added) No strata on the test set (90 samples, 30 profiles: too few per quintile); strata only on validation,
  exploratory.
- (added) Regions (wall band, wake) and divergence: descriptive; divergence as defined in phase 6 of 2026-10-14
  (implemented and tried on validation before the test is opened).
- The test set is evaluated **once**, all configurations together, with `scripts/evaluate.py` (batch 1), after this
  commit and after the team's explicit go-ahead; results are reported whatever they are. If a technical bug emerges
  after opening, it is fixed only if the fix does not depend on the results; the notebook records what changed and
  both versions are reported. The runs of the data-quantity experiment do not go on the test set.

### 3. Correction note (dated 2026-10-14) to the entry "Note on D10 (simplified SS2D) ..." (2026-10-13)

The phrase "as in the request" after the token counts (21 / 64 and 93 / 1024) refers to the user's request that
asked to verify those counts, not to a published source. The counts themselves were verified by the script described
in that entry. The entry is not rewritten.

---

## 2026-10-14 — Pre-registered ablation: faithful SS2D (MDv8)

**Context / why.** Decided on 2026-10-14 morning, before looking at any metric of seeds 3-7 and before opening the
test set. Purpose: check that the simplification D10 (one shared `Mamba` module for the 4 directions, no 2D conv, no
output norm) is not what penalises Mamba.

**Variant MDv8.** Block at 8x8, at the same point and in the same residual form as MD8 (`x + SS2D(LN(x))`), with
SS2D faithful to the version of the VMamba paper (Liu et al. 2024, arXiv 2401.10166): `in_proj` to x and z;
depthwise 2D 3x3 conv + SiLU on the x branch; 4 directions with separate `x_proj`, `dt_proj`, `A_log` and `D` per
direction; selective scan; sum of the 4 directions mapped back to 2D; output LayerNorm; gating with SiLU(z);
`out_proj`. d_state 16, expand 2, dt_rank as Mamba (ceil(C / 16)).
- Declared deviations: `out_proj` initialised to zero (consistent with MD8 and SA8); no drop path.
- Protocol identical to the other runs (133 epochs, grouped split, Adam lr 6e-4, batch 10, final checkpoint); seeds
  0-7.
- Question (validation, exploratory): does MDv8 differ from MD8 and from B? Same analyses as phase 4 (rel_all, rel_p
  with aggregate and median, MAE, two-level bootstrap).
- Test: MDv8 will be evaluated on the test set **whatever its validation result**, in a separate pass after the
  first one, with the same plan as section 2 of the previous entry (comparisons MDv8 vs MD8 and MDv8 vs B), declared
  in the report as a second pass decided in advance.
- Limitation: MDv8 has more parameters than MD8; the comparison mixes faithfulness and size.
- Implementation, tests and a 17-epoch control run follow later today (after the test pass); the official VMamba
  file and commit followed will be recorded then.

---

## 2026-10-14 — Night 3: run check (status and durations only)

**Action.** Read `~/runs/night3/queue.log` (status lines), every run's `config.json` and `queue_timing.json`; searched
the logs for `Traceback` / `Error:` / nan / inf and `metrics.csv` for NaN / Inf. Done after the pre-registration
commit `26b6e7f`.

**Evidence.** Queue started 2026-10-13 23:25 (not 22:30 as planned: later launch), ended 07:56:46, before the cutoff
(08:30). **All 30 runs finished and evaluated on validation**, none interrupted, no `*.interrupted-*` directory, no
STOP file, no error, no NaN / Inf. All at commit `7b663e2`, clean tree. **B-large: all 5 extra seeds (3-7) completed**
(so B-large has 8 seeds). Training wall times (min; `est=` / longest of nights 1-2 in the queue entry): B 12.0-12.5
(13 / 12.06), MD8 14.7-15.5 (16 / 14.82), SA8 12.6-13.0 (14 / 12.67), MD32 21.4-21.7 (23 / 21.51), SA32 24.4 (26 /
24.37), B-large 16.3 (17 / 16.22). it/s: B 111.4 (s3) and 114.9-115.3, MD8 90.0 (s3) and 93.9-94.2, SA8 107.3 (s3) and
110.1-110.5, MD32 63.5-64.4, SA32 56.0-56.1, B-large 85.1-85.2. The seed-3 runs (the first of the night) were ~3 %
slower; at the start of SA8_s3 the machine showed 1.1 GB of swap in use and less free memory (log "machine state"),
afterwards stable. Throughput does not affect results.

---

## 2026-10-14 — Consistency of B: seeds 0-2 (night 1) vs seeds 3-7 (night 3)

**Action.** New `scripts/b_consistency.py` (reads `eval_val.json`; Welch SE = sqrt(s_a^2/3 + s_b^2/5), ddof = 1):
`python scripts/b_consistency.py --a ~/runs/night1/ep133_s{0,1,2} --b ~/runs/night3/B_s{3,4,5,6,7}`.

**Evidence** (validation, %):

| metric  | seeds 0-2 mean (sd) | seeds 3-7 mean (sd) | diff (3-7 minus 0-2) | Welch SE | diff / SE |
|---------|---------------------|---------------------|----------------------|----------|-----------|
| rel_all | 2.4212 (0.0210)     | 2.4539 (0.0254)     | +0.0328              | 0.0166   | +1.97     |
| rel_p   | 12.4210 (0.2345)    | 12.5835 (0.3045)    | +0.1625              | 0.1920   | +0.85     |
| rel_vx  | 1.5513 (0.0237)     | 1.5700 (0.0124)     | +0.0186              | 0.0148   | +1.26     |
| rel_vy  | 5.9920 (0.0890)     | 6.1127 (0.2496)     | +0.1207              | 0.1229   | +0.98     |

**Reading (neutral).** No ratio exceeds 3 standard errors. All four differences have the same sign (seeds 3-7 slightly
higher error); the largest is rel_all at ~2 SE (+0.033 points), with SEs estimated from 3 and 5 runs, so rough. Not a
sign of a code or machine effect by the pre-registered threshold; it widens B's spread. As pre-registered, all 8
seeds of B are used.

---

## 2026-10-14 — Validation analysis with all seeds (EXPLORATORY)

**Context / why.** Night 3 brings B, MD8, SA8, MD32, SA32 and B-large to 8 seeds (CBAM8 3). Validation has already been
looked at, so everything here is **exploratory**; the pre-registered plan (commit `26b6e7f`) is not changed by it.

**Tooling changes.**
- `scripts/analyze_val.py`: `--external-seeds` (the external difficulty of A4 stays the seed-mean rel_p of B-large +
  CBAM8 **seeds 0-2**, as defined yesterday, so it remains independent of the new runs); comparisons MD8 - B-large and
  SA32 - SA8 added (MDv8 pairs skipped when absent); `--previous DIR` adds "earlier run vs this run" (A3 and A4 |alpha|
  rows; rule: *holds* = two-level CI excludes 0 in both runs, *disappeared* / *appeared* = only in the earlier / only
  in the current one, *none* = neither). Fixed a summary line that was not an f-string (figure path printed literally).
- New `scripts/final_analysis.py --split {val,test}`: the pre-registered final analysis (T1 mean +- std with aggregate
  and median rel_p and fluid MAE; T2 paired comparisons with the two-level bootstrap; T3 regions; T4 divergence; one
  figure of per-run values). It is the script that will be run unchanged on the test set; here it runs on validation.
- `mdf/explore.py`: cache file name now includes a hash of the cache key (the two scripts use different A7 sample
  lists, and would otherwise overwrite each other's cache); `CACHE_VERSION` 3 (divergence aggregates, next entry).

**Action.** Same configuration globs for both scripts: B = `night1/ep133_s*` + `night3/B_s*`; MD8, SA8, MD32, SA32,
B-large = `night2/<c>_s*` + `night3/<c>_s*`; CBAM8 = `night2/CBAM8_s*`; `--seeds 0-7`. Outputs
`docs/results/val-explore-8seed/` (+ figures) with `--external-seeds 0-2 --previous docs/results/val-explore`, and
`docs/results/val-final/` (+ figures). GPU inference only, 51 runs, ~70 s per script. Recomputed per-sample ratios vs
each run's `eval_val_samples.csv`: max |diff| 1.6e-7. Test set not used.

**Evidence: T1** (validation, %, mean (std over seeds); MAE x1e-3):

| config  | seeds | rel_all         | rel_p            | rel_vx          | rel_vy          | agg rel_p        | median rel_p   | MAE p / vx / vy |
|---------|-------|-----------------|------------------|-----------------|-----------------|------------------|----------------|-----------------|
| B       | 8     | 2.4416 (0.0280) | 12.5225 (0.2753) | 1.5630 (0.0185) | 6.0674 (0.2044) | 12.4487 (0.1708) | 9.1739 (0.2424) | 3.370 / 6.862 / 4.315 |
| MD8     | 8     | 2.4363 (0.0521) | 12.3436 (0.3404) | 1.5704 (0.0433) | 5.8278 (0.1448) | 12.5088 (0.3019) | 8.9010 (0.2373) | 3.392 / 6.897 / 4.271 |
| SA8     | 8     | 2.5041 (0.0371) | 12.7959 (0.3786) | 1.6107 (0.0290) | 6.1662 (0.1691) | 12.7046 (0.1947) | 9.2192 (0.3647) | 3.441 / 7.074 / 4.411 |
| B-large | 8     | 2.4785 (0.0484) | 12.6021 (0.4136) | 1.5898 (0.0321) | 6.0661 (0.1178) | 12.6467 (0.3808) | 9.2033 (0.4680) | 3.416 / 6.964 / 4.372 |
| CBAM8   | 3     | 2.4963 (0.0386) | 12.5244 (0.1238) | 1.6120 (0.0300) | 6.0032 (0.2109) | 12.6624 (0.2232) | 9.2316 (0.1270) | 3.442 / 7.099 / 4.405 |
| MD32    | 8     | 2.4668 (0.0294) | 12.4191 (0.2442) | 1.5958 (0.0250) | 5.8480 (0.1307) | 12.6305 (0.2618) | 9.1017 (0.2974) | 3.412 / 6.970 / 4.298 |
| SA32    | 8     | 2.4714 (0.0405) | 12.4465 (0.2691) | 1.5954 (0.0353) | 5.9544 (0.1880) | 12.6469 (0.1884) | 9.0107 (0.1668) | 3.426 / 7.012 / 4.335 |

**T2, main comparisons** (points; two-level 95 % CI):
- MD8 - B: rel_all -0.005 [-0.078, +0.083]; rel_p -0.18 [-0.63, +0.24]; rel_vx +0.007 [-0.047, +0.078]; **rel_vy -0.24
  [-0.46, -0.04]**; fluid MAE p / vx / vy all CIs include 0 (+0.02 / +0.04 / -0.04 x1e-3).
- MD8 - SA8: rel_all -0.068 [-0.152, +0.031]; rel_p -0.45 [-1.00, +0.04]; **rel_vy -0.34 [-0.56, -0.12]**; MAE CIs include 0.
- MD8 - B-large: rel_all -0.042 [-0.124, +0.048]; rel_p -0.26 [-0.75, +0.20]; **rel_vy -0.24 [-0.43, -0.06]**.
- Exploratory (MD32 - MD8, SA32 - SA8, MD32 - SA32, CBAM8 - B): no two-level CI excludes 0 for rel_* or fluid MAE.
- rel_all is ~82 % rel_vx (A1); MD8's lower rel_vy contributes -0.010 points to rel_all.

**Earlier (seeds 0-2) vs now (all seeds)** (two-level CI; all A3 / A4-|alpha| rows in `earlier_vs_now.csv`):
- *Holds*: MD8 - SA8 rel_vy (-0.42 -> -0.34); MD8 - B rel_all in |alpha| Q1 and Q2 (-0.07 -> -0.06; -0.11 -> -0.08);
  MD8 - SA8 rel_all Q1, Q2 and rel_p Q1.
- *Appeared*: MD8 - B rel_vy (-0.19 [-0.47, +0.06] -> -0.24 [-0.46, -0.04]); MD32 - B rel_all Q1.
- *Disappeared*: SA8 - B rel_vy (+0.24 -> +0.10 [-0.13, +0.32]); SA8 - B rel_all Q1; MD32 - B rel_p Q1.
- MD8 - B rel_p: -0.33 -> -0.18, CI still includes 0 (weakened). SA8 - B rel_all: +0.064 -> +0.062; two-level CI
  [-0.023, +0.151] yesterday, now still includes 0 (status none in both).
- **Hardest |alpha| quintile (18-22 deg), rel_all:** MD8 - B +0.26 -> **+0.09** [-0.20, +0.45]; MD32 - B +0.16 -> +0.17;
  SA8 - B +0.15 -> +0.21; SA32 - B +0.30 -> +0.18; none excludes 0 with the two-level CI. Yesterday's common
  positive shift partly came from B's three runs (MD8's shrinks); for SA8, MD32, SA32 it stays at ~+0.2 in point
  estimate.

**A6 (train mode, last epoch, x1e-3, 8 seeds).** Gap val - train: B 1.91 (sd 0.14), MD8 2.17, SA8 2.22, MD32 2.07, SA32 2.16,
B-large 2.04, CBAM8 2.14; train L1 MD8 2.69, SA8 2.78 vs B 2.96; val L1 MD8 4.87, SA8 4.99 vs B 4.87. Same pattern as
yesterday; checked in eval mode in phase 5a.

**Open questions.** Whether MD8's lower rel_vy (mostly on low-angle samples, small vy denominators, see A2 / A4)
carries to unseen shapes (test).

---

## 2026-10-14 — Divergence of predicted and target velocity (implementation; tried on validation, EXPLORATORY)

**Context / why.** A physical-consistency check (incompressible flow: div v = 0), descriptive, decided in the
pre-registered plan (commit `26b6e7f`, section 2).

**Implementation** (`mdf/explore.py`): `divergence(vx, vy, mapping)`: channels multiplied by their fixed maxima from
`mdata.MAX_TARGETS` (vx 2.04, vy 2.37; verified in the code) so both are in units of v / |v_inf| (the two channels are
normalised by different constants); central differences on the uniform grid (unit: per pixel); `valid_stencil`:
only fluid cells whose 4 neighbours are fluid (grid edge excluded). Per sample (`run_inference`, cache version 3):
mean |div pred|, mean |div target|, mean |div pred - div target| in the wall band (`wall` region) and in the other
fluid cells (`wake` + `rest`). Tests `tests/test_divergence.py` (4 passed): (x, -y) has divergence 0 and (x, y) has
divergence 2 for both axis assignments, flipping both signs flips div, scaling, the 4-neighbour mask.

**Axis orientation, from the data** (`axis_mapping_scan`, validation targets only, model independent). Mapping =
(array axis along which vx is differentiated, sign of the vx term, sign of the vy term); mean over samples of mean
|div target| over valid cells:

| vx along axis | signs (vx, vy) | equivalent pair | mean |div target| |
|---------------|----------------|-----------------|------------------|
| 0 (rows)      | (+, +) / (-, -) | yes, identical  | **0.000413**     |
| 0 (rows)      | (+, -) / (-, +) | yes, identical  | 0.008175         |
| 1 (cols)      | (+, +) / (-, -) | yes, identical  | 0.010605         |
| 1 (cols)      | (+, -) / (-, +) | yes, identical  | 0.006936         |

As expected, the 8 mappings give 4 distinct values in identical pairs (flipping both signs only flips div). The choice
concerns the axis of vx and the relative sign: **vx along axis 0 (rows), same sign** (`DIV_MAPPING = (0, 1, 1)`),
17x below the next one, consistent with the geometry found on 2026-10-13 (rows = streamwise x, flow towards increasing
row index). Scale: mean |d(vx)/dx| of the target is 2.4e-2 (wall band) and 3.6e-3 (other fluid) per pixel, so the
target's own divergence is ~0.26 (wall band) and ~0.07 (elsewhere) of a single derivative term: not zero (grid
interpolation of the RANS solution, normalisation by |v| constant per sample), hence the comparison is with the
target, not with 0.

**Evidence** (validation, x1e-3, mean (std) over seeds; `docs/results/val-final/summary.md` T4): target 6.21 (wall band),
0.25 (other fluid). Predictions: wall band 13.3-14.1, other fluid 0.85-0.89 for all configurations (B 13.56 / 0.89,
MD8 13.40 / 0.86, SA8 13.50 / 0.88, MD32 13.42 / 0.87, SA32 13.41 / 0.85, B-large 13.32 / 0.88, CBAM8 14.14 / 0.88), i.e.
~2.2x the target in the wall band and ~3.5x away from it. Differences of mean |div pred - div target| (two-level CI):
MD8 - B other fluid -0.032 [-0.055, -0.010] (-3 %), wall band -0.23 [-0.57, +0.19]; CBAM8 - B wall band +0.61 [+0.13, +1.08];
all other comparisons include 0.

**Reading.** All networks produce fields with several times the target's divergence; the differences between
configurations are small compared with that common excess.

---

## 2026-10-14 — Verification 5a: training error in eval mode; 5b: the pre-registered condition is satisfied

**Context / why.** A6 compared the last-epoch train L1 in train mode (BatchNorm batch statistics, weights changing
during the epoch) with the val L1 in eval mode. Here the final model of every run is evaluated in eval mode on the
training set itself, and the pre-registered condition (commit `26b6e7f`) is applied literally.

**Action.** `mdf/explore.py`: `load_split("train")` returns the training part of the grouped split (5998 files of
`data/train`; asserts train / validation disjoint and every path under `data/train`, none under `data/test`). New
`scripts/train_fit.py`: final model in eval mode, batch 10 on the training set (per-sample sums in float64 on the GPU),
batch 1 on validation (cached inference, as `evaluate.py`); L1 = per-sample mean |o - t| over the whole grid, mean
over samples (validation: 402 samples; `metrics.csv` uses 400 because of `drop_last`). Batch 10 vs 1 only changes the
TF32 rounding (<= 0.001 points on run means, entry of 2026-10-13), negligible against the seed spread. Estimate first:
4.6 s (B) and 6.5 s (MD32) per run on 2 runs -> ~5 min for 51 runs, under the 10-min limit, so all seeds. Actual GPU
time on the training set: 147 s in total. `python scripts/train_fit.py <same --config as the validation analysis>
--seeds 0-7 --out-dir docs/results/train-fit`. Test set not used.

**Evidence** (x1e-3, mean (std) over seeds; diff / SE = (mean - mean_B) / Welch SE):

| config  | seeds | train L1 (eval) | val L1          | gap             | train L1 - B (diff/SE) | val L1 - B (diff/SE) |
|---------|-------|-----------------|-----------------|-----------------|------------------------|----------------------|
| B       | 8     | 2.7623 (0.1065) | 4.8733 (0.0611) | 2.1109 (0.1225) |                        |                      |
| MD8     | 8     | 2.4756 (0.1746) | 4.8721 (0.1103) | 2.3965 (0.1593) | -0.2867 (-3.97)        | -0.0012 (-0.03)      |
| SA8     | 8     | 2.5572 (0.1122) | 4.9998 (0.0733) | 2.4425 (0.1065) | -0.2051 (-3.75)        | +0.1265 (+3.75)      |
| MD32    | 8     | 2.6746 (0.1069) | 4.9340 (0.0685) | 2.2594 (0.1401) | -0.0878 (-1.64)        | +0.0607 (+1.87)      |
| SA32    | 8     | 2.6019 (0.1156) | 4.9447 (0.0793) | 2.3428 (0.1124) | -0.1605 (-2.89)        | +0.0714 (+2.02)      |
| B-large | 8     | 2.7365 (0.0731) | 4.9526 (0.1024) | 2.2161 (0.1313) | -0.0259 (-0.57)        | +0.0793 (+1.88)      |
| CBAM8   | 3     | 2.5889 (0.0760) | 4.9923 (0.0804) | 2.4034 (0.1028) | -0.1734 (-3.00)        | +0.1191 (+2.33)      |

Training rel_all (eval, %): B 1.436, MD8 1.292, SA8 1.336 (validation 2.442 / 2.436 / 2.504). In eval mode B's train L1
(2.76) is lower than the train-mode epoch mean of A6 (2.96), and so are the variants' (MD8 2.48 vs 2.69).

**Does A6's reading hold?** Yes, and more clearly: in eval mode MD8 and SA8 fit the training set better than B (about
4 Welch SEs) with a validation L1 equal to B (MD8) or higher (SA8). CBAM8 and SA32 show the same direction, weaker;
B-large (more parameters, no block) does not fit the training set better. Exploratory.

**5b, condition applied literally** (pre-registered: for MD8 or SA8, (i) train diff/SE <= -2 and (ii) val diff/SE >= -1):

| variant | train diff/SE | (i) | val diff/SE | (ii) | both |
|---------|---------------|-----|-------------|------|------|
| MD8     | -3.97         | yes | -0.03       | yes  | yes  |
| SA8     | -3.75         | yes | +3.75       | yes  | yes  |

**Outcome: the condition is satisfied** -> the data-quantity experiment is prepared for the night 14 -> 15 (next entry),
in the same night queue as MDv8 (launch unified in `queues/night4.txt`, out root `~/runs/night4`, instead of the
`~/runs/data-eff` named in the session plan, so that one launcher and one cutoff serve both).

---

## 2026-10-14 — Data-quantity experiment: implementation and queue (pre-registered, EXPLORATORY; not launched)

**Context / why.** The condition of verification 5a is satisfied (previous entry), so the experiment fixed in the
pre-registration (commit `26b6e7f`, section 1) is prepared for the night 14 -> 15. Nothing in its design is changed
after seeing 5a.

**Implementation.**
- `mdf/data.py`: `subsample_per_profile(files, fraction, seed)` (k = max(1, floor(f n + 0.5)) per profile, first k of a
  `random.Random(f"{seed}:{profile}")` permutation of the sorted files); `load_train_val(..., train_fraction,
  subset_seed)` (default None: unchanged), `split_info` records fraction, subset seed, full count and rule.
- `mdf/models.py`: `step_lr(step, total_steps, ref_epochs, lr)` = original schedule at virtual epoch
  floor(step * ref_epochs / total_steps), applied only below the base LR (as the epoch loop).
- `scripts/train.py`: `--train-fraction F` (needs `--total-steps`; subset drawn with `--seed`), `--total-steps N`,
  `--lr-ref-epochs` (133), `--log-every` (599): a separate step-budget loop (loader restarted with a new shuffle when
  exhausted; one `metrics.csv` row and one validation every 599 steps, `epoch` = block index; `config.json`
  `epochs: null`, `step_budget`). The epoch loop is not changed; run ID `..._st79667_f25_...`.
- Alternative considered: keep the epoch loop with more epochs (e.g. 458 at 25 %): the LR would then follow the same
  curve in progress but on a different epoch grid, and the validation count would change; rejected.

**Evidence.**
- `tests/test_train_fraction.py`: **6 passed** (every training profile kept, exact k per profile; 25 % subset strictly
  inside the 50 % subset for seeds 0-2, different seeds give different subsets; no validation file, no test name
  (directory listing only); `step_lr` equals the epoch-loop LR at every one of the 79 667 steps; without the option the
  training list has the sha256 recorded in `night1/ep133_s0`). Full suite: see the commit (all pass).
- **Actual sample counts** (equal for seeds 0, 1, 2, since k depends only on n; the files differ): **f = 0.25: 1743 of
  5998 (29.1 %)**, 174 batches per data epoch, ~458 data epochs; **f = 0.5: 3335 (55.6 %)**, 333 batches, ~239 data
  epochs. The realised fractions exceed the nominal ones because of the minimum of 1 per profile and the half-up
  rounding (many profiles have 1-4 samples); declared, not changed.
- Functional smoke test (only training of the day, seconds; scratch directory, deleted): `--variant MD --train-fraction
  0.25 --total-steps 20 --log-every 7 --lr-ref-epochs 4`: 3 rows (7 / 7 / 6 steps), LR 6e-4, 6e-4, 1.275e-4, 1743 files,
  1371 profiles, status finished. Normal-run check: `--seed 0 --epochs 1 --deterministic --variant B` before and after
  the change: `config.json` identical apart from the new arguments (at their defaults) and the output directory
  (split, model, LR, iterations); metrics not bit-identical (train L1 0.03370 vs 0.03441), as expected from 6b
  (non-deterministic CUDA backward; the same happens between two runs of the same code: MD 0.03399 vs 0.03395).
- **Queue** `queues/data-eff.txt`: per seed 0, 1, 2: B, MD8, SA8 at 0.25, then B, MD8, SA8 at 0.5 (18 runs). `est=` 13 /
  16 / 14 min (longest real wall time of the 133-epoch runs + ~5 %: same steps and validation passes); total 258 min
  (real expected ~237). Not launched separately: its lines go into `queues/night4.txt` after the MDv8 runs, out root
  `~/runs/night4` (launch command and expected end in the MDv8 entry later today).

---

## 2026-10-14 — Addendum to the final analysis plan: aggregate and median rel_vy (decided after the 8-seed validation analysis, before opening the test set)

**Context / why.** Decided by the team on 2026-10-14 **after seeing the 8-seed validation analysis and before the test
set was opened** (declared as such). The only robust validation signal is on rel_vy (MD8 lower than B, SA8, B-large),
and rel_vy is a mean of per-sample ratios, sensitive to samples with small vy denominators (A2, A4). So the test
analysis also reports, for rel_vy as already planned for rel_p: the **aggregate ratio** (sum|e| / sum|t| over all
samples) and the **median**, for all configurations (T1) and in the paired comparisons.

**Implementation.** `mdf/explore.py`: `paired_bootstrap_agg_median` (paired difference of the aggregate ratio, from
seed-mean per-sample sums, and of the median over samples of the seed-mean per-sample ratio; one-level and two-level
bootstrap as `paired_bootstrap`; a sample counts as many times as its profile is drawn; the same resampled seeds for
errors and ratios of one configuration). `scripts/final_analysis.py`: T1 gains `agg_rel_vy`, `median_rel_vy`; new T2b
with aggregate and median differences for p and vy (for p this makes explicit in the comparisons what the
pre-registration states for rel_p: "always reported with the aggregate and the median"). Note: in T1 the median is
the per-run median averaged over seeds; in T2b it is the median of the seed-mean per-sample ratios. Tests
`tests/test_bootstrap.py` (2: identical configurations give 0; point estimates as computed directly, CI contains them).
All tests: 63 passed, 6 skipped.

**Tried on validation** (`docs/results/val-final/`, regenerated; exploratory). T1 agg / median rel_vy (%): B 5.660 /
4.279, MD8 5.598 / 4.067, SA8 5.788 / 4.256, B-large 5.748 / 4.202, CBAM8 5.771 / 4.108, MD32 5.647 / 4.061, SA32 5.688 /
4.162. T2b, MD8 - B: agg rel_vy -0.062 [-0.252, +0.146], median rel_vy -0.264 [-0.497, +0.044] (two-level); MD8 - SA8:
-0.190 [-0.396, +0.034] / -0.152 [-0.466, +0.091]; MD8 - B-large: -0.150 [-0.355, +0.058] / -0.156 [-0.380, +0.148].
So on validation MD8's rel_vy advantage over B is mostly in the mean of ratios (-0.24 [-0.46, -0.04]); in the aggregate
it is small and its interval includes 0.

---

## 2026-10-14 — TEST SET: pre-registered final analysis (single pass)

**Action.** After the team's go-ahead ("VIA AL TEST"). Checks: working tree clean, HEAD `ee9b8c5` descends from the
pre-registration commit `26b6e7f` and contains the addendum (`65aa729`). **One pass**, 13:2x local time: `scripts/evaluate.py
--run-dir <run> --split test` (batch 1) on all 51 final runs (B night1 s0-2 + night3 s3-7; MD8, SA8, MD32, SA32, B-large
night2 s0-2 + night3 s3-7; CBAM8 night2 s0-2; no data-quantity run exists yet, none included), 51 / 51 evaluated, 90
samples each, no failure; then `scripts/final_analysis.py <same --config as on validation> --seeds 0-7 --split test
--allow-test --out-dir docs/results/test --fig-dir docs/figures/test`, unchanged from the version tried on validation.
Results committed right after the computation without re-runs (`ca067a4`). Consistency vs `eval_test_samples.csv`: max
|diff| 3.0e-7. 90 samples, 30 profiles (none seen in training).

**Evidence: T1** (test, %, mean (std) over seeds; MAE x1e-3):

| config  | seeds | rel_all        | rel_p          | rel_vx         | rel_vy        | agg rel_p | median rel_p | agg rel_vy | median rel_vy | MAE p / vx / vy |
|---------|-------|----------------|----------------|----------------|---------------|-----------|--------------|------------|---------------|-----------------|
| B       | 8     | 2.776 (0.085)  | 14.91 (0.64)   | 1.797 (0.068)  | 6.64 (0.20)   | 13.74     | 8.53         | 6.24       | 4.13          | 3.78 / 7.90 / 5.03 |
| MD8     | 8     | 2.940 (0.235)  | 15.20 (1.20)   | 1.937 (0.168)  | 6.61 (0.47)   | 14.67     | 8.10         | 6.51       | 3.95          | 4.05 / 8.51 / 5.25 |
| SA8     | 8     | 2.734 (0.235)  | 14.70 (1.70)   | 1.763 (0.171)  | 6.51 (0.62)   | 13.44     | 8.70         | 6.10       | 4.30          | 3.69 / 7.78 / 4.91 |
| B-large | 8     | 2.896 (0.132)  | 14.98 (0.72)   | 1.883 (0.115)  | 6.70 (0.20)   | 14.43     | 8.40         | 6.55       | 4.09          | 3.96 / 8.24 / 5.26 |
| CBAM8   | 3     | 2.720 (0.153)  | 14.42 (0.78)   | 1.768 (0.113)  | 6.40 (0.16)   | 13.21     | 8.16         | 6.08       | 4.15          | 3.65 / 7.82 / 4.90 |
| MD32    | 8     | 2.736 (0.104)  | 13.40 (1.39)   | 1.794 (0.072)  | 5.97 (0.39)   | 13.68     | 8.32         | 6.16       | 3.83          | 3.76 / 7.81 / 4.95 |
| SA32    | 8     | 2.813 (0.195)  | 14.28 (1.20)   | 1.829 (0.158)  | 6.38 (0.52)   | 14.10     | 8.19         | 6.35       | 4.02          | 3.88 / 8.03 / 5.12 |

Per-run rel_all (descriptive, figure `docs/figures/test/per_seed.png`): B 2.63-2.89; MD8 2.55-3.19 (five runs >= 3.03,
three 2.54-2.74); SA8 2.31-3.10; MD32 2.56-2.87; SA32 2.61-3.17; B-large 2.79-3.09; CBAM8 2.57-2.88.

**Main comparisons** (points; two-level 95 % CI; T2 / T2b):
- MD8 - B: rel_all +0.164 [-0.102, +0.527]; rel_p +0.28 [-1.22, +1.76]; rel_vx +0.140 [-0.062, +0.445]; rel_vy -0.04
  [-0.59, +0.49]; agg rel_p +0.93 [-0.47, +2.70]; median rel_p -0.40 [-1.37, +0.57]; agg rel_vy +0.27 [-0.30, +0.94];
  median rel_vy -0.26 [-0.78, +0.42]; fluid MAE p / vx / vy +0.27 / +0.61 / +0.23 x1e-3, all CIs include 0.
- MD8 - SA8: rel_all +0.21 [-0.14, +0.74]; rel_p +0.50 [-1.37, +2.92]; rel_vy +0.10 [-0.55, +0.99]; median rel_vy -0.56
  [-1.10, +0.15] (one-level CI [-0.93, -0.01]).
- MD8 - B-large: rel_all +0.04 [-0.27, +0.38]; rel_p +0.22 [-1.22, +1.85]; rel_vy -0.09 [-0.60, +0.51].
- **No two-level interval of the main comparisons excludes 0, for any metric.**
- Exploratory (MD32 - MD8, SA32 - SA8, MD32 - SA32, CBAM8 - B): no two-level interval excludes 0 for rel_*, MAE,
  aggregate or median. Largest point estimate: MD32 - MD8 rel_p -1.80 [-6.12, +0.31].

**Regions** (T3, descriptive): wall-band MAE as on validation is the largest (B p 12.3, vx 31.5, vy 11.9 x1e-3). Two-level
CIs excluding 0: only MD32 - SA32 wall vy -0.43 [-0.82, -0.09] (-3.7 %).

**Divergence** (T4, descriptive): target 6.48 (wall band) / 0.27 (other fluid) x1e-3; predictions 14.2-15.3 / 0.99-1.03,
i.e. ~2.2x and ~3.7x the target for every configuration, as on validation. Differences of mean |div pred - div target|
with the two-level CI excluding 0: SA32 - SA8 (wall -0.44, other -0.037) and CBAM8 - B (wall +0.69, as on validation);
MD8 - B other -0.028 [-0.057, +0.006] (on validation -0.032, CI excluding 0).

**Validation vs test (qualitative).** Errors are higher on test (B rel_all 2.78 vs 2.44; profiles never seen). The
validation signal of MD8 on rel_vy (-0.24 vs B, interval excluding 0) **is not reproduced** on test (-0.04, interval
including 0); MD8's seed spread is ~3x larger on test than B's (0.235 vs 0.085), while on validation it was ~2x. SA8,
worst on validation, has the lower mean rel_all on test among the 8x8 variants, with an equally large spread. The ranking
of configurations changes between validation and test; with 8 seeds and 30 test profiles none of the differences is
established. The common excess of divergence and the regions pattern are the same on both splits.

**Open questions.** The larger seed spread of the block variants on unseen shapes (descriptive; not analysed further
here, post-test). MDv8: second, separate test pass after night 4, as pre-registered.

---

## 2026-10-14 — Correction to the entry "TEST SET: pre-registered final analysis"

In "Validation vs test", the phrase "SA8 ... has the lower mean rel_all on test among the 8x8 variants" is wrong: CBAM8
(2.720, 3 seeds) is lower than SA8 (2.734). Correct statement: SA8's mean rel_all on test is below B's (2.734 vs 2.776),
the reverse of validation, with an interval including 0 and a seed spread ~3x B's. The entry is not rewritten.

---

## 2026-10-14 — MDv8 (faithful SS2D): implementation, unit tests, control run pre-registered

**Context / why.** Ablation pre-registered this morning (entry "Pre-registered ablation: faithful SS2D (MDv8)", commit
`26b6e7f`), implemented after the test pass as planned.

**Official code followed.** `MzeroMiko/VMamba`, commit `9ca24aee4bbffdb34619151afe8aa7a6ca278e44` (2024-01-19, the last
commit before 2024-01-21; the code was first released on 2024-01-18, the date of arXiv 2401.10166 v1), file
`classification/models/vmamba/vmamba.py`, class `SS2D` (`__init__`, `dt_init`, `A_log_init`, `D_init`, `forward_core`,
`forward`), used by `VSSM` (the paper configs `classification/configs/vmamba/vmamba_{tiny,small,base}_224.yaml` have
`TYPE: vssm`) through `VSSBlock`: `x + drop_path(SS2D(ln_1(x)))`. At that commit the repository had three model files
(`vmamba.py`, `vss.py`, `vssmd.py`); `vmamba.py` is the one the paper configs build. Later versions of the repository
(v2 / v3 SS2D with other scan, init and norm options) were not used. Read with the GitHub API / raw files, nothing
installed. The repository has no LICENSE file at that commit; it is MIT-licensed today.

**Implementation.** `mdf/blocks.py` `SS2DVMamba` (variant `MDv`, `mdf/models.py`; level 8 -> `MDv8`): block
`y = x + SS2D(LN(x))` at the same point as MD8; SS2D: `in_proj` C -> 2E (no bias) split into x and z; depthwise 3x3
`conv2d` (bias) + SiLU on x; 4 scans (rows, columns, rows reversed, columns reversed) with separate `x_proj_weight`
(K, R + 2N, E), `dt_projs_weight` (K, E, R), `dt_projs_bias` (K, E), `A_logs` (K E, N), `Ds` (K E); `selective_scan_fn`
of mamba_ssm with grouped B / C (one group per direction, `(b, 4, N, L)`), delta bias and softplus; the 4 outputs mapped
back to row-major order and summed; `out_norm` LayerNorm(E); `* SiLU(z)`; `out_proj` E -> C (no bias). E = 2C, N = 16,
R = ceil(C / 16). Initialisation as upstream (dt "random" in [1e-3, 0.1], A = 1..N, D = 1, x_proj with `nn.Linear`'s
default; `in_proj` trunc_normal std 0.02 as `VSSM._init_weights` does for every `nn.Linear`); `weights_init` of DfpNet
never touches the block (same construction as MD). `mdf/mamba_ref.py`: `selective_scan_ref` extended to grouped
B / C (`(b, g, n, l)`, as upstream's `repeat "B G N L -> B (G H) N L"`); the 3-D path is unchanged. Parameter names are
those of upstream, so the official module loads our state dict directly.

**Remaining differences from the official SS2D / VSSBlock** (complete list):
1. `out_proj` initialised to zero (declared deviation; upstream: `nn.Linear` default, then trunc_normal 0.02).
2. No drop path (declared; the paper uses stochastic depth 0.1-0.5 across the whole network) and no dropout (upstream
   default 0 as well).
3. Same distributions at initialisation but a different RNG order (upstream draws `nn.Linear` defaults first and
   overwrites `in_proj` later), so the same seed does not give the same values as an upstream module.
4. mamba_ssm 2.3.2 (`selective_scan_fn`) instead of the January-2024 mamba_ssm 1.x / VMamba's own kernel; same
   operation (checked against the reference below).
5. Context: one block inside the U-Net at 8x8 with C = 128 (as MD8), instead of VMamba's hierarchy of stages,
   patch embedding and merging.
6. Layout only: our block receives (B, C, H, W) and permutes to channel-last before SS2D (upstream is channel-last);
   `reshape` instead of `view`. No numerical effect (fidelity test: bit-identical).
The block LayerNorm is `nn.LayerNorm(C)` (eps 1e-5), as VSSM passes `nn.LayerNorm` to VSSBlock (whose own default
would be eps 1e-6).

**Unit tests** (`tests/test_mdv.py`, CUDA, **15 passed**; full suite 78 passed, 6 skipped):
- shapes at 8x8 and 32x32 -> (2, 3, 128, 128), finite;
- identity at initialisation (zero `out_proj`) with B's skeleton weights: max |diff| vs B 1.49e-8 (CUDA), 0 (ref, CPU);
- CUDA vs reference scan with `out_proj` randomised: mixing term max |diff| 2.4e-7 (8x8, C = 128) and 4.8e-7 (32x32,
  C = 64); input and parameter gradients within the mamba_ssm float32 tolerances (largest 1.5e-5 on in_proj with values
  up to 57);
- after two Adam steps every block parameter has a non-zero gradient (first step: only `out_proj`, as for MD);
- `weights_init` does not touch the block (rebuilt from the same RNG state; negative control: applying it changes the
  conv2d);
- **parameters: MDv8 block 169 216 (total 2 501 251, +169 216 vs B; MD8 block 116 736, so +52 480 vs MD8)**, MDv32 block
  55 936; equal to the analytic count and to the official module + the block LayerNorm, same state-dict shapes;
- **fidelity**: transcription of the official `SS2D` (verbatim `__init__` / `forward_core` / `forward`, in the test file
  only, with attribution) loaded with our weights: **SS2D outputs bit-identical (max |diff| 0) at 8x8 and 32x32**, input
  gradients equal within 1e-5, full block equal to `x + SS2D_official(LN(x))`; A and D initial values identical.

**Control run (pre-registered now, before running it).** `queues/control-mdv8.txt`: MDv8, seed 0, 17 epochs, grouped
split, out root `~/runs/control-2026-10-14`, with `B` linked to `~/runs/control-2026-10-12/B` as the reference.
Criteria of 2026-10-12 (`scripts/control_gates.py`): (a) no NaN / Inf, (b) mean train L1 of the last 3 epochs < first 3,
(c) final val L1 <= 1.15 x B's (0.006324). Purpose: faults only; the number is not used for comparisons. it/s measured
(estimate of the night durations). If not passed by 20:00, MDv8 is left out of tonight's queue.

---

## 2026-10-14 — MDv8 control run: pass; Night 4: queue (pre-registered, not launched)

**Control run.** `scripts/run_queue.py queues/control-mdv8.txt --out-root ~/runs/control-2026-10-14 --cutoff 20:00` at
commit `82ed832` (clean tree), 13:34-13:36 local time; `python scripts/control_gates.py --root ~/runs/control-2026-10-14
--csv docs/results/control-mdv8.csv` (B = link to the 2026-10-12 control B):

| Run  | Params    | it/s  | Train L1 first 3 / last 3 | Val L1 final | / B   | a | b | c | Result |
|------|-----------|-------|---------------------------|--------------|-------|---|---|---|--------|
| B    | 2 332 035 | 115.0 | 0.023897 / 0.006655       | 0.006324     | 1.000 | Y | Y | Y | pass   |
| MDv8 | 2 501 251 | 92.1  | 0.023220 / 0.006537       | 0.006338     | 1.002 | Y | Y | Y | pass   |

Peak VRAM allocated 182 MiB (MD8 179). As pre-registered, the val number is not used for comparisons. Estimated
133-epoch wall time (training loop + validation x 133/17 + fixed overhead + evaluation): 15.1 min, the same as MD8's
estimate from its control run (15.1; MD8's real night-3 times were 14.7-15.5).

**Night 4 queue** `queues/night4.txt` (26 runs, order fixed): MDv8 seeds 0-7 (133 epochs, grouped split, in seed order,
`est=16m`), then the 18 data-quantity runs, lines identical to `queues/data-eff.txt` (per seed 0, 1, 2: B, MD8, SA8 at
0.25, then at 0.5). Out root `~/runs/night4` (the data-quantity runs go here, not to `~/runs/data-eff`, so one launcher
and one cutoff serve both). `est=` total 386 min; with real durations (MDv8 15.1, B 12.1, MD8 14.8, SA8 12.6 min; the
step-budget runs have the same number of steps and validation passes as the full runs) ~358 min. **Expected ends with a
start at 22:30:** MDv8 s0-s7 00:31; data-quantity seed 0 01:50, seed 1 03:09, seed 2 **04:28**; worst case with every run
at its `est=`: 04:56. Cutoff 08:30: all 26 runs are expected to fit.

**Launch.** `scripts/launch_night4.sh` (copy of `launch_night3.sh`, defaults `queues/night4.txt`, `~/runs/night4`, cutoff
08:30, tmux session `night4`): refuses to start on a dirty tree or if the session exists. Checked today: on the dirty
tree it exits 1 listing the files, no session created; `run_queue.parse_queue` reads 26 runs.

**After the night (planned).** MDv8: validation analysis as phase 4 (`final_analysis.py` / `analyze_val.py` with the
label MDv8; comparisons MDv8 - MD8 and MDv8 - B are already in the scripts), then the second, separate test pass decided
in advance. Data-quantity runs: validation only.

---

## 2026-10-14 — Exploratory post-test: night / commit effect and between-seed spread on test

**EXPLORATORY POST-TEST** (decided after the test results; not part of the pre-registered analysis and not changing
it). No training, no GPU: per-sample CSVs already computed by `evaluate.py`, run configs, git history; the test inputs
are read on the CPU for sample properties and one figure (CPU inference of 4 MD8 runs on 3 samples, Mamba reference
path). New `scripts/posttest_explore.py`, outputs `docs/results/posttest/` (`summary.md` has every table) and
`docs/figures/posttest/md8_top3_samples.png`.

### 1. Night / commit effect

**Why.** On validation B seeds 3-7 were slightly worse than seeds 0-2 (~2 SE on rel_all); on test MD8 splits into two
groups (five runs >= 3.03, three 2.54-2.74).

**MD8 groups on test (rel_all): low = seeds 6, 3, 1 (2.545 / 2.724 / 2.744; nights 3, 3, 2); high = seeds 5, 4, 0, 2, 7
(3.027-3.190; nights 3, 3, 2, 2, 3).** Both groups mix nights 2 and 3, so the split is not by night / commit.

**Seeds 3-7 minus seeds 0-2** (points; diff / Welch SE):

| config  | val rel_all    | val rel_p      | val rel_vy     | test rel_all   | test rel_p     | test rel_vy    |
|---------|----------------|----------------|----------------|----------------|----------------|----------------|
| B       | +0.033 (+1.97) | +0.163 (+0.85) | +0.121 (+0.98) | -0.111 (-2.29) | -0.926 (-3.06) | -0.275 (-2.89) |
| MD8     | +0.007 (+0.18) | +0.402 (+1.57) | +0.035 (+0.30) | -0.071 (-0.42) | -0.354 (-0.46) | -0.165 (-0.53) |
| SA8     | +0.030 (+1.46) | +0.041 (+0.14) | -0.101 (-1.01) | -0.232 (-1.49) | -1.689 (-1.61) | -0.780 (-2.23) |
| MD32    | +0.030 (+1.59) | +0.338 (+2.36) | +0.066 (+0.52) | +0.055 (+0.86) | +1.920 (+3.24) | +0.473 (+2.47) |
| SA32    | -0.022 (-0.57) | +0.050 (+0.20) | -0.015 (-0.11) | -0.200 (-1.34) | -1.109 (-1.37) | -0.461 (-1.22) |
| B-large | -0.000 (-0.01) | +0.100 (+0.34) | +0.098 (+1.14) | +0.001 (+0.01) | +0.134 (+0.21) | +0.080 (+0.41) |

Mean diff/SE over the 6 configurations (number positive): val rel_all +0.77 (4/6), rel_p +0.91 (6/6), rel_vy +0.31 (4/6);
test rel_all -0.78 (2/6), rel_p -0.51 (2/6), rel_vy -0.67 (2/6). For B the sign flips between validation (3-7 worse) and
test (3-7 better); MD32 goes the other way on test. Limits: the configurations share the same validation / test samples
and the same seed indices, so the 36 ratios are not independent; with 3 vs 5 runs the SEs are rough; and night is
confounded with the seed index (0-2 vs 3-7), so a night effect cannot be separated from which seeds were drawn.

**Code and environment between the commits actually used** (night 1 `3e69402`: 3 runs; night 2 `ee7ee46`: 18; night 3
`7b663e2`: 30; all clean trees). `git diff` of `scripts/train.py` and every module it imports directly or indirectly
(`mdf/data.py`, `mdf/models.py`, `mdf/runinfo.py`, `mdf/blocks.py`, `mdf/mamba_ref.py`, `mdf/__init__.py`,
`train/DfpNet.py`, `train/utils.py`):
- 3e69402 -> ee7ee46: variant factory added; B is built by `build_from_cfg` -> `build_dfpnet(expo=log2(32) = 5.0)` instead
  of `build_model("dfpnet", expo=5)`: same channel count (int(2**5.0 + 0.5) = 32), same initial weights and outputs at
  the same seed (unit test 2, `torch.equal`); `models.py` imports `blocks.py`, which imports `mamba_ssm` only inside
  `SS2DMamba.__init__` (not for B). No effect on B's training.
- ee7ee46 -> 7b663e2: k-fold split code only (`data.py`, argument parsing); the grouped split is unchanged (same train
  and validation sha256 in every config). No effect.
- Searched in all three commits and in HEAD: `allow_tf32`, `cudnn.benchmark` / `deterministic`,
  `set_float32_matmul_precision`, `use_deterministic_algorithms`, `num_workers`, `worker_init`, all seeding calls: the
  only occurrences are `runinfo.seed_everything` (random, numpy, torch, cuda seeds; cudnn flags only with
  `--deterministic`) and the DataLoader settings (shuffle, drop_last, num_workers 0, generator seeded with `--seed`),
  identical in all commits. No such setting in the installed `mamba_ssm` / `causal_conv1d` packages.
- Recorded in all 51 `config.json`: identical train / validation file lists (sha256), epochs 133, 599 batches, 79 667
  iterations, optimiser, LR schedule, normalisation, DataLoader, batch size, loss, cudnn deterministic / benchmark both
  false, and identical environment (Python 3.12.15, torch 2.10.0+cu128, cuDNN 91002, numpy 2.5.3, driver 615.71.09, same
  GPU).
- **The current code (night 4) has the same seeding, DataLoader and global settings**; its changes since `7b663e2` only add
  the MDv variant and the step-budget branch (the epoch loop is unchanged, entry "Data-quantity experiment"). Nothing to
  flag.

**Conclusion (neutral).** No systematic night or commit effect is visible: code paths, settings and environment are the
same; the seed-group differences change sign between validation and test and between configurations, and MD8's two test
groups mix nights. The differences are consistent with seed-to-seed variability (non-deterministic kernels, 6b, and
different seeds). Weight on the comparisons: B's seeds 0-2 come from night 1 and the variants' from night 2, but since
nothing differs between the commits for these runs, this does not bias the comparisons beyond seed noise.

### 2. Between-seed spread on test

**Variance ratio of rel_all, variant vs B** (F = var_variant / var_B, 8 vs 8 seeds unless stated, two-sided p of the F
distribution assuming normal per-run values; seed-bootstrap 95 % CI of the SD ratio; F CDF computed with the regularised
incomplete beta, checked against simulation):

| split | config  | SD ratio | F    | p     | SD ratio CI   |
|-------|---------|----------|------|-------|---------------|
| val   | MD8     | 1.86     | 3.48 | 0.12  | [1.10, 3.20]  |
| val   | SA8     | 1.33     | 1.76 | 0.47  | [0.48, 2.43]  |
| val   | B-large | 1.73     | 3.00 | 0.17  | [0.92, 3.05]  |
| test  | MD8     | 2.77     | 7.70 | 0.015 | [1.30, 5.73]  |
| test  | SA8     | 2.79     | 7.76 | 0.015 | [1.12, 5.99]  |
| test  | SA32    | 2.31     | 5.32 | 0.042 | [1.13, 4.87]  |
| test  | MD32    | 1.23     | 1.52 | 0.59  | [0.60, 2.63]  |
| test  | B-large | 1.56     | 2.44 | 0.26  | [0.64, 3.30]  |
| test  | CBAM8 (3 seeds) | 1.81 | 3.29 | 0.20 | -          |

On test the larger spread of MD8, SA8 and SA32 is unlikely under equal variances taken one at a time (p 0.015-0.04;
6 comparisons x 2 splits, no correction); B-large (no block) is not distinguishable from B.

**Where it comes from** (covariance decomposition of the between-seed variance of mean test rel_all into per-sample
shares). MD8: the top 5 samples carry 73 % of the variance, the top 3 profiles (`mue139`, `bw3`, `mh49`) 71 %; SA8: top 5
samples 76 %, top 3 profiles (`mue139`, `mh49`, `bw3`) 92 %. The two largest shares for both are
`mue139_9021_-3726` and `mue139_8935_-3679` (20 % and 18-20 % each): |alpha| 22.4 deg (the maximum of the test set),
600-630 target reverse-flow cells downstream of the trailing edge (separated flow, as `pfcm_6703` and `rae5214` on
validation), per-sample rel_all ~13-28 % against a test mean of ~2.8 %, and nearly identical flow conditions (their errors
move together). The other large shares are also high-angle, separated or near-separated samples (`bw3_*` 20.5-21.2 deg,
`m17` 22.3 deg, `goe147` 17.3 deg), plus the low-angle `mh49_1733_62` (2.1 deg, small pressure signal). 11 of the 90 test
samples have > 100 reverse-flow cells. Figure: in the `mue139` samples the target has a detached low-pressure vortex
behind the trailing edge; no seed of MD8, good or bad, reproduces it (all predict a low-pressure region attached to the
suction side); good and bad seeds differ in how far they are from it.

**Sensitivity (post hoc): test rel_all without the profile `mue139` (3 of 90 samples)**, mean (sd over seeds), %:
B 2.414 (0.119), MD8 2.555 (0.164), SA8 2.527 (0.188), MD32 2.306 (0.108), SA32 2.383 (0.165), B-large 2.453 (0.101),
CBAM8 2.546 (0.121); with all samples B 2.776 (0.085), MD8 2.940 (0.234). Without `mue139`, SD ratios vs B are 0.85-1.58
(p >= 0.25): **the larger test spread of the block variants is mostly produced by the two near-duplicate `mue139`
samples**, where the per-seed error of the variants ranges ~6-31 % (B 16-23 %). The test mean rel_all itself is shifted by
~0.36-0.43 points by these 3 samples for every configuration.

**Open questions.** Whether these RANS targets at the highest angles are converged steady solutions (open since the
pressure entry). The test has 30 profiles; one profile with three near-duplicate separated-flow samples weighs ~13 % of
the mean rel_all.

---

## 2026-10-14 — Correction to the entry "Exploratory post-test: night / commit effect and between-seed spread on test"

The sentence "The test mean rel_all itself is shifted by ~0.36-0.43 points by these 3 samples for every configuration"
is inaccurate. From the sensitivity table the shift (all samples minus without `mue139`) is: B 0.36, MD8 0.39, SA8 0.21,
MD32 0.43, SA32 0.43, B-large 0.44, CBAM8 0.17 points. Also, "per-seed error of the variants ranges ~6-31 %" should read
~5-31 % (SA8 5.5-22.3, MD8 13.3-28.1, MD32 13.4-28.9, SA32 17.6-30.9, B-large 16.2-30.8 %; B 16.3-22.9 %), i.e. B-large,
without a block, also spreads widely on these samples. The entry is not rewritten.

---

## 2026-10-15 — Night 4: run check (status and durations only)

**Action.** Read `~/runs/night4/queue.log`, every run's `config.json` and `queue_timing.json`; searched the logs for
`Traceback` / `Error:` / nan / inf and `metrics.csv` for NaN / Inf.

**Evidence.** Queue started **2026-10-14 18:07:22** (not 22:30 as planned in the entry "MDv8 control run: pass; Night 4":
earlier launch), ended **2026-10-15 00:03:21**, before the cutoff (08:30). **26 / 26 runs finished and evaluated on
validation** (MDv8 s0-7, 18 data-quantity runs), none interrupted, no `*.interrupted-*` directory, no STOP file, no
error, no NaN / Inf. All at commit `f3bfc17`, clean tree. Training wall times, min (`est=` in the queue): MDv8 14.6-14.7
(16; control-run estimate 15.1), B 12.0 (13), MD8 14.7-14.8 (16), SA8 12.6-12.7 (14); the step-budget runs took the same
time as the full runs of the same configuration, as expected (same steps and validation passes). Evaluation 3-7 s.

---

## 2026-10-15 — MDv8 on validation (EXPLORATORY, as pre-registered)

**Action.** `python scripts/final_analysis.py --config B=~/runs/night1/ep133_s* --config B=~/runs/night3/B_s* --config
MD8=~/runs/night2/MD8_s* --config MD8=~/runs/night3/MD8_s* --config MDv8=~/runs/night4/MDv8_s* --seeds 0-7 --split val
--out-dir docs/results/val-mdv8 --fig-dir docs/figures/val-mdv8`, and `scripts/train_fit.py` with the same configs
(`docs/results/train-fit-mdv8/`; its "pre-registered condition" section is the 5a condition on MD8, repeated, not new).
Consistency vs `eval_val_samples.csv`: max |diff| 1.5e-7. Test set not used.

**Evidence: T1** (validation, %, mean (std) over 8 seeds):

| config | rel_all         | rel_p            | rel_vx          | rel_vy          | agg rel_vy | median rel_vy |
|--------|-----------------|------------------|-----------------|-----------------|------------|---------------|
| B      | 2.4416 (0.0280) | 12.5225 (0.2753) | 1.5630 (0.0185) | 6.0674 (0.2044) | 5.660      | 4.279         |
| MD8    | 2.4363 (0.0521) | 12.3436 (0.3404) | 1.5704 (0.0433) | 5.8278 (0.1448) | 5.598      | 4.067         |
| MDv8   | 2.4945 (0.0321) | 12.4265 (0.2460) | 1.6089 (0.0254) | 5.9792 (0.2235) | 5.759      | 4.101         |

**T2** (points, two-level 95 % CI):
- MDv8 - MD8: rel_all +0.058 [-0.028, +0.143]; rel_p +0.08 [-0.33, +0.49]; rel_vx +0.038 [-0.028, +0.101]; rel_vy
  +0.15 [-0.06, +0.38]; fluid MAE p / vx / vy +0.06 / +0.17 / +0.12 x1e-3, all CIs include 0. (One-level CIs, profiles
  only, exclude 0 for rel_all, rel_vy, MAE vy.)
- MDv8 - B: rel_all +0.053 [-0.039, +0.154]; rel_p -0.10 [-0.59, +0.38]; rel_vx +0.046 [-0.021, +0.121]; rel_vy -0.09
  [-0.35, +0.16]; MAE all CIs include 0.
- T2b: no two-level interval excludes 0 (MDv8 - MD8 agg rel_vy +0.16 [-0.05, +0.37]). Regions: none excludes 0.
  Divergence: MDv8 - B other fluid -0.025 [-0.045, -0.004] (as MD8 - B), wall band includes 0.

**Training error in eval mode** (x1e-3, mean (std); diff / Welch SE vs B):

| config | train L1        | val L1          | gap             | train - B       | val - B         |
|--------|-----------------|-----------------|-----------------|-----------------|-----------------|
| B      | 2.7623 (0.1065) | 4.8733 (0.0611) | 2.1109 (0.1225) |                 |                 |
| MD8    | 2.4756 (0.1746) | 4.8721 (0.1103) | 2.3965 (0.1593) | -0.287 (-3.97)  | -0.001 (-0.03)  |
| MDv8   | 2.3162 (0.0588) | 4.9900 (0.0635) | 2.6738 (0.0836) | -0.446 (-10.37) | +0.117 (+3.75)  |

**Reading (neutral).** On validation MDv8 is not distinguishable from MD8 or from B on any pre-registered metric with
the two-level interval; its point estimates are slightly worse than MD8 on every metric. It fits the training set
better than both (lowest train L1 of all configurations so far, smallest seed spread) and has the largest train /
validation gap. The faithful SS2D therefore does not change the picture of the simplified block on validation. Limit
(declared): MDv8 has 52 480 more parameters than MD8.

**Open questions.** Test set: second pre-registered pass, next entry.

---

## 2026-10-15 — TEST SET: MDv8, second pre-registered pass

**Context / why.** Second, separate test pass decided in advance (entry "Pre-registered ablation: faithful SS2D (MDv8)",
commit `26b6e7f`): MDv8 is evaluated on the test set whatever its validation result, with the plan of section 2
(comparisons MDv8 vs MD8 and MDv8 vs B, same metrics, two-level bootstrap). Done after the validation analysis
(previous entry), which does not change the plan.

**Action.** Checks: tree clean, HEAD `ed7637e` descends from `26b6e7f`. **One pass**: `scripts/evaluate.py --run-dir
~/runs/night4/MDv8_s<k> --split test` (batch 1) for k = 0..7, one run at a time, 8 / 8 evaluated, 90 samples each; only
the MDv8 runs are new on test (B and MD8 test values recomputed from the cached inference, identical to the first
pass). Then `scripts/final_analysis.py <same --config as the previous entry> --seeds 0-7 --split test --allow-test
--out-dir docs/results/test-mdv8 --fig-dir docs/figures/test-mdv8` (script unchanged since the first pass).
Consistency vs `eval_test_samples.csv`: max |diff| 2.6e-7. Committed right after the computation, no re-runs.

**Evidence: T1** (test, %, mean (std) over 8 seeds; MAE x1e-3):

| config | rel_all       | rel_p         | rel_vx        | rel_vy        | agg rel_p | median rel_p | agg rel_vy | median rel_vy | MAE p / vx / vy    |
|--------|---------------|---------------|---------------|---------------|-----------|--------------|------------|---------------|--------------------|
| B      | 2.776 (0.085) | 14.91 (0.64)  | 1.797 (0.068) | 6.64 (0.20)   | 13.74     | 8.53         | 6.24       | 4.13          | 3.78 / 7.90 / 5.03 |
| MD8    | 2.940 (0.235) | 15.20 (1.20)  | 1.937 (0.168) | 6.61 (0.47)   | 14.67     | 8.10         | 6.51       | 3.95          | 4.05 / 8.51 / 5.25 |
| MDv8   | 2.767 (0.174) | 14.88 (0.99)  | 1.772 (0.149) | 6.65 (0.32)   | 13.59     | 8.15         | 6.24       | 4.17          | 3.74 / 7.83 / 5.03 |

Per-run MDv8 rel_all on test: 2.43, 2.73, 3.02, 2.88, 2.79, 2.83, 2.64, 2.82 (seeds 0-7).

**Comparisons** (points; two-level 95 % CI):
- MDv8 - MD8: rel_all -0.17 [-0.64, +0.14]; rel_p -0.31 [-2.06, +1.41]; rel_vx -0.17 [-0.53, +0.07]; rel_vy +0.04
  [-0.58, +0.68]; agg rel_p -1.08 [-3.50, +0.62]; median rel_p -0.05 [-0.72, +1.32]; agg rel_vy -0.27 [-1.21, +0.44];
  median rel_vy +0.36 [-0.37, +0.91]; fluid MAE p / vx / vy -0.30 / -0.68 / -0.22 x1e-3, all CIs include 0.
- MDv8 - B: rel_all -0.009 [-0.34, +0.26]; rel_p -0.03 [-1.49, +1.43]; rel_vx -0.03 [-0.28, +0.17]; rel_vy +0.004
  [-0.52, +0.50]; agg / median and MAE all include 0.
- **No two-level interval excludes 0, for any metric, in either comparison.** Regions: none excludes 0. Divergence:
  MDv8 - B other fluid -0.034 [-0.063, -0.005] (as on validation); the rest include 0.

**Reading (neutral).** On test MDv8's mean equals B's (2.767 vs 2.776) and is below MD8's (2.940), with intervals
including 0; its seed spread (0.174) is ~2x B's, between B and MD8. On validation the order was the reverse (MDv8 slightly
above MD8 and B). As for the first pass, the ranking changes between validation and test and no difference is
established with 8 seeds and 30 test profiles. Test results final: no re-runs.

---

## 2026-10-15 — Data-quantity experiment: results (validation only, EXPLORATORY)

**Context / why.** Analysis as pre-registered (commit `26b6e7f`, section 1; entry "Data-quantity experiment:
implementation and queue"). Nothing in the design changed after the runs.

**Action.** New `scripts/data_quantity.py` (reuses `train_fit.train_metrics` / `welch`, `explore.cached_inference`,
`explore.paired_bootstrap`): groups 29.1 % (`night4/*_f25_s0-2`), 55.6 % (`*_f50_*`) and 100 % (the main runs with the
same seeds: B `night1/ep133_s0-2`, MD8 / SA8 `night2/*_s0-2`), configurations B, MD8, SA8. Train L1 in eval mode on the
training files each run actually used: list rebuilt with `subsample_per_profile` and its sha256 asserted equal to the
run's `config.json` (27 / 27 equal). `python scripts/data_quantity.py --out-dir docs/results/data-quantity --fig-dir
docs/figures/data-quantity` (~2 min GPU). Test set not used.

**Evidence** (validation; mean (sd) over seeds 0-2; L1 x1e-3; full tables in `docs/results/data-quantity/summary.md`):

| data   | config | rel_all %     | rel_p %        | rel_vy %      | val L1 final (csv) | train L1 (eval) | gap          |
|--------|--------|---------------|----------------|---------------|--------------------|-----------------|--------------|
| 29.1 % | B      | 2.608 (0.074) | 13.49 (0.12)   | 6.35 (0.09)   | 5.19 (0.15)        | 1.94 (0.06)     | 3.26 (0.16)  |
| 29.1 % | MD8    | 2.567 (0.089) | 13.24 (0.45)   | 6.20 (0.25)   | 5.11 (0.18)        | 1.66 (0.05)     | 3.46 (0.16)  |
| 29.1 % | SA8    | 2.665 (0.028) | 13.64 (0.22)   | 6.55 (0.08)   | 5.30 (0.05)        | 1.87 (0.06)     | 3.44 (0.09)  |
| 55.6 % | B      | 2.535 (0.052) | 13.09 (0.14)   | 6.06 (0.14)   | 5.06 (0.10)        | 2.28 (0.06)     | 2.78 (0.09)  |
| 55.6 % | MD8    | 2.483 (0.063) | 12.69 (0.31)   | 5.90 (0.06)   | 4.95 (0.14)        | 2.06 (0.05)     | 2.90 (0.09)  |
| 55.6 % | SA8    | 2.553 (0.119) | 13.14 (0.31)   | 6.26 (0.14)   | 5.09 (0.25)        | 2.13 (0.12)     | 2.96 (0.19)  |
| 100 %  | B      | 2.421 (0.021) | 12.42 (0.23)   | 5.99 (0.09)   | 4.82 (0.05)        | 2.71 (0.03)     | 2.12 (0.07)  |
| 100 %  | MD8    | 2.432 (0.050) | 12.09 (0.41)   | 5.81 (0.17)   | 4.86 (0.10)        | 2.48 (0.03)     | 2.39 (0.12)  |
| 100 %  | SA8    | 2.486 (0.008) | 12.77 (0.41)   | 6.23 (0.07)   | 4.95 (0.01)        | 2.53 (0.09)     | 2.42 (0.08)  |

Differences (points, two-level 95 % CI; L1 rows: Welch diff / SE; **3 seeds: rough**):

| data   | MD8 - B rel_all         | MD8 - B rel_vy        | MD8 - B gap (diff/SE) | SA8 - B rel_all         | SA8 - B rel_vy        | SA8 - B gap (diff/SE) |
|--------|-------------------------|-----------------------|-----------------------|-------------------------|-----------------------|-----------------------|
| 29.1 % | -0.042 [-0.191, +0.116] | -0.15 [-0.49, +0.21]  | +0.20 (+1.50)         | +0.057 [-0.079, +0.206] | +0.20 [-0.11, +0.49]  | +0.18 (+1.68)         |
| 55.6 % | -0.053 [-0.204, +0.102] | -0.15 [-0.44, +0.13]  | +0.12 (+1.63)         | +0.018 [-0.145, +0.178] | +0.20 [-0.06, +0.46]  | +0.18 (+1.47)         |
| 100 %  | +0.011 [-0.085, +0.142] | -0.19 [-0.47, +0.06]  | +0.27 (+3.36)         | +0.064 [-0.023, +0.151] | +0.24 [+0.03, +0.47]  | +0.30 (+4.96)         |

MD8 - B rel_p: -0.25 / -0.40 / -0.33 (all two-level CIs include 0). Train L1, MD8 - B: diff/SE -6.3 / -4.9 / -9.7;
SA8 - B: -1.3 / -1.9 / -3.3. Validation curves (diagnostic): for every group the minimum of the val-L1 curve is in the
last third (blocks 100-132 of 133), and the curves are flat over the last ~20 blocks, at 29 % as well; no late rise
visible (figure `docs/figures/data-quantity/val_curves.png`).

**Answer to the pre-registered question (neutral).** With less data all three configurations have higher validation
error, lower training error and a larger gap (B gap 3.26 / 2.78 / 2.12 at 29 / 56 / 100 %). The MD8 - B and SA8 - B
differences keep their sign at every fraction for rel_p and rel_vy (MD8 lower, SA8 higher); for rel_all MD8 - B is
-0.04 / -0.05 / +0.01, i.e. slightly negative at reduced data and ~0 at full data, with every two-level interval
including 0. The excess gap of the block variants over B is present at all fractions (+0.12 to +0.30 x1e-3) and is not
larger at reduced data (the largest diff/SE is at 100 %, where the seed spreads are smallest). With 3 seeds none of these
changes across fractions is established. Limits as pre-registered (geometric diversity not varied, low power).

---

## 2026-10-15 — Multi-level models (implementation, tests); control runs for a position experiment pre-registered

**Context / why.** Preparation of a possible exploratory experiment on the position of the faithful SS2D block (MDv at
32x32, at 64x64, and at several levels at once), requested by the team after the MDv8 results. Its composition and
pre-registration are decided later by the team; here only code, tests and 17-epoch control runs (no long runs).

**Implementation.**
- `mdf/models.py`: configuration key `levels` (>= 2 distinct `LEVELS`, stored sorted descending; `normalize_levels`,
  `levels_tag` -> e.g. `8-16-32`). Multi-level models are a separate class `TurbNetGMultiMixer`: `weights_init` on the
  skeleton first (so the skeleton consumes the RNG exactly as B), then one block per level in encoder order (largest
  map first), in `mixers` (`ModuleDict` keyed by level); each block output replaces its layer's output, as for one
  block. Every block has a zero-initialised output projection, so at initialisation the network equals B.
  Single-level variants are untouched: same class `TurbNetGMixer`, attribute `mixer`, `normalize_cfg` output without
  a `levels` key. `model_cfg_from_run_config` reads `levels`; `block_params` sums all blocks.
- `scripts/train.py`: `--levels 8,16,32` (needs a block variant, excludes `--level`; error otherwise), run tag
  `MDv8-16-32`, `config.json` key `levels` (null for single-level runs).
- Alternative considered: generalise `TurbNetGMixer` to a list of levels; rejected, since it would rename `mixer` in
  the state dicts of all existing checkpoints.

**Tests** `tests/test_multilevel.py`: **48 passed** (full suite **126 passed, 6 skipped**):
- shapes (2, 3, 128, 128), finite, for all 11 combinations of >= 2 levels of {8, 16, 32, 64}, MDv and MD;
- identity at initialisation: skeleton state dict equal to B's at the same seed, output max |diff| vs B 1.49e-8 (MDv,
  MD, SA; levels 8+16+32, 32+64, 8+16+32+64);
- after two Adam steps every block parameter has a non-zero gradient (MDv and MD, 8+16+32);
- parameter counts = sum of the single-level blocks: **MDv8-16-32 block 281 088 (169 216 + 55 936 + 55 936), total
  2 613 123** (+281 088 vs B); MDv32-64 76 736; MD8-16-32 182 272. MDv64 block 20 800, MDv32 55 936;
- existing variants unchanged: sha256 fingerprints of the initial state dicts (seed 0) of B, B-large, CBAM8, SA8, MD8,
  MDv8, SA32, MD32, MDv32, MDv64, MD64 computed with the code before the change (git worktree of `686398f`) and
  after: identical; hard-coded in the test;
- configuration handling and a strict state-dict round trip.

**Control runs (pre-registered now, before running them).** `queues/control-night5.txt`: MDv32 (`--level 32`), MDv64
(`--level 64`, 4096 tokens, C = 32), MDv8-16-32 (`--levels 8,16,32`); seed 0, 17 epochs, grouped split, out root
`~/runs/control-2026-10-15` with `B` linked to `~/runs/control-2026-10-12/B`. Criteria of 2026-10-12
(`scripts/control_gates.py`): (a) no NaN / Inf, (b) mean train L1 of the last 3 epochs < first 3, (c) final val L1 <=
1.15 x B's (0.006324). Purpose: faults only; the val numbers are not used for comparisons. it/s and peak VRAM measured
(night durations; MDv64 must fit in 8 GB).

---

## 2026-10-15 — Control runs MDv32, MDv64, MDv8-16-32: pass; Night 5 queue prepared (NOT launched, not pre-registered)

**Control runs.** `scripts/run_queue.py queues/control-night5.txt --out-root ~/runs/control-2026-10-15 --cutoff 08:30`
at commit `82b3c43` (clean tree), 01:11-01:25 local time; `python scripts/control_gates.py --root
~/runs/control-2026-10-15 --csv docs/results/control-night5.csv` (B = link to the 2026-10-12 control B):

| Run        | Params    | it/s  | Train L1 first 3 / last 3 | Val L1 final | / B   | a | b | c | Result | Peak VRAM alloc / reserved (MiB) |
|------------|-----------|-------|---------------------------|--------------|-------|---|---|---|--------|----------------------------------|
| B          | 2 332 035 | 115.0 | 0.023897 / 0.006655       | 0.006324     | 1.000 | Y | Y | Y | pass   |                                  |
| MDv32      | 2 387 971 | 53.9  | 0.023733 / 0.006639       | 0.006444     | 1.019 | Y | Y | Y | pass   | 282 / 316                        |
| MDv64      | 2 352 835 | 29.8  | 0.022099 / 0.006002       | 0.005951     | 0.941 | Y | Y | Y | pass   | 434 / 558                        |
| MDv8-16-32 | 2 613 123 | 44.3  | 0.023373 / 0.006374       | 0.006114     | 0.967 | Y | Y | Y | pass   | 332 / 386                        |

As pre-registered, the val numbers are not used for comparisons (17 epochs, one seed). MDv64 (4096 tokens) fits easily
(434 MiB allocated of 8 GB). For reference: MDv8 92.1 it/s / 182 MiB, MD32 64.7 it/s / 349 MiB.

**133-epoch estimates** (same formula as for MDv8: (training loop + validation) x 133 / 17 + data loading + setup +
process overhead + evaluation; it gave MDv8 15.1 min vs 14.6-14.7 real, MD32 21.4 vs 21.4-21.7 real): **MDv32 25.5 min,
MDv64 45.7 min, MDv8-16-32 31.0 min**; 102.2 min per seed.

**Queue** `queues/night5.txt` (9 runs: MDv32, MDv64, MDv8-16-32; seeds 0-2 in seed order, s0 of all three first;
`est=` 27 / 48 / 33 min = estimate + ~5 %; total `est=` 324 min, expected ~307 min). Out root `~/runs/night5`, cutoff
08:30. Launcher `scripts/launch_night5.sh` (copy of `launch_night4.sh`: session `night5`, refuses on a dirty tree or
an existing session; syntax checked with `bash -n`; `run_queue.parse_queue` reads 9 runs). **Not launched**: the
composition is decided by the team after the MDv8 results, and the pre-registration entry (drafted, not committed)
must be committed before the launch. With a start at 01:30 the expected ends would be: s0 03:12, s1 04:54, s2
**06:37**; worst case at `est=`: 06:54.

---

## 2026-10-15 — Position experiment (night 5): pre-registration (EXPLORATORY; decided AFTER the test set was opened)

**Context / why.** After the MDv8 results (validation and the second pre-registered test pass), the team decided to
look at the position of the faithful SS2D block. **This experiment was decided after the test set was opened** (first
test pass `ca067a4`, MDv8 second pass `98cbb84`); it is therefore exploratory, and its results can not be used as
confirmatory evidence of any configuration.

**Design (fixed now, before any run).**
- Queue composition confirmed by the team: `queues/night5.txt` as committed in `7844663`.
- Configurations: MDv32 (MDv block at 32x32, C = 64), MDv64 (at 64x64, C = 32, 4096 tokens), MDv8-16-32 (one MDv
  block at each of 8x8, 16x16, 32x32; entry "Multi-level models"). Every block with a zero-initialised output
  projection (identity with B at initialisation).
- Protocol identical to the main runs: 133 epochs, grouped split, Adam lr 6e-4, batch 10; seeds 0-2 (queue
  `queues/night5.txt`, in seed order); out root `~/runs/night5`, cutoff 08:30.
- **Validation only. These runs are NEVER evaluated on the test set** (no `evaluate.py --split test`, no
  `final_analysis.py --split test` with these labels).
- Comparisons (seeds 0-2 of each configuration against seeds 0-2 of the reference, same seeds; also against all 8
  seeds of the reference where available, reported separately): **MDv32 vs MD32**, **MDv32 vs B**, **MDv64 vs B**,
  **MDv8-16-32 vs MDv8**, **MDv8-16-32 vs B**. Metrics as in the final analysis plan (rel_all primary; rel_p with
  aggregate and median; rel_vx; rel_vy with aggregate and median; fluid MAE), two-level bootstrap (seeds + profiles,
  10 000), plus train L1 in eval mode and the train / validation gap (as verification 5a). With 3 seeds the intervals
  are rough; descriptive only, no correction for multiple comparisons.
- Main number: **final checkpoint**. Validation curves only as a diagnostic.
- **Seeds: fixed at 3 (0-2) and not increased**, whatever the results. Runs that do not start before the cutoff
  (08:30) are completed in the following night, with the same commands and no other runs added.
- **All runs are reported**, including failed or interrupted ones (with the reason); no run is repeated or dropped
  because of its result.
- The 17-epoch control runs (entry "Control runs MDv32, MDv64, MDv8-16-32: pass") serve only to find faults and to
  measure throughput and memory; their validation numbers do not shape expectations or decisions about this experiment.
- Limits: parameter counts differ (MDv32 +55 936, MDv64 +20 800, MDv8-16-32 +281 088 vs B); 3 seeds; validation already
  used for many decisions in this project.

---

## 2026-10-15 — Night 5: run check; analysis tooling for the position experiment

**Run check.** Read `~/runs/night5/queue.log`, every run's `config.json` / `queue_timing.json` / `metrics.csv`, and the
logs. Queue started **2026-10-15 01:34:14**, ended **06:32:49**, before the cutoff (08:30). **9 / 9 runs finished and
evaluated on validation** (MDv32, MDv64, MDv8-16-32, seeds 0-2), all at commit `72e5b1b` (the pre-registration
commit), clean tree; none interrupted, no `*.interrupted-*` directory, no STOP file, no `Traceback` / `Error:` in the
logs, no NaN / Inf in `metrics.csv` (133 rows each), no `eval_test*` file. Training wall times, min (`est=` in the queue;
estimate in the entry "Control runs MDv32, MDv64, MDv8-16-32: pass"): MDv32 24.7 (27; 25.5), MDv64 44.5-44.6 (48;
45.7), MDv8-16-32 29.9 (33; 31.0). Evaluation 7 s per run. Nothing to complete in a following night.

**Tooling.** `scripts/final_analysis.py`: new comparison group `position` = the five pre-registered comparisons
(MDv32 - MD32, MDv32 - B, MDv64 - B, MDv8-16-32 - MDv8, MDv8-16-32 - B), used in T2 / T2b / T3 / T4 only when the
labels are loaded (as `second pass`); the three labels appended to `ORDER`. The main / exploratory / second-pass groups
are unchanged, so earlier outputs are reproducible. `scripts/train_fit.py` unchanged. Test set not used (the runs are
never given `--split test` / `--allow-test`). Results: next entry.

---

## 2026-10-15 — Position experiment (night 5): results (validation only, EXPLORATORY)

**Context / why.** Analysis exactly as pre-registered (entry "Position experiment (night 5): pre-registration", commit
`72e5b1b`); nothing in the design changed after the runs. **Exploratory**: decided after the test set was opened; these
runs are never evaluated on test. All 9 runs are reported (none failed; previous entry). Main number: final checkpoint.

**Action** (commit `9b5ac01`, clean tree; configs: B = `night1/ep133_s*` + `night3/B_s*`, MD8 / MD32 = `night2` +
`night3`, MDv8 = `night4/MDv8_s*`, MDv32 / MDv64 / MDv8-16-32 = `night5`):
- `scripts/final_analysis.py <configs> --seeds 0-2 --split val` -> `docs/results/val-positions-s0-2/` (pre-registered
  main reading: 3 vs 3, same seeds) and without `--seeds` -> `docs/results/val-positions-all/` (reference with all 8
  seeds, reported separately). Consistency vs `eval_val_samples.csv`: max |diff| 1.5e-7. Bootstrap 10 000, seed 0.
- `scripts/train_fit.py <configs> [--seeds 0-2]` -> `docs/results/train-fit-positions-{s0-2,all}/` (GPU 128 s on the
  training set for 41 runs); its "pre-registered condition" section only repeats the 5a condition on MD8 (not new).
- New `scripts/positions_figure.py` (CPU): gap vs B in Welch SE (`docs/results/positions/gap_vs_B.md`) and the figure
  `docs/figures/positions/positions_summary.pdf` (IEEE single column, 3.5 in; (a) validation rel_all per seed and mean,
  (b) train L1 in eval mode vs validation L1; hollow markers = 3 seeds; grey B, vermillion MD, blue MDv, marker =
  position; palette checked with a CVD validator: MD / MDv pair passes, B is deliberately achromatic).
- Test set not read by any command.

**Evidence: T1** (validation, %, mean (std) over seeds; MDv32 / MDv64 / MDv8-16-32 have 3 seeds, the others 8):

| config     | seeds | rel_all         | rel_p            | rel_vx          | rel_vy          | agg rel_vy | median rel_vy |
|------------|-------|-----------------|------------------|-----------------|-----------------|------------|---------------|
| B          | 8     | 2.4416 (0.0280) | 12.5225 (0.2753) | 1.5630 (0.0185) | 6.0674 (0.2044) | 5.660      | 4.279         |
| MD8        | 8     | 2.4363 (0.0521) | 12.3436 (0.3404) | 1.5704 (0.0433) | 5.8278 (0.1448) | 5.598      | 4.067         |
| MDv8       | 8     | 2.4945 (0.0321) | 12.4265 (0.2460) | 1.6089 (0.0254) | 5.9792 (0.2235) | 5.759      | 4.101         |
| MD32       | 8     | 2.4668 (0.0294) | 12.4191 (0.2442) | 1.5958 (0.0250) | 5.8480 (0.1307) | 5.647      | 4.061         |
| MDv32      | 3     | 2.4366 (0.0249) | 12.0676 (0.1676) | 1.5779 (0.0251) | 5.7306 (0.1603) | 5.575      | 4.064         |
| MDv64      | 3     | 2.4144 (0.0215) | 12.1993 (0.5373) | 1.5755 (0.0164) | 5.3330 (0.0394) | 5.415      | 3.697         |
| MDv8-16-32 | 3     | 2.4102 (0.0849) | 11.8431 (0.4006) | 1.5554 (0.0619) | 5.6562 (0.2064) | 5.549      | 4.008         |

Full T1 (agg / median rel_p, fluid MAE) in the summaries. Per-run rel_all: MDv32 2.433 / 2.414 / 2.463,
MDv64 2.396 / 2.409 / 2.438, MDv8-16-32 2.505 / 2.383 / 2.342 (seeds 0-2).

**Pre-registered comparisons** (points; two-level 95 % CI; **3 seeds: rough; 3 vs 8 is rougher still** and mixes
seed sets):

| comparison        | rel_all, 3 vs 3          | rel_all, vs 8 seeds      | rel_p, 3 vs 3          | rel_vy, 3 vs 3           | rel_vy, vs 8 seeds       |
|-------------------|--------------------------|--------------------------|------------------------|--------------------------|--------------------------|
| MDv32 - MD32      | -0.011 [-0.094, +0.068]  | -0.030 [-0.114, +0.045]  | -0.14 [-0.67, +0.28]   | -0.08 [-0.39, +0.22]     | -0.12 [-0.36, +0.12]     |
| MDv32 - B         | +0.016 [-0.065, +0.100]  | -0.005 [-0.082, +0.070]  | -0.35 [-1.13, +0.27]   | -0.26 [-0.64, +0.05]     | **-0.34 [-0.70, -0.01]** |
| MDv64 - B         | -0.007 [-0.097, +0.115]  | -0.027 [-0.112, +0.083]  | -0.22 [-0.98, +0.46]   | **-0.66 [-0.96, -0.41]** | **-0.73 [-1.03, -0.47]** |
| MDv8-16-32 - MDv8 | -0.066 [-0.216, +0.104]  | -0.084 [-0.215, +0.080]  | -0.33 [-1.13, +0.41]   | -0.12 [-0.52, +0.20]     | -0.32 [-0.72, +0.02]     |
| MDv8-16-32 - B    | -0.011 [-0.157, +0.186]  | -0.031 [-0.177, +0.154]  | -0.58 [-1.54, +0.19]   | -0.34 [-0.79, +0.02]     | **-0.41 [-0.85, -0.06]** |

- rel_all, rel_p, rel_vx, fluid MAE: no two-level interval excludes 0 in either reading.
- T2b: only MDv64 - B median rel_vy excludes 0 (3 vs 3: -0.58 [-0.92, -0.28]; vs 8: -0.68 [-0.98, -0.34]); all
  aggregate rel_vy / rel_p intervals include 0 (MDv64 - B agg rel_vy -0.18 [-0.42, +0.12] 3 vs 3).
- Regions (descriptive): wall-band velocity MAE lower than the reference with intervals excluding 0 for MDv64 - B (vx
  -5.0 %, vy -8.5 %), MDv8-16-32 - MDv8 (vx -6.4 %, vy -5.7 %), MDv8-16-32 - B (vy -6.6 %), MDv32 - B (vy -3.2 %)
  (3 vs 3). Divergence error (3 vs 3) lower than B for all three new configurations in both regions with intervals
  excluding 0 (largest MDv64 - B wall -1.26 x1e-3; vs 8 seeds the same except MDv32 - B wall, which includes 0);
  MDv32 - MD32 is higher in the wall band (+0.24) and lower elsewhere (-0.03) 3 vs 3, both include 0 vs 8 seeds.

**Verification 5a** (train L1 in eval mode, x1e-3, mean (std); diff / Welch SE vs B, seeds 0-2 of both):

| config     | train L1        | val L1          | gap             | train - B       | val - B        | gap - B        |
|------------|-----------------|-----------------|-----------------|-----------------|----------------|----------------|
| B          | 2.7117 (0.0296) | 4.8275 (0.0443) | 2.1159 (0.0684) |                 |                |                |
| MD32       | 2.7240 (0.0580) | 4.8885 (0.0546) | 2.1644 (0.0325) | +0.012 (+0.33)  | +0.061 (+1.50) | +0.049 (+1.11) |
| MDv32      | 2.5708 (0.0572) | 4.8698 (0.0600) | 2.2990 (0.1081) | -0.141 (-3.79)  | +0.042 (+0.98) | +0.183 (+2.48) |
| MDv64      | 2.4682 (0.0829) | 4.8443 (0.0548) | 2.3761 (0.1244) | -0.243 (-4.79)  | +0.017 (+0.41) | +0.260 (+3.18) |
| MDv8       | 2.3026 (0.0593) | 4.9576 (0.0740) | 2.6549 (0.0965) | -0.409 (-10.69) | +0.130 (+2.61) | +0.539 (+7.90) |
| MDv8-16-32 | 2.1678 (0.0282) | 4.8303 (0.1832) | 2.6625 (0.1858) | -0.544 (-23.05) | +0.003 (+0.03) | +0.547 (+4.78) |

With all 8 B seeds the reading is the same (`docs/results/positions/gap_vs_B.md`): every MDv configuration fits the
training set better than B (train diff/SE -3.8 to -14.5) with validation L1 within 1 SE of B (MDv8 excepted, +3.75).

**Reading (neutral).** On the primary metric (rel_all) no position or number of MDv blocks is distinguishable from B or
from its reference, in either reading; the point estimates are within 0.09 points. The one consistent signal is on the
vertical velocity: MDv64 - B rel_vy (mean and median) excludes 0 in both readings, with a smaller wall-band velocity
error and divergence error; MDv32 and MDv8-16-32 have negative rel_vy differences whose intervals touch 0 depending on
the reading. Every MDv configuration fits the training data better than B, more so with the block at 8x8 or with three
blocks, and the train / validation gap grows accordingly, without a matching change in validation error. Cautions: 3
seeds; a validation rel_vy advantage of MD8 over B (-0.24, interval excluding 0) did not reproduce on test; parameter
counts differ; the validation set has been used for many decisions; the experiment was decided after the test set was
opened, so nothing here is confirmatory.

**Open questions.** Whether the rel_vy signal of MDv64 is more than seed noise would need more seeds or new data; the
pre-registration fixes the seeds at 3 and excludes the test set, so no follow-up is planned here.
