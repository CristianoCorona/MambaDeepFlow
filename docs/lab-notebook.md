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
