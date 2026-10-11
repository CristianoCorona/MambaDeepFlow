# Project
Course project (Deep Machine Learning, Group 30): Mamba-augmented U-Nets for RANS airfoil flow prediction.
Base code: thunil/Deep-Flow-Prediction (Apache-2.0), remote `upstream`. Keep LICENSE and attribution.

# Environment
- Local only: Fedora, RTX 4060 (8 GB VRAM). No cloud unless explicitly requested.
- Isolated Python env named mdf; never touch system Python; ask before sudo.

# Rules
- Do NOT modify train/DfpNet.py, runTrain.py, runTest.py, dataset.py (reference baseline). New code goes in new files.
- Data lives outside the repo (symlinked into data/); never commit data, checkpoints, results.
- Every run: explicit --seed, same seeds across models, log config + git commit hash.
- Report evidence (command output, numbers), not just "done".

# Key facts
- DfpNet expo=5: 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 (bottleneck 1x1); channels 32/64/64/128/256/256/256.
- Candidate global-mixer positions: 8x8, 16x16, 32x32 (optionally 64x64).
- Repo defaults: 10k iterations, batch 10, lr 6e-4; paper: 80k iterations, lr 4e-4.
- Budget calibration (validation, docs/lab-notebook.md 2026-10-11): rule outcome 133 epochs
  (`--iterations 80000`, ~12 min/run); protocols are named by epochs (17 / 67 / 133).

# How to run
Full details (versions, checksums, measured numbers): docs/setup-log.md.

- Env: conda env `mdf` (conda-forge, Python 3.12, torch 2.10.0+cu128, causal-conv1d 1.7.0, mamba-ssm 2.3.2.post1).
  Non-interactive: `conda run -n mdf python <script>` (or `conda activate mdf` in an interactive shell).
  Recreate: `conda env create -f environment/conda-from-history.yml`, then
  `TMPDIR=$HOME/.cache/pip-tmp pip install -r environment/pip-freeze.txt` inside the env.
- TMPDIR: always `export TMPDIR=$HOME/.cache/pip-tmp` (`mkdir -p` it first) for pip installs and large temp files.
- Dataset: 6.4k set (`data_6k.tar.gz`, TUM dataserv) extracted outside the repo to `~/data/dfp/data/{train,test}`
  (6400 / 90 `.npz`). Symlinks: `data/train -> ~/data/dfp/data/train`, `data/test -> ~/data/dfp/data/test`
  (gitignored). Check: `find -L data/train -name '*.npz' | wc -l`.
- Env smoke test: `conda run -n mdf python scripts/check_env.py --seed 0`
  (exits non-zero unless the `selective_scan_cuda` / `causal_conv1d_cuda` kernels are actually used; expect `OK`).
- Original baseline: `cd train && python runTrain.py` then `python runTest.py` (~2 min train, ~440 MiB VRAM;
  random seed, so indicative only).
- Controlled pipeline (fixed split `splits/val_seed0.txt`, runs in `~/runs/<id>/`, see docs/lab-notebook.md):
  `python scripts/train.py --seed 0 --iterations 10000 [--out-dir DIR]`,
  `python scripts/evaluate.py --run-dir DIR --split val` (test split only for final results),
  original baseline with a seed: `python scripts/run_original.py --seed 0 [--out-dir DIR]`,
  checks: `scripts/verify_loader.py`, `scripts/compare_runs.py {curves,groups}`.

## Lessons learned
- `/tmp` is a tmpfs with a ~6 GB per-user quota. pip unpacks wheels there; the ~900 MB torch wheel plus
  leftovers from an interrupted install hit `OSError: [Errno 122] Disk quota exceeded`. Use TMPDIR above,
  and clean orphaned `/tmp/pip-*` dirs after an aborted install (`quota -s` shows usage).
- mamba-ssm / causal-conv1d are installed from prebuilt wheels downloaded from their GitHub releases
  (`cu12torch2.10cxx11abiTRUE-cp312`), kept in `~/data/wheels`, installed from local files. Do NOT build
  from source: system nvcc 12.6 needs GCC <= 13 but the system has GCC 16; no wheels exist for Python 3.14
  or torch >= 2.11 (hence Python 3.12 + torch 2.10). `pip install --dry-run` first to make sure torch/triton are not replaced.
- Large downloads can fail mid-way: use `pip install --resume-retries 20`.

## Lab notebook (standing rule)
- Keep docs/lab-notebook.md as a chronological log (newest entry last). Add an entry for every decision, experiment, change of plan or notable finding, BEFORE committing the related code, and commit it together with the code.
- Entry format: date (YYYY-MM-DD) and title; Context / why; Decision or action (with alternatives considered); Evidence (commands, key numbers, run IDs, commit hash); Open questions.
- Record negative results and abandoned approaches, with the reason.
- Never rewrite past entries; add a correction entry instead.
- English only. The repo is public: no personal paths, names or credentials.
- docs/setup-log.md remains the environment record; the notebook links to it.
