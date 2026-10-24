# Scripts

Every script is run from the repository root inside the `mdf` environment
(`conda run -n mdf python scripts/<name>.py ...`). `--help` lists all options; the module docstring at the top of
each file explains what it computes. Runs are written outside the repository (default `~/runs/<id>/`).

## Pipeline: train, evaluate, queue

| Script | What it does |
|---|---|
| `train.py` | Trains one model (`--variant`, `--level`, `--seed`, `--epochs`) with the original optimiser, loss and LR decay on the fixed split. Writes `config.json` (arguments, git commit, versions, timings), `metrics.csv` and `model.pt` to the run directory. |
| `evaluate.py` | Evaluates a run on `--split val` (or `test`, final results only): per-sample relative L1 errors of p, vx, vy as in `train/runTest.py`, saved as `eval_<split>.json` and `eval_<split>_samples.csv`. |
| `run_queue.py` | Runs a queue file from `queues/` one run at a time (training plus validation evaluation), with a wall-clock cutoff; finished runs are skipped. |
| `run_original.py` | Runs the unmodified `train/runTrain.py` with an explicit seed (reference for the reimplemented pipeline). |

## Checks

| Script | What it does |
|---|---|
| `check_env.py` | Environment smoke test: CUDA plus a Mamba forward / backward pass that must use the compiled `selective_scan_cuda` / `causal_conv1d_cuda` kernels. Prints `OK`. |
| `verify_loader.py` | Checks the float32 data loader (`mdf/data.py`) against the original float64 loader (`train/dataset.py`). |
| `make_split.py` | Writes / checks the committed validation lists in `splits/` (profile-disjoint, no profile in both sets). |

## Analysis behind the poster and report

| Script | What it does | Output |
|---|---|---|
| `final_analysis.py` | The pre-registered comparison of configurations: per-configuration means over seeds, paired differences with two-level (profile x seed) bootstrap 95 % intervals, region errors (wall band, wake), and the confirm / not-confirm decision of a pre-registered test (`--confirm-*`). | `docs/results/{val,test}-*` |
| `analyze_val.py` | Per-sample analysis helpers (also imported by the other analysis scripts). | `docs/results/val-explore*` |
| `train_fit.py` | Training error in eval mode vs validation error (does a block fit the training shapes better, or generalise better?). | `docs/results/train-fit*` |
| `cost.py` | Parameters, FLOPs, peak VRAM, latency and training speed per configuration. | `docs/results/cost-final` |
| `microbench.py` | Time and memory of a single block (Mamba vs attention) as the number of tokens grows. | `docs/results/microbench.csv` |
| `bigdata_analysis.py` | Analysis of the full-data-set experiment (runs synced from the cloud VM). | `docs/results/val-bigdata` |
| `poster_figures.py` | The poster figures (validation values from the analysis cache, test numbers read from `docs/results/test-*`). | `docs/figures/poster` |

The analysis scripts read the per-sample predictions through a cache (`~/runs/explore-cache`, `mdf/explore.py`),
so a second pass over the same runs needs no GPU.

## Cloud: full data set (`cloud/`)

Scripts for the full 53.8k-sample experiment on a rented GPU VM, in the order of `docs/cloud-guide.md`:
`setup_vm.sh` (environment, tests), `get_full_data.sh` (download, unpack), `inspect_full.py` (data checks),
`smoke_test.sh` + `estimate_budget.py` (short runs, time / cost estimate), `launch_bigdata.sh` (the queue
`queues/bigdata.txt`), `post_bigdata.py` (per-run post-processing on the VM), `sync_results.sh` (copy back);
`env.sh` holds the shared settings.

## Retired scripts

The night launchers, single test sessions, one-off checks and exploratory analyses used during the project are
not in the repository any more. They are at the git tag `experiments-final`
(`git show experiments-final:scripts/<name>`); `docs/lab-notebook.md` refers to them by their original path.
