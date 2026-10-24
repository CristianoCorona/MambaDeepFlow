# Setup log: local environment and original-baseline smoke test

Date: 2026-10-09. Repo commit at time of runs: `1fd9822aa32bea7c9c83e60e613fd86fd34f9f93`
(original files in `train/` unmodified; `git diff -- train/` empty).

Scope: isolated Python env with torch + mamba-ssm + causal-conv1d, a Mamba smoke test,
the 6.4k dataset, and one unmodified run of `train/runTrain.py` + `train/runTest.py`.
No new models.

## 1. System

| Item | Value | Source |
|---|---|---|
| OS | Fedora release 44 (Forty Four) | `cat /etc/fedora-release` |
| Kernel | 7.2.8-200.fc44.x86_64 | `uname -r` |
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 threads | `lscpu`, `nproc` |
| RAM / swap | 14 GiB / 8.0 GiB | `free -h` |
| Disk (/home, btrfs) | 952G, 851G available (before installs; 841G after env, wheels and dataset) | `df -h ~` |
| /tmp | tmpfs 7.5G, **per-user quota 6111M** (`usrquota`) | `df -h /tmp`, `quota -s`, `findmnt -T /tmp` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, compute capability 8.9 | `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv` |
| Driver | 615.71.09, CUDA UMD Version 13.4 | `nvidia-smi` |
| Secure Boot | disabled | `mokutil --sb-state` |
| System gcc | 16.2.1 20260819 (Red Hat 16.2.1-2) | `gcc --version` |
| System nvcc | 12.6 (V12.6.85), `/usr/local/cuda-12.6` | `nvcc --version` |
| System python | 3.14.7 (not used) | `python3 --version` |
| conda / uv | conda 26.7.1 (`~/miniconda3`), uv 0.12.19 | `--version` |

Implications:
- No prebuilt mamba-ssm / causal-conv1d wheels exist for cp314 or torch >= 2.11, so the env uses Python 3.12 + torch 2.10.
- A source build with the system toolchain would fail: nvcc 12.6 accepts host GCC <= 13, the system has GCC 16.
  The source-build fallback (conda-forge `cuda-toolkit=12.8` + `gxx=13`, `MAX_JOBS=2`) was **not needed**.

## 2. Version choice

Prebuilt Linux x86_64 wheels exist for mamba-ssm 2.3.2.post1 and causal-conv1d 1.7.0 for
cp310-cp313 x torch 2.6-2.10 (cu12; cu13 from torch 2.9), cxx11abi TRUE.
Sources: <https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1>,
<https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0>.

Chosen: **Python 3.12, torch 2.10.0+cu128, causal-conv1d 1.7.0, mamba-ssm 2.3.2.post1**,
wheels downloaded explicitly and installed from local files (not via `setup.py` auto-download).

## 3. Environment installation (no sudo used)

```bash
conda create -y -n mdf --override-channels -c conda-forge python=3.12 pip
# -> Python 3.12.15, pip 26.2.1

# torch (TMPDIR moved to /home: see "Problems" below)
mkdir -p ~/.cache/pip-tmp
TMPDIR=$HOME/.cache/pip-tmp pip install --progress-bar off --resume-retries 20 \
    'torch==2.10.*' --index-url https://download.pytorch.org/whl/cu128
# -> torch-2.10.0+cu128, triton-3.6.0, nvidia-cudnn-cu12-9.10.2.21

TMPDIR=$HOME/.cache/pip-tmp pip install --progress-bar off numpy matplotlib
# -> numpy-2.5.3, matplotlib-3.11.2, pillow-12.3.0

# exact wheels from the GitHub releases
mkdir -p ~/data/wheels && cd ~/data/wheels
curl -sSL --fail -o causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
curl -sSL --fail -o mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  'https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
pip install --dry-run ./causal_conv1d-*.whl ./mamba_ssm-*.whl   # checked: does not replace torch/triton
TMPDIR=$HOME/.cache/pip-tmp pip install -v ./causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
TMPDIR=$HOME/.cache/pip-tmp pip install -v ./mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Wheel files (sizes match the GitHub API asset sizes):

| File | Bytes | sha256 |
|---|---|---|
| causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl | 196558813 | 8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7 |
| mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl | 322288410 | 7a67070c1e7e99c95abd1319623f044e8a1b3fb46f774bfdea949f0a4fc79638 |

Checks:
- Both installs: exit code 0 (echoed to the console; the pip logs end with
  `Successfully installed causal-conv1d-1.7.0` / `... mamba-ssm-2.3.2.post1`). `grep -ciE 'Building wheel|running build_ext|nvcc|Guessing wheel URL'`
  on both pip logs: 0 matches (no source build).
- After install, `pip freeze` still shows `torch==2.10.0+cu128`, `triton==3.6.0`.
- Extra dependencies pulled by mamba-ssm include `tilelang==0.1.8`, `quack-kernels==0.6.5`,
  `nvidia-cutlass-dsl==4.8.0` (cu12 libs), `einops==0.8.2`, `transformers==5.19.0`.

Verification command and output:
```
$ python -c "import torch;print(torch.__version__, torch.version.cuda, torch.compiled_with_cxx11_abi(), torch.cuda.is_available(), torch.cuda.get_device_name(0))"
2.10.0+cu128 12.8 True True NVIDIA GeForce RTX 4060 Laptop GPU
```

Recorded environment: `environment/conda-from-history.yml`, `environment/pip-freeze.txt`
(both lightly edited for reproducibility; the edits are documented in the files' header comments).

Recreate:
```bash
conda env create -f environment/conda-from-history.yml
conda activate mdf
TMPDIR=$HOME/.cache/pip-tmp pip install -r environment/pip-freeze.txt
```

### Problems encountered during installation

1. First torch install interrupted manually (Esc). Nothing was installed (`pip list` showed no torch/nvidia
   packages), but orphaned `pip-unpack-*` / `pip-*` temp dirs were left in `/tmp`.
2. Second attempt: the 916.9 MB torch wheel download failed at 547.9 MB after 6 attempts
   (`error: incomplete-download`, network). Retried with `--resume-retries 20`.
3. Third attempt: download completed, install failed with
   `OSError: [Errno 122] Disk quota exceeded`. Cause: `/tmp` is a tmpfs with a 6111M per-user quota;
   the orphaned pip temp dirs used 2202M of it and pip unpacks the wheel into `/tmp`.
   Fix: removed the orphaned `/tmp/pip-*` dirs (quota usage 2202M -> 184M) and set
   `TMPDIR=$HOME/.cache/pip-tmp` for all subsequent pip installs. Fourth attempt: `PIP_EXIT=0`.

## 4. Mamba smoke test: `scripts/check_env.py`

`python scripts/check_env.py --seed 0` (exit code 0). The script exits non-zero if `selective_scan_cuda` or
`causal_conv1d_cuda` is not importable, and counts calls into both extensions and into the fused
`mamba_inner_fn` during one forward+backward to report which path is actually taken.

```
python          3.12.15
torch           2.10.0+cu128
torch CUDA      12.8
cudnn           91002
cxx11 ABI       True
CUDA available  True
GPU             NVIDIA GeForce RTX 4060 Laptop GPU, sm_89, 7889 MiB
mamba_ssm       2.3.2.post1
causal_conv1d   1.7.0

Mamba block: d_model=128, d_state=16, d_conv=4, expand=2, params=116480, use_fast_path=True, seed=0
calls in one fwd+bwd: {'mamba_inner_fn': 1, 'selective_scan_cuda': 2, 'causal_conv1d_cuda': 3}
path: CUDA kernels (fused mamba_inner_fn -> selective_scan_cuda + causal_conv1d_cuda)

== Forward+backward, batch=10, fp32, 3 warmup + 20 timed iters ==
     L |   ms/iter |  peak MiB | finite
    64 |     0.995 |      25.7 | True
   256 |     1.047 |      51.2 | True
  1024 |     3.765 |     154.4 | True
OK
```

Peak MiB = `torch.cuda.max_memory_allocated` (tensor allocations only, excludes the CUDA context).
`causal_conv1d_cuda` is called 3 times: forward, recomputed forward in backward (`checkpoint_lvl=1`), backward.

GPU during check_env (nvidia-smi every 200 ms, 34 samples; the run is a few seconds and mostly imports,
so medians reflect idle): temperature 36-39 C, SM clock max 2625 MHz, memory clock max 8001 MHz,
power max 18 W, memory used max 324 MiB, pstates P0/P3/P8.

## 5. Dataset

```bash
mkdir -p ~/data/dfp && cd ~/data/dfp
curl -sSL --fail -o data_6k.tar.gz 'https://dataserv.ub.tum.de/s/m1470791/download?path=%2F&files=data_6k.tar.gz'
tar tzf data_6k.tar.gz          # 6493 entries, no absolute or '..' paths
tar xzf data_6k.tar.gz --no-same-owner
ln -s ~/data/dfp/data/train <repo>/data/train
ln -s ~/data/dfp/data/test  <repo>/data/test
```

| Item | Value |
|---|---|
| Archive size | 1318778020 bytes |
| Archive sha256 | 1ee811c32bf25aa1951cb8cd2b29bc211a49a1fbad602e3888a2b4f41e160a4d |
| `data/train` | 6400 files, all `.npz` (1.3G) |
| `data/test` | 90 files, all `.npz` (18M) |

Counts: `find -L data/train -type f -name '*.npz' | wc -l` and the same for `data/test`.
The symlinks and the training/test outputs are listed in `.gitignore`.

## 6. Original baseline smoke test (indicative)

> **Indicative only: single run, random seed** (the original `runTrain.py` draws a random seed and prints it;
> CLAUDE.md requires an explicit `--seed`, which the unmodified script does not support).
> Numbers below are not suitable for model comparison.

Measurement conditions:

| Item | Value |
|---|---|
| AC power | connected (`/sys/class/power_supply/ACAD/online` = 1), battery `Full`, 100% |
| Power profile | ACPI `platform_profile` = performance; `tuned-adm active` = throughput-performance; CPU governor `performance`, EPP `performance` (`powerprofilesctl` not installed) |
| GPU idle before runs | 37 C, SM 1890 MHz, mem 7001 MHz, P3, 13 MiB used; power limit `[N/A]` |
| GPU throttle reasons at idle | `SW Power Cap: Active`, `SW Thermal Slowdown: Active` (`nvidia-smi -q -d PERFORMANCE`) |

Note: nvidia-smi `power.draw` reported 590.01 W at idle (impossible for this laptop GPU, 65 W cap shown
in the Phase A `nvidia-smi` header). During training 2 of 240 samples read 590 W, both in pstate P3;
treat these power readings as sensor artefacts.

### Training: `train/runTrain.py` (unmodified)

```bash
cd train
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,memory.used \
  --format=csv,noheader,nounits -lms 500 > gpu_train.csv &
/usr/bin/time -v python -u runTrain.py 2>&1 | perl -MTime::HiRes=time -ne 'BEGIN{$|=1} printf "%.3f %s", time, $_' > train.log
```

| Metric | Value |
|---|---|
| Random seed (printed) | 1162998757 |
| Data loaded | 6400 (training batches 600, validation batches 40) |
| Trainable params | 2332035 |
| Epochs x batches = iterations | 17 x 600 = **10200** (script: `epochs = int(10000/600 + 0.5)`) |
| Throughput, incl. validation (epoch 1 start -> last epoch line, 99.2 s) | **102.8 it/s** |
| Throughput, train only (sum of per-epoch start->end, 95.8 s) | 106.5 it/s |
| Data load + setup | 20.4 s |
| Total wall clock (`/usr/bin/time`) | **2:01.81** |
| Peak VRAM (nvidia-smi `memory.used`, whole process incl. CUDA context; idle 13 MiB) | **442 MiB** |
| Peak RSS | 7908624 kB |
| Final L1 (last batch of epoch 17, single batch) | **0.0067697507329285145** |
| GPU during training (240 samples @ 500 ms) | temp 38/61/64 C (min/median/max), SM median 2550 MHz (max 2655), mem median 8001 MHz, power median 54 W |

Output: `train/modelG` (9365073 bytes), `train/results_train/` (34 files).

### Test: `train/runTest.py` (unmodified)

```bash
cd train && /usr/bin/time -v python -u runTest.py > test.log 2>&1   # exit 0
```

| Metric | Value |
|---|---|
| Test samples | **90** (90 "Test sample" lines in output and in `testout.txt`) |
| Relative error p / v / combined | **17.429916 % / 2.761411 % / 3.365578 %** |
| L1 error (normalized) | 0.006706 |
| Denormalized error | 0.015287 |
| Wall clock | 0:05.89 |

Note: in test mode `dataset.py` first loads only 10 training files (prints `Number of data loaded: 10`)
before replacing them with all files from `data/test`; the evaluation is on 90 samples.

No fixes were needed: both original scripts ran unmodified with torch 2.10 / numpy 2.5.3 / pillow 12.3.0.

## 7. Test dependency (2026-10-12)

`pytest` added for the unit tests of the mixing blocks (`tests/`):
`TMPDIR=$HOME/.cache/pip-tmp pip install pytest==9.1.1` inside `mdf`, after `pip install --dry-run pytest`
showed only `iniconfig-2.3.1 pluggy-1.6.0 pytest-9.1.1` would be installed (torch unchanged, 2.10.0+cu128).
`environment/pip-freeze.txt` updated with these three pins.
