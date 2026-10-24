#!/usr/bin/env bash
# Environment on the cloud VM (docs/cloud-guide.md step 3): the same versions as the laptop (docs/setup-log.md):
# Python 3.12 (conda-forge, 3.12.15 on the laptop), torch 2.10.0+cu128, causal-conv1d 1.7.0 and mamba-ssm 2.3.2.post1
# from the same prebuilt GitHub wheels (sha256 pinned in environment/pip-freeze.txt), never built from source.
#
# Steps: GPU and driver check (nvidia-smi) -> system tools (git, tmux, curl; apt, with sudo, only if missing) ->
# Miniforge in ~/miniforge3 (if no conda) -> conda env mdf -> pip install -r environment/pip-freeze.txt (TMPDIR on disk,
# --resume-retries) -> version checks + pip freeze compared with the file -> scripts/check_env.py --seed 0 (fails unless
# the selective_scan_cuda / causal_conv1d_cuda kernels are used) -> the unit tests that need no data set (the tests that
# read data/train are skipped: tests/test_holdout.py, test_kfold.py, test_train_fraction.py).
# Re-runnable: finished steps are skipped. Log: ~/setup_vm.log.
#
# Usage: scripts/cloud/setup_vm.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
exec > >(tee -a "$HOME/setup_vm.log") 2>&1
echo "== setup_vm.sh $(date -Is), repo commit $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR"

echo "== 1. GPU"
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi does not work. Use a boot image with the NVIDIA driver (docs/cloud-guide.md); on a" >&2
    echo "Deep Learning VM image the driver is installed at the first login (answer 'y' when asked)." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$drv" -ge 570 ] || echo "WARNING: driver $drv < 570; torch cu128 is tested with >= 570 (CUDA 12.8)"
free -g | head -2
df -h "$HOME" | tail -1

echo "== 2. System tools"
missing=()
for t in git tmux curl bzip2; do command -v "$t" >/dev/null || missing+=("$t"); done
if [ ${#missing[@]} -gt 0 ]; then
    echo "Installing ${missing[*]} with apt (sudo)"
    sudo apt-get update -qq && sudo apt-get install -y -qq "${missing[@]}"
fi

echo "== 3. conda (Miniforge)"
if [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA="$HOME/miniforge3/bin/conda"
elif command -v conda >/dev/null; then
    CONDA="$(command -v conda)"
else
    curl -L --fail --retry 10 -o "$TMPDIR/Miniforge3.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$TMPDIR/Miniforge3.sh" -b -p "$HOME/miniforge3"
    rm -f "$TMPDIR/Miniforge3.sh"
    CONDA="$HOME/miniforge3/bin/conda"
fi
"$CONDA" --version

echo "== 4. env mdf"
if ! "$CONDA" env list | awk '{print $1}' | grep -qx mdf; then
    "$CONDA" create -y -n mdf --override-channels -c conda-forge python=3.12.15 pip \
        || "$CONDA" create -y -n mdf --override-channels -c conda-forge python=3.12 pip
fi
ENV_PREFIX="$("$CONDA" env list | awk '$1 == "mdf" {print $NF}')"
PY="$ENV_PREFIX/bin/python"
"$PY" --version

echo "== 5. pip packages (environment/pip-freeze.txt; torch from the cu128 index, wheels from GitHub releases)"
if "$PY" -c 'import torch, mamba_ssm, causal_conv1d' 2>/dev/null; then
    echo "already installed"
else
    "$PY" -m pip install --progress-bar off --resume-retries 20 -r environment/pip-freeze.txt
fi

echo "== 6. versions"
"$PY" - <<'PYEOF'
import platform, sys, torch, mamba_ssm, causal_conv1d
v = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
     "mamba_ssm": mamba_ssm.__version__, "causal_conv1d": causal_conv1d.__version__,
     "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(v)
want = {"torch": "2.10.0+cu128", "torch_cuda": "12.8", "mamba_ssm": "2.3.2.post1", "causal_conv1d": "1.7.0"}
bad = {k: (v[k], w) for k, w in want.items() if v[k] != w}
if not v["python"].startswith("3.12.") or bad or not v["cuda_available"]:
    sys.exit(f"ERROR: versions differ from docs/setup-log.md: {bad}, python {v['python']}, cuda {v['cuda_available']}")
print("versions OK" + ("" if v["python"] == "3.12.15" else f" (python {v['python']}, laptop 3.12.15)"))
PYEOF
"$PY" -m pip freeze | sort > "$HOME/pip-freeze-vm.txt"
grep -vE '^(#|--)' environment/pip-freeze.txt | sort > "$TMPDIR/pip-freeze-repo.txt"
if diff "$TMPDIR/pip-freeze-repo.txt" "$HOME/pip-freeze-vm.txt" > "$HOME/pip-freeze-diff.txt"; then
    echo "pip freeze identical to environment/pip-freeze.txt"
else
    echo "pip freeze differs from environment/pip-freeze.txt (see ~/pip-freeze-diff.txt):"
    head -20 "$HOME/pip-freeze-diff.txt"
fi

echo "== 7. scripts/check_env.py --seed 0"
"$PY" scripts/check_env.py --seed 0

echo "== 8. unit tests without the data set"
# Declared exclusion (docs/lab-notebook.md, 2026-10-19, fingerprint diagnosis): on the VM CPU the initial weights of the
# Mamba blocks (MD / MDv) differ from the laptop's in the last bits (fingerprint_report.py, tag experiments-final), so the
# bit-exact fingerprints of test_existing_variants_unchanged are deselected for MD8, MDv8, MD32, MDv32, MDv64, MD64
# only (FINGERPRINTS keys 4, 5, 7, 8, 9, 10); B, B-large, CBAM8, SA8, SA32 are still checked.
echo "NOTE: deselected on the VM: tests/test_multilevel.py::test_existing_variants_unchanged for MD/MDv (CPU-dependent last bits of the Mamba-block init; see the notebook)"
FP=tests/test_multilevel.py::test_existing_variants_unchanged
"$PY" -m pytest -q tests --ignore tests/test_holdout.py --ignore tests/test_kfold.py \
    --ignore tests/test_train_fraction.py \
    --deselect "$FP[key4]" --deselect "$FP[key5]" --deselect "$FP[key7]" \
    --deselect "$FP[key8]" --deselect "$FP[key9]" --deselect "$FP[key10]"

echo "== setup_vm.sh OK. Next: scripts/cloud/get_full_data.sh"
