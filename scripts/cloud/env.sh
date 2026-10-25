# Sourced by the scripts in scripts/cloud/: TMPDIR on the disk (docs/setup-log.md) and PY = python of the conda env "mdf"
# (~/miniforge3 on the VM, created by setup_vm.sh; ~/miniconda3 on the laptop; or `conda run`).
export TMPDIR="${TMPDIR_MDF:-$HOME/.cache/pip-tmp}"
mkdir -p "$TMPDIR"
if [ -z "${PY:-}" ]; then
    for c in "$HOME/miniforge3/envs/mdf/bin/python" "$HOME/miniconda3/envs/mdf/bin/python"; do
        if [ -x "$c" ]; then PY="$c"; break; fi
    done
fi
if [ -z "${PY:-}" ] && command -v conda >/dev/null; then
    PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
fi
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
    echo "ERROR: python of the conda env mdf not found (run scripts/cloud/setup_vm.sh first)" >&2
    exit 1
fi
export PY
