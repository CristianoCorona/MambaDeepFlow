#!/usr/bin/env bash
# Training-budget calibration runs (docs/lab-notebook.md, "Pre-registered criterion for budget calibration").
# Runs sequentially, one process at a time: 40k (67 epochs) and 80k (133 epochs) requested iterations,
# seeds 0 and 1, each followed by evaluation on the validation split. The 10k runs already exist.
#
# Usage: nohup scripts/run_calibration.sh [OUT_ROOT] > OUT_ROOT/calibration.log 2>&1 &
#        OUT_ROOT defaults to ~/runs/calib
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${1:-$HOME/runs/calib}"
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR" "$OUT_ROOT"
cd "$REPO"

if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree is dirty; commit first" >&2
    exit 1
fi
PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
echo "repo $REPO  commit $(git rev-parse HEAD)  python $PY"

machine_state() {
    echo "--- machine state $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for f in /sys/class/power_supply/*/online; do echo "power_supply $(basename "$(dirname "$f")") online=$(cat "$f")"; done
    echo "platform_profile $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo n/a)"
    free -m | awk '/^Mem:/ {print "mem_available_mb " $7} /^Swap:/ {print "swap_used_mb " $3}'
    nvidia-smi --query-gpu=temperature.gpu,pstate,memory.used --format=csv,noheader
}

for spec in "40000 0" "40000 1" "80000 0" "80000 1"; do
    read -r iters seed <<< "$spec"
    epochs=$(( (iters * 10 / 600 + 5) / 10 ))  # int(iters / 600 + 0.5), as train.py
    id="ep${epochs}_s${seed}"
    echo "=== $id: --iterations $iters --seed $seed"
    machine_state
    /usr/bin/time -v "$PY" scripts/train.py --seed "$seed" --iterations "$iters" \
        --out-dir "$OUT_ROOT/$id" > "$OUT_ROOT/$id.log" 2>&1
    tail -n 30 "$OUT_ROOT/$id.log" | grep -E "Done:|Elapsed|Maximum resident" || true
    "$PY" scripts/evaluate.py --run-dir "$OUT_ROOT/$id" --split val > "$OUT_ROOT/$id.eval_val.log" 2>&1
    grep -A1 "original definition" "$OUT_ROOT/$id.eval_val.log" || true
done
machine_state
echo "=== all calibration runs finished"
