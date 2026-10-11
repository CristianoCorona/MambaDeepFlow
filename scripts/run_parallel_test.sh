#!/usr/bin/env bash
# Parallelism test: two scripts/train.py processes at the same time (seeds 0 and 1, 10k requested
# iterations = 17 epochs), to compare with the sequential runs new_s0_a / new_s1 (docs/lab-notebook.md).
# Skips the test (exit 0, message in the log) if MemAvailable < MIN_AVAILABLE_MB before the start.
#
# Usage: nohup scripts/run_parallel_test.sh [OUT_ROOT] > OUT_ROOT/parallel.log 2>&1
#        OUT_ROOT defaults to ~/runs/par
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${1:-$HOME/runs/par}"
MIN_AVAILABLE_MB=9000
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR" "$OUT_ROOT"
cd "$REPO"

if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree is dirty; commit first" >&2
    exit 1
fi
PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
echo "repo $REPO  commit $(git rev-parse HEAD)  python $PY"
echo "--- machine state $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for f in /sys/class/power_supply/*/online; do echo "power_supply $(basename "$(dirname "$f")") online=$(cat "$f")"; done
echo "platform_profile $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo n/a)"
avail=$(free -m | awk '/^Mem:/ {print $7}')
swap=$(free -m | awk '/^Swap:/ {print $3}')
echo "mem_available_mb $avail"
echo "swap_used_mb $swap"
nvidia-smi --query-gpu=temperature.gpu,pstate,memory.used --format=csv,noheader
if [ "$avail" -lt "$MIN_AVAILABLE_MB" ]; then
    echo "SKIPPED: not run, insufficient RAM (available = $avail MB < $MIN_AVAILABLE_MB MB)"
    exit 0
fi

# Memory sampler: MemAvailable and SwapFree (kB) once per second while the runs are alive
sampler() {
    while true; do
        echo "$(date +%s) $(awk '/^MemAvailable:/ {a=$2} /^SwapFree:/ {s=$2} END {print a, s}' /proc/meminfo)"
        sleep 1
    done
}
sampler > "$OUT_ROOT/meminfo.txt" &
sampler_pid=$!

t0=$(date +%s.%N)
pids=()
for seed in 0 1; do
    /usr/bin/time -v "$PY" scripts/train.py --seed "$seed" --iterations 10000 \
        --out-dir "$OUT_ROOT/par_s$seed" > "$OUT_ROOT/par_s$seed.log" 2>&1 &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
t1=$(date +%s.%N)
kill "$sampler_pid"
echo "both runs finished, status $status, total wall $(echo "$t1 - $t0" | bc) s"
[ "$status" -eq 0 ] || exit 1

for seed in 0 1; do
    tail -n 30 "$OUT_ROOT/par_s$seed.log" | grep -E "Done:|Elapsed|Maximum resident" || true
    "$PY" scripts/evaluate.py --run-dir "$OUT_ROOT/par_s$seed" --split val > "$OUT_ROOT/par_s$seed.eval_val.log" 2>&1
    grep -A1 "original definition" "$OUT_ROOT/par_s$seed.eval_val.log" || true
done
awk 'NR == 1 {a0 = $2; s0 = $3} {if (min_a == "" || $2 < min_a) min_a = $2; if (min_s == "" || $3 < min_s) min_s = $3}
     END {printf "MemAvailable at start %.0f MB, min %.0f MB, max drop %.0f MB; SwapFree drop %.0f MB\n",
          a0/1024, min_a/1024, (a0-min_a)/1024, (s0-min_s)/1024}' "$OUT_ROOT/meminfo.txt"
echo "=== parallel test finished"
