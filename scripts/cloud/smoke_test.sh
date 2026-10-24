#!/usr/bin/env bash
# Short check on the cloud VM before the big-dataset queue (docs/cloud-guide.md step 6; docs/preregistration-bigdata.md).
# 1. scripts/cloud/inspect_full.py on data/train_full (names, split sizes; with LOAD_RAM=1 also loads train + val with
#    the training loader and prints the peak RSS) -> $OUT/inspect.json; stops if it reports a problem.
# 2. B, MDv64-local and MDv64 (in this order), seed 0, split mode grouped-full, on a per-profile subset of the training
#    files (--train-fraction $FRACTION, default 0.1) for --total-steps $STEPS (default 600, about one epoch of the
#    subset; config.json step_budget.data_epochs gives the exact value), one validation pass on the FULL validation
#    set, then evaluate.py --split val. MDv64 is not in the minimum list of the session plan; it is added (one short
#    run) because the budget rule needs its it/s on this GPU.
# 3. GPU memory sampled with nvidia-smi every second ($OUT/gpu.csv); scripts/cloud/estimate_budget.py prints it/s,
#    memory and the estimated duration of a 133-epoch run on the full set per configuration, the queue total, and
#    (with HOURLY_USD) the cost, the runs that fit the budget and a suggested CUTOFF; writes $OUT/est.json.
# Outputs in $OUT (default ~/runs/bigdata-smoke), not in the repo. Re-running moves an existing smoke run aside.
#
# Usage: scripts/cloud/smoke_test.sh
#   Overridable: OUT, FRACTION (0.1), STEPS (600), LOAD_RAM (1), HOURLY_USD (unset), BUDGET_USD (50), SPENT_USD (0),
#                RESERVE_USD (5), ARCHIVE_TEST_NAMES (~/data/dfp_full/archive_test_names.txt, from get_full_data.sh)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-$HOME/runs/bigdata-smoke}"
FRACTION="${FRACTION:-0.1}"
STEPS="${STEPS:-600}"
LOAD_RAM="${LOAD_RAM:-1}"
ARCHIVE_TEST_NAMES="${ARCHIVE_TEST_NAMES:-$HOME/data/dfp_full/archive_test_names.txt}"
cd "$REPO"
# shellcheck source=scripts/cloud/env.sh
source scripts/cloud/env.sh

[ -e data/train_full ] || { echo "ERROR: data/train_full missing (run scripts/cloud/get_full_data.sh)" >&2; exit 1; }
mkdir -p "$OUT"
echo "== smoke test: commit $(git rev-parse --short HEAD), $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader), out $OUT"

INSPECT_ARGS=(--json "$OUT/inspect.json")
[ -f "$ARCHIVE_TEST_NAMES" ] && INSPECT_ARGS+=(--archive-test-names "$ARCHIVE_TEST_NAMES")
[ "$LOAD_RAM" = 1 ] && INSPECT_ARGS+=(--load-ram)
"$PY" scripts/cloud/inspect_full.py "${INSPECT_ARGS[@]}" | tee "$OUT/inspect.log"

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits \
    -l 1 > "$OUT/gpu.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

run_one() {  # name, extra train.py args
    local name="$1"; shift
    local d="$OUT/$name"
    if [ -e "$d" ]; then
        mv "$d" "$d.old-$(date +%Y%m%dT%H%M%S)"
    fi
    echo "== $name: training ($STEPS steps on fraction $FRACTION) ..."
    local t0 t1 t2
    t0=$(date +%s.%N)
    "$PY" scripts/train.py --seed 0 --split-mode grouped-full --train-fraction "$FRACTION" --total-steps "$STEPS" \
        --log-every "$STEPS" "$@" --out-dir "$d" > "$OUT/$name.log" 2>&1 \
        || { echo "ERROR: training of $name failed, see $OUT/$name.log" >&2; tail -n 20 "$OUT/$name.log" >&2; exit 1; }
    t1=$(date +%s.%N)
    "$PY" scripts/evaluate.py --run-dir "$d" --split val > "$OUT/$name.eval_val.log" 2>&1 \
        || { echo "ERROR: evaluation of $name failed, see $OUT/$name.eval_val.log" >&2; exit 1; }
    t2=$(date +%s.%N)
    "$PY" - "$d" "$t0" "$t1" "$t2" <<'PYEOF'
import json, sys
from pathlib import Path
d, t0, t1, t2 = Path(sys.argv[1]), *map(float, sys.argv[2:])
(d / "smoke_timing.json").write_text(json.dumps({"train_wall_s": t1 - t0, "eval_wall_s": t2 - t1}, indent=2) + "\n")
c = json.loads((d / "config.json").read_text())
e = json.loads((d / "eval_val.json").read_text())
print(f"   {c['variant']}: train {c['split']['n_train']} files ({c['step_budget']['data_epochs']:.2f} epochs), "
      f"val {e['n_samples']}; {c['it_per_s_train']:.1f} it/s; data load {c['timing_seconds']['data_load']:.0f} s; "
      f"peak RSS {c['peak_rss_mib']:.0f} MiB, VRAM {c['peak_vram']['max_memory_reserved_mib']:.0f} MiB; "
      f"val rel_vy {100 * e['means']['rel_vy']:.2f} % (after {c['effective_iterations']} steps only: not a result); "
      f"wall {t1 - t0:.0f} s + eval {t2 - t1:.0f} s")
PYEOF
}

run_one B
run_one MDv64-local --variant MDv-local --level 64
run_one MDv64 --variant MDv --level 64

kill $SMI 2>/dev/null || true
echo "== GPU memory used (nvidia-smi, whole GPU) max: $(cut -d, -f2 "$OUT/gpu.csv" | sort -n | tail -1) MiB"
EST_ARGS=(--smoke-root "$OUT" --inspect-json "$OUT/inspect.json" --est-json "$OUT/est.json"
          --budget-usd "${BUDGET_USD:-50}" --spent-usd "${SPENT_USD:-0}" --reserve-usd "${RESERVE_USD:-5}")
[ -n "${HOURLY_USD:-}" ] && EST_ARGS+=(--hourly-usd "$HOURLY_USD")
"$PY" scripts/cloud/estimate_budget.py "${EST_ARGS[@]}" | tee "$OUT/estimate.log"
echo "== smoke test done: $OUT (send $OUT/inspect.log and $OUT/estimate.log back with sync_results.sh)"
