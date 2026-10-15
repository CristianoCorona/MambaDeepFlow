#!/usr/bin/env bash
# Single test-set session for the MDv64 confirmatory test (Saturday 2026-10-17 morning; pre-registration d46b279,
# postponed by the amendment in docs/preregistration-night6b.md, commit a986ca1). ONE pass:
#   1. scripts/evaluate.py --split test (batch 1, one run at a time) on MDv64 s0-7 only (night5 s0-2, night7 s3-7);
#      B s0-7 were evaluated on test in the first pass (ca067a4) and are only read from the cache;
#   2. scripts/final_analysis.py --confirm-mdv64 on test (decision on rel_vy: CONFIRMED / NOT CONFIRMED), as in
#      scripts/test_session_night6.sh (which this replaces for MDv64) -> docs/results/test-confirm-mdv64.
# Runs missing or not finished are skipped (pre-registered: a failed run is repeated once, otherwise declared missing).
#
# Refuses: without the argument --go; on a dirty working tree; if any MDv64 run already has eval_test.json (single
# pass). Everything is logged to ~/runs/test-session-night7/session.log; results are committed right after.
#
# Usage: scripts/test_session_night7.sh --go
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="$HOME/runs"
LOGDIR="$R/test-session-night7"
cd "$REPO"

[ "${1:-}" = "--go" ] || { echo "Single pre-registered test pass: run with --go" >&2; exit 1; }
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is dirty; commit first:" >&2; git status --short >&2; exit 1
fi
PY="$(conda run -n mdf python -c 'import sys; print(sys.executable)')"
export TMPDIR="$HOME/.cache/pip-tmp"
mkdir -p "$TMPDIR" "$LOGDIR"

NEW_RUNS=()
for d in "$R"/night5/MDv64_s{0,1,2} "$R"/night7/MDv64_s{3,4,5,6,7}; do
    if [ -f "$d/config.json" ] && grep -q '"status": "finished"' "$d/config.json"; then
        NEW_RUNS+=("$d")
    else
        echo "skipped (missing or not finished): $d" | tee -a "$LOGDIR/session.log"
    fi
done
for d in "${NEW_RUNS[@]}"; do
    if [ -e "$d/eval_test.json" ]; then echo "ERROR: $d already has eval_test.json (single pass)" >&2; exit 1; fi
done

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGDIR/session.log"; }
log "=== test session start, commit $(git rev-parse HEAD), ${#NEW_RUNS[@]} new runs"
for d in "${NEW_RUNS[@]}"; do
    log "evaluate.py --split test $d"
    "$PY" scripts/evaluate.py --run-dir "$d" --split test > "$LOGDIR/$(basename "$(dirname "$d")")_$(basename "$d").eval_test.log" 2>&1
    log "  done: $(grep -A1 'Relative error % (original' "$LOGDIR/$(basename "$(dirname "$d")")_$(basename "$d").eval_test.log" | tail -1)"
done

log "final_analysis.py --confirm-mdv64 --split test"
"$PY" scripts/final_analysis.py --config "B=$R/night1/ep133_s*" --config "B=$R/night3/B_s*" \
    --config "MDv64=$R/night5/MDv64_s*" --config "MDv64=$R/night7/MDv64_s*" --seeds 0-7 \
    --split test --allow-test --confirm-mdv64 \
    --out-dir docs/results/test-confirm-mdv64 --fig-dir docs/figures/test-confirm-mdv64 2>&1 | tee "$LOGDIR/confirm.log"
log "=== test session end"
